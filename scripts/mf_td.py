# -*- coding: utf-8 -*-
"""Extensão small/mid caps — FASE MF-2: papéis + contagens LM + Tone Distance.

Papéis (deterministico; o formato Motley Fool marca analistas explicitamente):
  1. "Operator" -> operador;
  2. cabeçalho termina em "Analyst" -> analista;
  3. título executivo na afiliação (regex validada no S&P) -> gestor;
  4. residual: falou no Prepared Remarks -> gestor; só Q&A -> analista
     (fallback conservador, o mesmo do pipeline S&P).
Identidade do gestor: mkey (sobrenome|inicial) — convenção do pipeline S&P.

TD por call (>=2 gestores, transcrição inteira, frações de palavras LM):
  td_a  = leitura A (centro = média simples das frações por gestor);
  td_b  = leitura B (centro = fração agregada — "Average Transcript Tone");
  td_w  = ponderada pelo volume de palavras (Tabela 8 col 1 do artigo).
Controles textuais: disclosure_tone, analyst_tone, analyst_tone_disp (sd),
length (ln palavras total), lagged por firma (série própria), n_managers.

Validação impressa: cobertura de papéis, distribuição da TD vs paper/S&P,
corr entre construções. Saída: data/interim/mf/tone_distance_mf.parquet
Uso: python scripts/mf_td.py
"""
from __future__ import annotations
import re, unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MF = ROOT / "data" / "interim" / "mf"
RAW = ROOT / "data" / "raw"
TOKEN = re.compile(r"[A-Za-z]+")
TITLE = re.compile(r"\b(chief|officer|ceo|cfo|coo|cto|president|vice president|vp|chairman|"
                   r"founder|treasurer|controller|general counsel|head of|head,|director|"
                   r"investor relations|executive|secretary|principal accounting)\b", re.I)

_QAp = re.compile(r"^[QA]\s*[-–:]\s*"); _TS = re.compile(r"\s+[-–—]\s+.*$"); _CM = re.compile(r"(?<=[a-z])(?=[A-Z])")
def _clean(n):
    if not isinstance(n, str): return ""
    n = _QAp.sub("", n.strip()); n = _TS.sub("", n); n = _CM.sub(" ", n)
    n = unicodedata.normalize("NFKD", n); n = "".join(c for c in n if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", n).strip().casefold()
def mkey(n):
    c = _clean(n)
    if not c: return ""
    t = [x for x in c.replace(".", " ").split() if len(x) > 1]
    return f"{t[-1]}|{t[0][0]}" if t else c


def td_from(fp, fn, w=None, center="mean", weight_dist=False):
    """center: 'mean' (leitura A) ou 'pooled' (leitura B, exige w).
    weight_dist: True pondera as DISTÂNCIAS por w (Tabela 8 col 1)."""
    if center == "pooled":
        cp = (fp * w).sum() / w.sum()
        cn = (fn * w).sum() / w.sum()
    else:
        cp, cn = fp.mean(), fn.mean()
    d = np.sqrt((fp - cp) ** 2 + (fn - cn) ** 2)
    if weight_dist:
        return float((d * w).sum() / w.sum())
    return float(d.mean())


def main():
    lmd = pd.read_csv(RAW / "LM_MasterDictionary.csv")
    POS = set(lmd.loc[lmd["Positive"] != 0, "Word"].str.upper())
    NEG = set(lmd.loc[lmd["Negative"] != 0, "Word"].str.upper())

    um = pd.read_parquet(MF / "utterances_mf.parquet")
    # ---------- papéis ----------
    sp = um["speaker"].astype(str).str.strip()
    is_op = sp.str.fullmatch("Operator", case=False)
    ends_an = sp.str.lower().str.rstrip().str.endswith("analyst")
    aff = sp.str.split("--").str[1:].str.join(" ")
    has_title = aff.str.contains(TITLE, regex=True, na=False)
    role = np.where(is_op, "operator",
                    np.where(ends_an, "analyst",
                             np.where(has_title, "manager", "resid")))
    um["role"] = role
    # residual: presença em remarks NA MESMA call -> gestor; senão analista
    rem_spk = set(map(tuple, um.loc[(um["section"] == "remarks") & (um["role"] == "resid"),
                                    ["call_id", "speaker"]].drop_duplicates().to_numpy()))
    mask_res = um["role"] == "resid"
    um.loc[mask_res, "role"] = [
        "manager" if (c, s) in rem_spk else "analyst"
        for c, s in um.loc[mask_res, ["call_id", "speaker"]].to_numpy()]
    print("papéis (por fala):")
    print(um["role"].value_counts().to_string())
    print(f"residuais resolvidos por seção: {int(mask_res.sum())} falas", flush=True)

    # ---------- contagens LM ----------
    nw = np.empty(len(um), dtype=np.int32)
    npos = np.empty(len(um), dtype=np.int32)
    nneg = np.empty(len(um), dtype=np.int32)
    for j, txt in enumerate(um["text"].astype(str)):
        toks = [w.upper() for w in TOKEN.findall(txt)]
        nw[j] = len(toks)
        npos[j] = sum(1 for w in toks if w in POS)
        nneg[j] = sum(1 for w in toks if w in NEG)
    um["nw"], um["npos"], um["nneg"] = nw, npos, nneg
    um["mk"] = sp.map(mkey)
    um[["call_id", "utterance_idx", "mk", "role", "section", "nw", "npos", "nneg"]].to_parquet(
        MF / "speaker_counts_mf.parquet", index=False)

    # ---------- TD por call ----------
    cm = pd.read_parquet(MF / "calls_mf.parquet")
    rows = []
    for cid, g in um.groupby("call_id", sort=False):
        mg = g[(g["role"] == "manager") & (g["mk"] != "")]
        an = g[g["role"] == "analyst"]
        tot_w = int(g.loc[g["role"].isin(["manager", "analyst"]), "nw"].sum())
        per = mg.groupby("mk").agg(nw=("nw", "sum"), npos=("npos", "sum"), nneg=("nneg", "sum"))
        per = per[per["nw"] > 0]
        if len(per) < 2 or tot_w == 0:
            continue
        fp = (per["npos"] / per["nw"]).to_numpy(float)
        fn = (per["nneg"] / per["nw"]).to_numpy(float)
        w = per["nw"].to_numpy(float)
        mw, mp, mn = float(w.sum()), float(per["npos"].sum()), float(per["nneg"].sum())
        aw, ap_, an_ = float(an["nw"].sum()), float(an["npos"].sum()), float(an["nneg"].sum())
        pa = an[an["mk"] != ""].groupby("mk").agg(nw=("nw", "sum"), npos=("npos", "sum"),
                                                  nneg=("nneg", "sum"))
        pa = pa[pa["nw"] > 0]
        atd = float(((pa["npos"] - pa["nneg"]) / pa["nw"]).std()) if len(pa) >= 2 else np.nan
        rows.append(dict(
            call_id=cid,
            td_a=td_from(fp, fn),
            td_b=td_from(fp, fn, w=w, center="pooled"),
            td_w=td_from(fp, fn, w=w, center="mean", weight_dist=True),
            n_managers=len(per), min_words=int(per["nw"].min()),
            disclosure_tone=(mp - mn) / mw if mw > 0 else np.nan,
            analyst_tone=(ap_ - an_) / aw if aw > 0 else np.nan,
            analyst_tone_disp=atd, length=float(np.log(tot_w))))
    out = pd.DataFrame(rows).merge(cm, on="call_id", how="left")
    out = out.sort_values(["ticker", "call_datetime"])
    g = out.groupby("ticker")
    out["lagged_td_a"] = g["td_a"].shift(1)
    out["lagged_td_w"] = g["td_w"].shift(1)
    out.to_parquet(MF / "tone_distance_mf.parquet", index=False)

    print(f"\ncalls com TD: {len(out)} | tickers: {out['ticker'].nunique()}", flush=True)
    for c in ["td_a", "td_b", "td_w"]:
        s = out[c]
        print(f"{c}: média {s.mean():.4f} mediana {s.median():.4f} sd {s.std():.4f}")
    print("(paper T1: 0.0079/0.0074/0.0048 | nosso S&P: 0.0086/0.0080/0.0047)")
    print(f"corr(td_a, td_w) = {out['td_a'].corr(out['td_w']):+.3f} | "
          f"corr(td_a, td_b) = {out['td_a'].corr(out['td_b']):+.3f}")
    print(f"gestores por call: mediana {out['n_managers'].median():.0f}")
    print("DONE.")


if __name__ == "__main__":
    main()
