# -*- coding: utf-8 -*-
"""Replicação S&P 500 — FASE A: transcrições -> Tone Distance LM + controles textuais.

Processa as ~33k calls do dataset COMPLETO (todos os setores; cache HF local):
  1. Reconstrói falas por orador (maquinaria do repo) em blocos de 2.000 calls
     (streaming: o texto é descartado após a contagem LM — memória bounded).
  2. Papéis: heurística ADR-006 (seções) + classificação de orador em 2 níveis:
     cache do LLM (Stage 1) quando o header é conhecido; senão classificador
     DETERMINÍSTICO validado (cargo/firma/n_tickers/remarks; 99% de concordância
     com o LLM nos headers frequentes).
  3. Tone Distance fiel (Eq.1 Angelo, frações de PALAVRAS LM oficiais) por call
     + permutation null (200x, seed 42) -> td, td_adj, z.
  4. Controles textuais: disclosure_tone (net tone LM da gestão), analyst_tone,
     analyst_tone_distance (>=2 analistas), length (log palavras), lagged_td
     (TD anterior da MESMA empresa — do sensor NOVO), n_managers, min_words.

Saída: data/interim/sp500/tone_distance_sp500.parquet (uma linha por call).
Uso: python scripts/sp500_phase_a.py [n_perm=200] [chunk=2000]
"""
from __future__ import annotations
import json, re, sys, time, unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonediv.config import load_config
from tonediv.data import transcripts as tr
from tonediv.nlp.roles import infer_roles

INT, RAW = ROOT / "data" / "interim", ROOT / "data" / "raw"
OUTDIR = INT / "sp500"
N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
SEED, MIN_MGR = 42, 2
TOKEN = re.compile(r"[A-Za-z]+")

# ---------- identidade (roles._match_key) ----------
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

# ---------- classificador determinístico de orador (stage1, validado) ----------
TITLE = re.compile(r"\b(chief|officer|ceo|cfo|coo|cto|president|vice president|vp|chairman|"
                   r"founder|treasurer|controller|general counsel|head of|director of|"
                   r"investor relations|executive|principal accounting)\b", re.I)
FIRM = re.compile(r"\b(securities|capital|partners|research|brokerage|& co|llc|l\.l\.c|"
                  r"morgan|goldman|sachs|jpmorgan|j\.p\. morgan|merrill|barclays|credit suisse|"
                  r"ubs|deutsche|citigroup|citi|wells fargo|jefferies|baird|cowen|piper|wedbush|"
                  r"raymond james|stifel|oppenheimer|needham|canaccord|bernstein|evercore|mizuho|"
                  r"nomura|rbc|bmo|keybanc|susquehanna|william blair|cantor|guggenheim|truist|"
                  r"loop capital|rosenblatt|davidson|macquarie|scotiabank|bank of america|bofa|"
                  r"new street|wolfe|redburn|northland|roth|craig|benchmark|mkm|argus|moffett|"
                  r"arete|melius|tigress|sanford|bnp|hsbc|prudential|bear stearns|wachovia)\b", re.I)
GROUP = re.compile(r"^(executives?|analysts?|unidentified|company representative|"
                   r"conference call participants?|multiple speakers?|participants?)", re.I)
QP = re.compile(r"^Q\s*[-–:]"); AP = re.compile(r"^A\s*[-–:]")

def org_part(h):
    parts = re.split(r"\s*[-–—,/]\s*", h, maxsplit=1)
    return parts[1] if len(parts) > 1 else ""

def det_classify(h, n_tk, n_rem_clean):
    if h.lower() == "operator": return "operator"
    if GROUP.match(h): return "group"
    if QP.match(h): return "analyst"
    if AP.match(h): return "manager"
    if TITLE.search(h): return "manager"
    if FIRM.search(org_part(h)): return "analyst"
    if n_tk >= 4: return "analyst"
    if n_tk <= 2 and n_rem_clean >= 1: return "manager"
    if n_rem_clean >= 1: return "manager"
    # resíduo só-Q&A: fallback conservador = analista (estreia no Q&A)
    return "analyst"


def td_points(fp, fn):
    return float(np.sqrt((fp - fp.mean()) ** 2 + (fn - fn.mean()) ** 2).mean())


def main():
    t00 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "config.yaml"))

    lmd = pd.read_csv(RAW / "LM_MasterDictionary.csv")
    POS = set(lmd.loc[lmd["Positive"] != 0, "Word"].str.upper())
    NEG = set(lmd.loc[lmd["Negative"] != 0, "Word"].str.upper())
    print(f"LM: {len(POS)} pos, {len(NEG)} neg", flush=True)

    llm_map = {}
    cache = INT / "llm_roles_cache.jsonl"
    if cache.exists():
        for ln in cache.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(ln); llm_map[o["h"]] = o["role"]
            except Exception:
                pass
    print(f"cache LLM: {len(llm_map)} headers conhecidos", flush=True)

    print("carregando HF (cache local)...", flush=True)
    calls = tr.prepare_calls(tr.load_hf_transcripts(cfg), cfg)
    print(f"calls normalizadas: {len(calls)} | tickers: {calls['ticker'].nunique()} "
          f"({time.time()-t00:.0f}s)", flush=True)

    # ---------- PASSO 1: streaming por blocos (texto -> contagens; texto morre aqui) ----------
    reduced = []
    q = cfg.quality
    for i0 in range(0, len(calls), CHUNK):
        chunk = calls.iloc[i0:i0 + CHUNK]
        utts = tr.build_utterances(chunk)
        meta = tr.calls_metadata(chunk, utts)
        keep_ids = set(meta.loc[(meta["n_chars_total"] >= q.min_chars_call)
                                & (meta["n_utterances"] >= q.min_utterances_call), "call_id"])
        utts = utts[utts["call_id"].isin(keep_ids)].reset_index(drop=True)
        if utts.empty:
            continue
        ann = infer_roles(utts, cfg.roles)
        nw = np.empty(len(ann), dtype=np.int32)
        npos = np.empty(len(ann), dtype=np.int32)
        nneg = np.empty(len(ann), dtype=np.int32)
        for j, txt in enumerate(ann["text"].astype(str)):
            toks = [w.upper() for w in TOKEN.findall(txt)]
            nw[j] = len(toks)
            npos[j] = sum(1 for w in toks if w in POS)
            nneg[j] = sum(1 for w in toks if w in NEG)
        red = ann[["call_id", "speaker", "section", "role"]].copy()
        red["sp"] = red["speaker"].astype(str).str.strip()
        red["nw"], red["npos"], red["nneg"] = nw, npos, nneg
        red = red.drop(columns=["speaker"])
        reduced.append(red)
        print(f"  bloco {i0//CHUNK+1}/{(len(calls)+CHUNK-1)//CHUNK}: {len(chunk)} calls -> "
              f"{len(red)} falas ({time.time()-t00:.0f}s)", flush=True)
    df = pd.concat(reduced, ignore_index=True)
    del reduced
    df["tk"] = df["call_id"].astype(str).str.split("_").str[0]
    print(f"falas totais (pós-qualidade): {len(df)} | calls: {df['call_id'].nunique()}", flush=True)

    # ---------- PASSO 2: papéis globais ----------
    qa_calls = set(df.loc[df["section"] == "qa", "call_id"].unique())
    df["rem_clean"] = (df["section"] == "remarks") & df["call_id"].isin(qa_calls)
    stats = df.groupby("sp").agg(n_tk=("tk", "nunique"), n_rem_clean=("rem_clean", "sum"))
    role_map = {}
    n_llm = 0
    for sp, row in stats.iterrows():
        r = llm_map.get(sp)
        if r is not None:
            n_llm += 1
        else:
            r = det_classify(sp, int(row["n_tk"]), int(row["n_rem_clean"]))
        role_map[sp] = r
    df["role_eff"] = df["sp"].map(role_map)
    print(f"papéis: {n_llm} headers via LLM, {len(role_map)-n_llm} via determinístico", flush=True)
    print(df.groupby("role_eff")["nw"].sum().to_string(), flush=True)

    # ---------- PASSO 3: TD + null + controles por call ----------
    df["mk"] = df["sp"].map(mkey)
    cmeta = pd.read_parquet(RAW / "calls_all.parquet")[["call_id", "ticker", "call_datetime", "year", "quarter"]]
    rng = np.random.default_rng(SEED)
    rows = []
    t1 = time.time()
    for k, (cid, g) in enumerate(df.groupby("call_id", sort=True)):
        mg = g[(g["role_eff"] == "manager") & (g["mk"] != "")]
        an = g[g["role_eff"] == "analyst"]
        tot_w = int(g.loc[g["role_eff"].isin(["manager", "analyst"]), "nw"].sum())
        # gestão
        per = mg.groupby("mk").agg(nw=("nw", "sum"), npos=("npos", "sum"),
                                   nneg=("nneg", "sum"), n_utt=("nw", "size"))
        per = per[per["nw"] > 0]
        if len(per) < MIN_MGR or tot_w == 0:
            continue
        fp = (per["npos"] / per["nw"]).to_numpy(float)
        fn = (per["nneg"] / per["nw"]).to_numpy(float)
        td = td_points(fp, fn)
        mw, mp, mn = float(per["nw"].sum()), float(per["npos"].sum()), float(per["nneg"].sum())
        disclosure = (mp - mn) / mw if mw > 0 else np.nan
        # analistas
        aw, ap_, an_ = float(an["nw"].sum()), float(an["npos"].sum()), float(an["nneg"].sum())
        analyst_tone = (ap_ - an_) / aw if aw > 0 else np.nan
        pa = an[an["mk"] != ""].groupby("mk").agg(nw=("nw", "sum"), npos=("npos", "sum"), nneg=("nneg", "sum"))
        pa = pa[pa["nw"] > 0]
        if len(pa) >= 2:
            atd = td_points((pa["npos"] / pa["nw"]).to_numpy(float), (pa["nneg"] / pa["nw"]).to_numpy(float))
        else:
            atd = np.nan
        # null da TD (embaralha falas entre gestores preservando contagem)
        cnt = per["n_utt"].to_numpy()
        starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        gm = mg.sort_values("mk")
        uw = gm["nw"].to_numpy(float); up = gm["npos"].to_numpy(float); un = gm["nneg"].to_numpy(float)
        N = len(uw)
        tds = np.empty(N_PERM)
        for b in range(N_PERM):
            perm = rng.permutation(N)
            ws = np.add.reduceat(uw[perm], starts)
            with np.errstate(invalid="ignore", divide="ignore"):
                fpb = np.add.reduceat(up[perm], starts) / ws
                fnb = np.add.reduceat(un[perm], starts) / ws
            ok = ws > 0
            tds[b] = td_points(fpb[ok], fnb[ok]) if ok.sum() >= MIN_MGR else np.nan
        e0, s0 = float(np.nanmean(tds)), float(np.nanstd(tds))
        rows.append(dict(call_id=cid, td=td, e_td_null=e0, sd_td_null=s0, td_adj=td - e0,
                         z=(td - e0) / s0 if s0 > 1e-12 else np.nan,
                         n_managers=len(per), min_words=int(per["nw"].min()),
                         disclosure_tone=disclosure, analyst_tone=analyst_tone,
                         analyst_tone_distance=atd, length=float(np.log(tot_w))))
        if (k + 1) % 3000 == 0:
            print(f"  TD: {k+1} calls ({time.time()-t1:.0f}s)", flush=True)
    out = pd.DataFrame(rows).merge(cmeta, on="call_id", how="left")
    out = out.sort_values(["ticker", "call_datetime"])
    out["lagged_td"] = out.groupby("ticker")["td"].shift(1)
    dest = OUTDIR / "tone_distance_sp500.parquet"
    out.to_parquet(dest, index=False)

    s = out["td"]
    print(f"\nEscrito {dest}: {len(out)} calls, {out['ticker'].nunique()} tickers", flush=True)
    print(f"TD_LM S&P500: mean={s.mean():.4f} median={s.median():.4f} sd={s.std():.4f} "
          f"CV={s.std()/s.mean():.3f}  (paper: 0.0079/0.0074/0.0048/0.61)", flush=True)
    print(f"corr(td, min_words)={out['td'].corr(out['min_words']):+.3f} | "
          f"corr(z, min_words)={out['z'].corr(out['min_words']):+.3f}", flush=True)
    print(f"FASE A concluída em {(time.time()-t00)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
