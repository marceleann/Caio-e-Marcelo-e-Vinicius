# -*- coding: utf-8 -*-
"""Baseline Loughran-McDonald — réplica FIEL da Eq.(1) do Angelo (2025).

Teste decisivo (2026-07-10): nosso FinBERT sentence-level pode estar atrapalhando
em vez de ajudar. Aqui medimos o tom EXATAMENTE como o Angelo: fração de PALAVRAS
positivas/negativas (listas LM oficiais, 354 pos / 2.355 neg, sraf.nd.edu) no
texto INTEGRAL de cada gestor; cada gestor = ponto (Pos, Neg); TD = média das
distâncias euclidianas ao centroide. Sem filtro de mínimo de palavras (fiel);
>=2 gestores; papéis do Stage 1 (LLM). Por robustez, também o permutation null
(embaralha falas entre gestores preservando a contagem) -> td_adj e z.

Comparação-chave: a distribuição da TD_LM vs o paper (mean 0.0079, median 0.0074,
sd 0.0048, CV 0.61). Se bater, a construção está certa e a regressão (Stage 4)
responde se o EFEITO existe na nossa amostra.

Saída: data/interim/tone_distance_lm.parquet
Uso: python scripts/stage3_lm_baseline.py [n_perm=200]
"""
from __future__ import annotations
import re, sys, time, unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INT, RAW = ROOT / "data" / "interim", ROOT / "data" / "raw"
N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SEED, MIN_MGR = 42, 2
TOKEN = re.compile(r"[A-Za-z]+")

# identidade do gestor (mesma lógica de roles._match_key)
_QA = re.compile(r"^[QA]\s*[-–:]\s*"); _TS = re.compile(r"\s+[-–—]\s+.*$"); _CM = re.compile(r"(?<=[a-z])(?=[A-Z])")
def _clean(n):
    if not isinstance(n, str): return ""
    n = _QA.sub("", n.strip()); n = _TS.sub("", n); n = _CM.sub(" ", n)
    n = unicodedata.normalize("NFKD", n); n = "".join(c for c in n if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", n).strip().casefold()
def mkey(n):
    c = _clean(n)
    if not c: return ""
    t = [x for x in c.replace(".", " ").split() if len(x) > 1]
    return f"{t[-1]}|{t[0][0]}" if t else c

def td_points(fp, fn):
    return float(np.sqrt((fp - fp.mean()) ** 2 + (fn - fn.mean()) ** 2).mean())


def main():
    lm = pd.read_csv(RAW / "LM_MasterDictionary.csv")
    POS = set(lm.loc[lm["Positive"] != 0, "Word"].str.upper())
    NEG = set(lm.loc[lm["Negative"] != 0, "Word"].str.upper())
    print(f"LM oficial: {len(POS)} positivas, {len(NEG)} negativas")

    r = pd.read_parquet(INT / "utterances_roles.parquet")
    roles = pd.read_parquet(INT / "speaker_roles_llm.parquet").set_index("speaker")["llm_role"].to_dict()
    heur_map = {"management": "manager", "analyst": "analyst", "operator": "operator", "unknown": "analyst"}
    r = r.assign(sp=r["speaker"].astype(str).str.strip())
    r["role_eff"] = r["sp"].map(roles).fillna(r["role"].map(heur_map))
    mg = r[r["role_eff"] == "manager"].copy()
    mg["mk"] = mg["sp"].map(mkey)
    mg = mg[mg["mk"] != ""]
    print(f"falas de gestão (papéis LLM): {len(mg)}")

    # contagem LM por fala (texto integral, remarks + Q&A)
    t0 = time.time()
    nw = np.empty(len(mg), dtype=np.int64)
    npos = np.empty(len(mg), dtype=np.int64)
    nneg = np.empty(len(mg), dtype=np.int64)
    for i, txt in enumerate(mg["text"].astype(str)):
        toks = [w.upper() for w in TOKEN.findall(txt)]
        nw[i] = len(toks)
        npos[i] = sum(1 for w in toks if w in POS)
        nneg[i] = sum(1 for w in toks if w in NEG)
    mg["nw"], mg["npos"], mg["nneg"] = nw, npos, nneg
    print(f"contagem LM: {time.time()-t0:.0f}s | palavras totais: {nw.sum():,}")

    rng = np.random.default_rng(SEED)
    rows = []
    for cid, g in mg.groupby("call_id", sort=True):
        per = g.groupby("mk").agg(nw=("nw", "sum"), npos=("npos", "sum"), nneg=("nneg", "sum"),
                                  n_utt=("nw", "size"))
        per = per[per["nw"] > 0]
        if len(per) < MIN_MGR:
            continue
        fp = (per["npos"] / per["nw"]).to_numpy(dtype=np.float64)
        fn = (per["nneg"] / per["nw"]).to_numpy(dtype=np.float64)
        td = td_points(fp, fn)
        # null por FALAS: redistribui falas entre gestores preservando contagem
        cnt = per["n_utt"].to_numpy()
        starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        uw = g.sort_values("mk")["nw"].to_numpy(dtype=np.float64)  # ordem estável por mk
        up = g.sort_values("mk")["npos"].to_numpy(dtype=np.float64)
        un = g.sort_values("mk")["nneg"].to_numpy(dtype=np.float64)
        N = len(uw)
        tds = np.empty(N_PERM)
        for b in range(N_PERM):
            perm = rng.permutation(N)
            w_s = np.add.reduceat(uw[perm], starts)
            with np.errstate(invalid="ignore", divide="ignore"):
                fp_b = np.add.reduceat(up[perm], starts) / w_s
                fn_b = np.add.reduceat(un[perm], starts) / w_s
            ok = w_s > 0
            tds[b] = td_points(fp_b[ok], fn_b[ok]) if ok.sum() >= MIN_MGR else np.nan
        e_null, sd_null = float(np.nanmean(tds)), float(np.nanstd(tds))
        rows.append(dict(call_id=cid, n_managers=len(per), min_sent=int(per["nw"].min()),
                         avg_pos=float(fp.mean()), td=td,
                         e_td_null=e_null, sd_td_null=sd_null, td_adj=td - e_null,
                         z=(td - e_null) / sd_null if sd_null > 1e-12 else np.nan))
    out = pd.DataFrame(rows)
    out.to_parquet(INT / "tone_distance_lm.parquet", index=False)
    print(f"\nEscrito tone_distance_lm.parquet: {len(out)} calls")
    s = out["td"]
    print("\n== TONE DISTANCE LM (réplica fiel Angelo, palavras) ==")
    print(f"  mean={s.mean():.4f} median={s.median():.4f} sd={s.std():.4f} CV={s.std()/s.mean():.3f}")
    print("  paper Angelo:  mean=0.0079 median=0.0074 sd=0.0048 CV=0.61")
    print(f"\n  corr(TD_LM, palavras do gestor mais quieto): {out['td'].corr(out['min_sent']):+.3f}")
    print(f"  corr(z_LM , palavras do gestor mais quieto): {out['z'].corr(out['min_sent']):+.3f}")
    print(f"  E[null]/TD: {out['e_td_null'].mean()/s.mean():.2f}")


if __name__ == "__main__":
    main()
