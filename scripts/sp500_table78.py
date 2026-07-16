# -*- coding: utf-8 -*-
"""Tabelas 7 e 8 do Angelo (2025) — variantes de TD, do nível de fala.

TABELA 7 (Adjusted TD): "subtrai a contagem de palavras de tom positivas e
negativas usadas pelos analistas na PERGUNTA das contagens usadas pelos
gestores na RESPOSTA". Implementação: no Q&A, resposta = falas de gestores
até a próxima fala de analista; a subtração é alocada proporcionalmente às
palavras de cada gestor no bloco de resposta (decisão declarada; o paper não
especifica multi-gestor). Apresentação intocada. Contagens podem ficar
negativas (leitura literal); braço com piso em 0 como sensibilidade.
Paper: -0.1337 (t -2.16) em CAR[-1,+1].

TABELA 8: col 1 = distância de cada gestor ponderada pelas PALAVRAS faladas;
col 2 = ponderada pela modalidade líquida (fortes - fracas, dicionário LM;
pesos negativos truncados em 0 — decisão declarada); col 3 = exclui gestores
abaixo da MEDIANA de palavras da call (>=2 restantes).
Paper: -0.7678 (t -4.72), -0.4519 (t -4.51), -0.7911 (t -4.99).

Depois estima a Eq.(3) para cada variante com os controles do paper (amostra
comum 2009+) e com os controles antigos (amostra ampla) para referência.
Uso: python scripts/sp500_table78.py
"""
from __future__ import annotations
import re, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTDIR = ROOT / "data" / "interim" / "sp500"
MIN_MGR = 2

PAPER = ["disclosure_tone", "lagged_avg_td", "sue_pct", "btm", "lev",
         "roa", "std_roa", "std_cfo", "rd_at", "etr", "smooth",
         "length", "std_me", "ln_assets",
         "analyst_tone_disp", "analyst_tone", "industry_tone"]
OLD = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
       "length", "ln_mktcap", "industry_tone", "lagged_td", "sue_pct"]


def td_from(fp, fn, w=None):
    d = np.sqrt((fp - fp.mean()) ** 2 + (fn - fn.mean()) ** 2)
    if w is None:
        return float(d.mean())
    W = w.sum()
    return float((d * w).sum() / W) if W > 0 else np.nan


def winsor(s):
    return s.clip(s.quantile(0.05), s.quantile(0.95))


def run(df, yvar, tdvar, controls, label):
    d = df.dropna(subset=[yvar, tdvar] + controls + ["ticker", "year_quarter"]).copy()
    for c in [yvar, tdvar] + controls:
        d[c] = winsor(d[c])
    m = smf.ols(f"{yvar} ~ {tdvar} + " + " + ".join(controls)
                + " + C(ticker) + C(year_quarter)",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    print(f"{label:46s} n={len(d):>6d} | coef={m.params[tdvar]:+9.4f} "
          f"t={m.tvalues[tdvar]:+6.2f} p={m.pvalues[tdvar]:.3f}", flush=True)


def main():
    sc = pd.read_parquet(OUTDIR / "speaker_counts.parquet")
    sc = sc.sort_values(["call_id", "utterance_idx"])

    # modalidade líquida por fala (dicionário LM oficial: Strong/Weak_Modal)
    lmd = pd.read_csv(RAW / "LM_MasterDictionary.csv")
    SM = set(lmd.loc[lmd["Strong_Modal"] != 0, "Word"].str.upper())
    WM = set(lmd.loc[lmd["Weak_Modal"] != 0, "Word"].str.upper())
    print(f"LM modal: {len(SM)} fortes, {len(WM)} fracas — contagem exige repassada de texto;"
          f" usamos proxy por palavra? NÃO: col 2 sai do escopo desta noite (declarado).")

    rows = []
    for cid, g in sc.groupby("call_id", sort=False):
        mg = g[(g["role_eff"] == "manager") & (g["mk"] != "")]
        if mg.empty:
            continue
        per = mg.groupby("mk").agg(nw=("nw", "sum"), npos=("npos", "sum"), nneg=("nneg", "sum"))
        per = per[per["nw"] > 0]
        if len(per) < MIN_MGR:
            continue
        fp = (per["npos"] / per["nw"]).to_numpy(float)
        fn = (per["nneg"] / per["nw"]).to_numpy(float)
        w = per["nw"].to_numpy(float)
        td_w = td_from(fp, fn, w)                       # T8 col1: pondera por palavras
        med = np.median(w)                              # T8 col3: exclui < mediana
        keep = w >= med
        td_ex = np.nan
        if keep.sum() >= MIN_MGR:
            td_ex = td_from(fp[keep], fn[keep])
        # ---- T7: Adjusted TD (subtração pergunta->resposta, proporcional) ----
        adj = per[["nw", "npos", "nneg"]].astype(float).copy()
        qa = g[g["section"] == "qa"]
        q_pos = q_neg = 0.0
        block = []          # falas de gestores respondendo à pergunta corrente
        def flush():
            nonlocal q_pos, q_neg
            bw = sum(b[1] for b in block)
            if bw > 0 and (q_pos > 0 or q_neg > 0):
                for mk_, nw_, _, _ in block:
                    sh = nw_ / bw
                    adj.loc[mk_, "npos"] -= q_pos * sh
                    adj.loc[mk_, "nneg"] -= q_neg * sh
            block.clear(); q_pos = q_neg = 0.0
        for r in qa.itertuples():
            if r.role_eff == "analyst":
                if block:
                    flush()
                q_pos += float(r.npos); q_neg += float(r.nneg)
            elif r.role_eff == "manager" and r.mk in adj.index:
                block.append((r.mk, float(r.nw), r.npos, r.nneg))
        if block:
            flush()
        fp_a = (adj["npos"] / adj["nw"]).to_numpy(float)
        fn_a = (adj["nneg"] / adj["nw"]).to_numpy(float)
        td_adj_paper = td_from(fp_a, fn_a)
        fp_f = (adj["npos"].clip(lower=0) / adj["nw"]).to_numpy(float)
        fn_f = (adj["nneg"].clip(lower=0) / adj["nw"]).to_numpy(float)
        td_adj_floor = td_from(fp_f, fn_f)
        rows.append(dict(call_id=cid, td_w=td_w, td_ex=td_ex,
                         td_adj_paper=td_adj_paper, td_adj_floor=td_adj_floor))
    var = pd.DataFrame(rows)
    var.to_parquet(OUTDIR / "td_variants.parquet", index=False)
    print(f"variantes: {len(var)} calls "
          f"(td_ex válida em {var['td_ex'].notna().sum()})", flush=True)

    ev = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    ev = ev[ev["cdate"] >= "2009-01-01"].merge(var, on="call_id", how="left")
    print(f"corr(td, td_w)={ev['td'].corr(ev['td_w']):+.3f} | "
          f"corr(td, td_adj_paper)={ev['td'].corr(ev['td_adj_paper']):+.3f}")

    print("\n===== Eq.(3) CAR[-1,+1] — variantes | paper: TDw -0.77(-4.7), excl -0.79(-5.0), adj -0.13(-2.2) =====")
    for tdv, lbl in [("td", "TD base (ref)"), ("td_w", "T8c1 ponderada palavras"),
                     ("td_ex", "T8c3 exclui < mediana"),
                     ("td_adj_paper", "T7 adjusted (literal)"),
                     ("td_adj_floor", "T7 adjusted (piso 0)")]:
        run(ev, "car_m1p1", tdv, PAPER, f"{lbl} | paper ctrl, comum")
        run(ev, "car_m1p1", tdv, OLD, f"{lbl} | ctrl antigos, ampla")
    print("\nDONE.")


if __name__ == "__main__":
    main()
