# -*- coding: utf-8 -*-
"""Replicação S&P 500 — H3 (pedaço viável): TD prevê o SUE do trimestre seguinte?

Especificação do Angelo (TD CRUA, controles do H1 incluindo o SUE corrente, FE
firma+trimestre, cluster por firma, winsor 5/95, TD padronizada). Duas variáveis
dependentes, declaradas a priori (a operacionalização exata do paper não está
nas nossas notas): (a) sue_next (nível: surpresa pior?) e (b) |sue_next|
(magnitude: surpresa mais imprevisível = risco operacional).
Uso: python scripts/sp500_h3_sue.py
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "interim" / "sp500"
CONTROLS = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
            "length", "ln_mktcap", "industry_tone", "lagged_td", "sue_pct"]


def winsor(s, lo=0.05, hi=0.95):
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def zscore(s):
    return (s - s.mean()) / s.std()


def run(df, yvar, label):
    d = df.dropna(subset=[yvar, "td"] + CONTROLS + ["ticker", "year_quarter"]).copy()
    d["_td"] = zscore(winsor(d["td"]))
    d["_y"] = winsor(d[yvar])
    for c in CONTROLS:
        d[c] = winsor(d[c])
    m = smf.ols("_y ~ _td + " + " + ".join(CONTROLS) + " + C(ticker) + C(year_quarter)",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    return dict(spec=label, n=len(d), n_firms=d["ticker"].nunique(),
                coef=m.params["_td"], t=m.tvalues["_td"], p=m.pvalues["_td"], r2=m.rsquared)


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_h2.parquet")
    ev = ev.sort_values(["ticker", "cdate"]).reset_index(drop=True)
    g = ev.groupby("ticker")
    ev["sue_next"] = g["sue_pct"].shift(-1)
    ev["gap_next_d"] = (g["cdate"].shift(-1) - ev["cdate"]).dt.days
    # só transições call->call consecutivas plausíveis (um trimestre ~ 60-180 dias)
    ok = ev["gap_next_d"].between(30, 200)
    ev.loc[~ok, "sue_next"] = np.nan
    ev["abs_sue_next"] = ev["sue_next"].abs()

    print(f"eventos com sue_next válido: {ev['sue_next'].notna().sum()} "
          f"(gap mediano {ev.loc[ok,'gap_next_d'].median():.0f} dias)")
    print("\nH3 (pedaço viável) — TD CRUA, spec Angelo | FE firma+trimestre, cluster firma")
    print(f"{'dependente':28s} {'n':>6s} {'firmas':>6s} {'coef(SD)':>10s} {'t':>7s} {'p':>7s} {'R2':>6s}")
    for yv, lbl in [("sue_next", "SUE t+1 (nível)"), ("abs_sue_next", "|SUE t+1| (magnitude)")]:
        r = run(ev, yv, lbl)
        print(f"{lbl:28s} {r['n']:>6d} {r['n_firms']:>6d} {r['coef']:>10.4f} {r['t']:>7.2f} {r['p']:>7.3f} {r['r2']:>6.3f}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
