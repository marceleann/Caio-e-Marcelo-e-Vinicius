# -*- coding: utf-8 -*-
"""Replicação S&P 500 — H2: Tone Distance prevê risco de mercado PÓS-call?

Desfecho (Angelo, H2): volatilidade realizada após o anúncio. Aqui: desvio-padrão
dos retornos diários nas janelas [t0+2, t0+21] (20 pregões) e [t0+2, t0+121]
(120 pregões), t0 = 1º pregão >= data da call. Fora do escopo (declarado):
amplitude high-low (não temos OHLC no cache) e volatilidade implícita (opções,
base paga).

Especificação idêntica ao H1 (8 controles + FE firma + FE trimestre + cluster
por firma + winsor 5/95 + TD padronizada), em duas variantes:
  A) controles do H1;
  B) controles do H1 + volatilidade PRÉ-call (janela [-120,-21]) — teste mais
     duro: TD prevê risco ALÉM da persistência da volatilidade?
Sinal esperado: POSITIVO. Uso: python scripts/sp500_h2_risk.py
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "data" / "raw" / "sp500"
OUTDIR = ROOT / "data" / "interim" / "sp500"
CONTROLS = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
            "length", "ln_mktcap", "industry_tone", "lagged_td", "sue_pct"]


def winsor(s, lo=0.05, hi=0.95):
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def zscore(s):
    return (s - s.mean()) / s.std()


def run(df, yvar, tdvar, controls, label):
    d = df.dropna(subset=[yvar, tdvar] + controls + ["ticker", "year_quarter"]).copy()
    if len(d) < 100:
        return dict(spec=label, n=len(d), coef=np.nan, t=np.nan, p=np.nan, r2=np.nan, n_firms=0)
    d["_td"] = zscore(winsor(d[tdvar]))
    d["_y"] = winsor(d[yvar])
    for c in controls:
        d[c] = winsor(d[c])
    m = smf.ols("_y ~ _td + " + " + ".join(controls) + " + C(ticker) + C(year_quarter)",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    return dict(spec=label, n=len(d), coef=m.params["_td"], t=m.tvalues["_td"],
                p=m.pvalues["_td"], r2=m.rsquared, n_firms=d["ticker"].nunique())


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500.parquet")
    px = pd.concat([pd.read_parquet(p) for p in (SP / "price_shards").glob("*.parquet")],
                   ignore_index=True).sort_values(["ticker", "date"])
    tick = {}
    for t, g in px[px["ticker"] != "^GSPC"].groupby("ticker"):
        g = g.dropna(subset=["adj_close"])
        tick[t] = (g["date"].to_numpy("datetime64[ns]"),
                   g["adj_close"].pct_change().to_numpy())

    v20 = np.full(len(ev), np.nan)
    v120 = np.full(len(ev), np.nan)
    vpre = np.full(len(ev), np.nan)
    for k, row in enumerate(ev.itertuples()):
        d = tick.get(row.ticker)
        if d is None:
            continue
        dates, r = d
        p = int(np.searchsorted(dates, np.datetime64(row.cdate)))
        w = r[p + 2: p + 22]
        w = w[np.isfinite(w)]
        if len(w) >= 15:
            v20[k] = w.std()
        w = r[p + 2: p + 122]
        w = w[np.isfinite(w)]
        if len(w) >= 80:
            v120[k] = w.std()
        if p - 120 >= 0:
            w = r[p - 120: p - 20]
            w = w[np.isfinite(w)]
            if len(w) >= 60:
                vpre[k] = w.std()
    ev["vol_20"], ev["vol_120"], ev["vol_pre"] = v20, v120, vpre
    ev.to_parquet(OUTDIR / "events_sp500_h2.parquet", index=False)

    print("=== SANITY ===")
    print(f"eventos: {len(ev)} | com vol_20: {np.isfinite(v20).sum()} | vol_120: "
          f"{np.isfinite(v120).sum()} | vol_pre: {np.isfinite(vpre).sum()}")
    print(f"vol_20 diária: mediana {np.nanmedian(v20)*100:.2f}% | p90 {np.nanpercentile(v20,90)*100:.2f}%")
    print(f"corr(vol_pre, vol_20) = {pd.Series(vpre).corr(pd.Series(v20)):+.2f}  (persistência esperada alta)")

    for yv, lbl in [("vol_20", "H2 — vol realizada 20 pregões pós-call"),
                    ("vol_120", "H2 — vol realizada 120 pregões pós-call")]:
        print("\n" + "=" * 86)
        print(f"{lbl} (esperado POSITIVO) | FE firma+trimestre, cluster firma")
        print("=" * 86)
        print(f"{'spec':34s} {'n':>6s} {'firmas':>6s} {'coef(SD)':>10s} {'t':>7s} {'p':>7s} {'R2':>6s}")
        for tdv, sl in [("td", "TD crua"), ("td_adj", "TD_adj"), ("z", "z")]:
            r = run(ev, yv, tdv, CONTROLS, f"A: {sl} + controles H1")
            print(f"{r['spec']:34s} {r['n']:>6d} {r['n_firms']:>6d} {r['coef']:>10.6f} {r['t']:>7.2f} {r['p']:>7.3f} {r['r2']:>6.3f}")
        for tdv, sl in [("td", "TD crua"), ("td_adj", "TD_adj"), ("z", "z")]:
            r = run(ev, yv, tdv, CONTROLS + ["vol_pre"], f"B: {sl} + ctrl + vol_pre")
            print(f"{r['spec']:34s} {r['n']:>6d} {r['n_firms']:>6d} {r['coef']:>10.6f} {r['t']:>7.2f} {r['p']:>7.3f} {r['r2']:>6.3f}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
