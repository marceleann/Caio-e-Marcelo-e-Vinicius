# -*- coding: utf-8 -*-
"""CAR v3 — EXATAMENTE como o Angelo (2025), conferido no PDF do paper.

Metodologia do paper (Seção 5.1, verificada no texto):
  - Retorno anormal relativo ao CAPM: AR = (r - rf) - beta*(rm - rf), com beta
    estimado por OLS de excesso sobre excesso;
  - Janela de estimação de 100 dias com MÍNIMO de 70 e GAP de >=50 dias antes
    da call (posições [t0-150, t0-51]);
  - Janelas de evento: CAR[-1,+1], CAR[-1,+2], CAR[-1,+5] (começam 1 dia antes
    do dia do ANÚNCIO);
  - rf diário de Ken French (CRSP no paper; ^GSPC como proxy do mercado);
  - Eq.(3): CAR ~ TD + controles + FE firma + FE ano-trimestre.
Divergência DECLARADA que permanece: os controles contábeis da Eq.(3) (BTM,
Lev, ROA, Std ROA, Std CFO, RD, ETR) exigem Compustat; usamos os controles
disponíveis (tom, lnME, SUE, etc.) — os FE de firma absorvem a parte
persistente dos fundamentos. Uso: python scripts/sp500_car_v3.py
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
EST_LO, EST_HI, MIN_EST = -150, -51, 70     # 100 dias, gap 50, mínimo 70 (paper)
WINDOWS = {"car_m1p1": 1, "car_m1p2": 2, "car_m1p5": 5}
CONTROLS = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
            "length", "ln_mktcap", "industry_tone", "lagged_td", "sue_pct"]


def winsor(s, lo=0.05, hi=0.95):
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def zscore(s):
    return (s - s.mean()) / s.std()


def run(df, yvar, tdvar, label):
    d = df.dropna(subset=[yvar, tdvar] + CONTROLS + ["ticker", "year_quarter"]).copy()
    d["_td"] = zscore(winsor(d[tdvar]))
    d["_y"] = winsor(d[yvar])
    for c in CONTROLS:
        d[c] = winsor(d[c])
    m = smf.ols("_y ~ _td + " + " + ".join(CONTROLS) + " + C(ticker) + C(year_quarter)",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    return dict(spec=label, n=len(d), n_firms=d["ticker"].nunique(),
                coef=m.params["_td"], t=m.tvalues["_td"], p=m.pvalues["_td"])


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_h2.parquet")
    rf = pd.read_parquet(ROOT / "data" / "raw" / "rf_daily.parquet").set_index("date")["rf"]

    px = pd.concat([pd.read_parquet(p) for p in (SP / "price_shards").glob("*.parquet")],
                   ignore_index=True).sort_values(["ticker", "date"])
    mkt = px[px["ticker"] == "^GSPC"].set_index("date")["adj_close"]
    mkt_ret = mkt.pct_change()
    tick = {}
    for t, g in px[px["ticker"] != "^GSPC"].groupby("ticker"):
        g = g.dropna(subset=["adj_close"])
        dates = g["date"].to_numpy("datetime64[ns]")
        r = g["adj_close"].pct_change().to_numpy()
        rm = mkt_ret.reindex(g["date"]).to_numpy()
        rfv = rf.reindex(g["date"]).ffill().to_numpy()
        tick[t] = (dates, r - rfv, rm - rfv, r, rm)   # excessos p/ CAPM

    cars = {k: np.full(len(ev), np.nan) for k in WINDOWS}
    for k_ev, row in enumerate(ev.itertuples()):
        d = tick.get(row.ticker)
        if d is None:
            continue
        dates, re_, rem, _, _ = d
        t0 = int(np.searchsorted(dates, np.datetime64(row.cdate)))  # dia do anúncio (como o paper)
        if t0 + EST_LO < 0 or t0 + 5 >= len(re_):
            continue
        sl = slice(t0 + EST_LO, t0 + EST_HI + 1)
        x, y = rem[sl], re_[sl]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < MIN_EST:
            continue
        b = np.polyfit(x[ok], y[ok], 1)[0]
        for col, w in WINDOWS.items():
            seg_r, seg_m = re_[t0 - 1:t0 + w + 1], rem[t0 - 1:t0 + w + 1]
            ar = seg_r - b * seg_m                    # CAPM: sem alfa na previsão
            if np.isfinite(ar).all():
                cars[col][k_ev] = float(ar.sum())
    for col in WINDOWS:
        ev[col] = cars[col]
    ev.to_parquet(OUTDIR / "events_sp500_car3.parquet", index=False)

    print("=== CAR v3 (CAPM, estimação 100d/gap 50/min 70 — spec do paper) ===")
    for col in WINDOWS:
        print(f"{col}: n={ev[col].notna().sum()}, sd={100*ev[col].std():.2f}%, "
              f"corr c/ v1={ev['car'].corr(ev[col]):+.2f}")
    print("\n=== Eq.(3) do paper — TD CRUA | FE firma+trimestre, cluster firma ===")
    print("(controles contábeis do Compustat indisponíveis — declarado; usamos os nossos)")
    print(f"{'janela':12s} {'n':>6s} {'firmas':>6s} {'coef(SD)':>10s} {'t':>7s} {'p':>7s}")
    for col, lbl in [("car_m1p1", "CAR[-1,+1]"), ("car_m1p2", "CAR[-1,+2]"), ("car_m1p5", "CAR[-1,+5]")]:
        r = run(ev, col, "td", lbl)
        print(f"{lbl:12s} {r['n']:>6d} {r['n_firms']:>6d} {r['coef']:>10.4f} {r['t']:>7.2f} {r['p']:>7.3f}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
