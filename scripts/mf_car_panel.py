# -*- coding: utf-8 -*-
"""Extensão MF — FASE MF-4: CAR (CAPM, espelho do v3) + painel H1 pré-registrado.

CAR idêntico ao sp500_car_v3: AR = (r - rf) - beta*(rm - rf); estimação 100
pregões (mín. 70) com gap de 50; janelas [-1,+1], [-1,+2], [-1,+5] ancoradas
na DATA do anúncio; mercado ^GSPC; rf diário Ken French.
Preços: shards MF (novos) + shards S&P (tickers overlap).

Painel (PREREG_MF.md, commitado antes de qualquer resultado):
  CAR ~ sinal + disclosure_tone + analyst_tone + analyst_tone_disp + length
        + lagged_td + FE firma + FE ano-tri | cluster firma | winsor 5/95
  sinais: td_b e td_w (td_a como referência); amostra principal: só tickers
  FORA do S&P 500; braço com todos como robustez. ln_mktcap: sem dados de
  ações em circulação neste universo -> excluído pela condição do pré-registro
  (cobertura <70%), declarado.
Sanity checks impressos ANTES dos betas: n, sd do CAR, % positivos.
Uso: python scripts/mf_car_panel.py
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
MF = ROOT / "data" / "interim" / "mf"
EST_LO, EST_HI, MIN_EST = -150, -51, 70
WINDOWS = {"car_m1p1": 1, "car_m1p2": 2, "car_m1p5": 5}
CTRL = ["disclosure_tone", "analyst_tone", "analyst_tone_disp", "length", "lagged_td"]


def winsor(s):
    return s.clip(s.quantile(0.05), s.quantile(0.95))


def main():
    ev = pd.read_parquet(MF / "tone_distance_mf.parquet")
    ev["cdate"] = pd.to_datetime(ev["call_datetime"]).dt.normalize()
    ev["year_quarter"] = ev["year"].astype(str) + "Q" + ev["quarter"].astype(str)
    ev["lagged_td"] = ev["lagged_td_w"]

    px_mf = pd.concat([pd.read_parquet(p) for p in
                       (ROOT / "data" / "raw" / "mf" / "price_shards").glob("*.parquet")],
                      ignore_index=True)
    need_overlap = set(ev["ticker"]) - set(px_mf["ticker"])
    sp_frames = []
    for p in (ROOT / "data" / "raw" / "sp500" / "price_shards").glob("*.parquet"):
        d = pd.read_parquet(p)
        sp_frames.append(d[d["ticker"].isin(need_overlap | {"^GSPC"})])
    px = pd.concat([px_mf] + sp_frames, ignore_index=True)
    px["date"] = pd.to_datetime(px["date"])
    px["adj_close"] = pd.to_numeric(px["adj_close"], errors="coerce")
    px = (px.dropna(subset=["adj_close"])
            .drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"]))

    rf = pd.read_parquet(ROOT / "data" / "raw" / "rf_daily.parquet").set_index("date")["rf"]
    mkt = px[px["ticker"] == "^GSPC"].set_index("date")["adj_close"].pct_change()
    tick = {}
    for t, g in px[px["ticker"] != "^GSPC"].groupby("ticker"):
        dates = g["date"].to_numpy("datetime64[ns]")
        r = g["adj_close"].pct_change().to_numpy(dtype=float)
        rm = mkt.reindex(g["date"]).to_numpy(dtype=float)
        rfv = rf.reindex(g["date"]).ffill().to_numpy(dtype=float)
        tick[t] = (dates, r - rfv, rm - rfv)

    cars = {k: np.full(len(ev), np.nan) for k in WINDOWS}
    for k_ev, row in enumerate(ev.itertuples()):
        d = tick.get(row.ticker)
        if d is None:
            continue
        dates, re_, rem = d
        t0 = int(np.searchsorted(dates, np.datetime64(row.cdate)))
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
            ar = seg_r - b * seg_m
            if np.isfinite(ar).all() and len(ar) == w + 2:
                cars[col][k_ev] = float(ar.sum())
    for col in WINDOWS:
        ev[col] = cars[col]
    ev.to_parquet(MF / "events_mf.parquet", index=False)

    # ---------- sanity checks (antes de qualquer beta) ----------
    print("=== SANITY ===")
    print(f"eventos: {len(ev)} | com CAR[-1,+1]: {ev['car_m1p1'].notna().sum()} "
          f"({100*ev['car_m1p1'].notna().mean():.0f}%)")
    print(f"sd CAR[-1,+1]: {100*ev['car_m1p1'].std():.1f}%  (S&P era ~6%; small caps deve ser maior)")
    print(f"% CAR>0: {100*(ev['car_m1p1'] > 0).mean() / ev['car_m1p1'].notna().mean():.0f}%")

    sp_tickers = set(pd.read_parquet(ROOT / "data" / "raw" / "calls_all.parquet")["ticker"].unique())
    novo = ev[~ev["ticker"].isin(sp_tickers)]
    print(f"amostra principal (fora do S&P): {len(novo)} eventos, "
          f"{novo['ticker'].nunique()} firmas, com CAR: {novo['car_m1p1'].notna().sum()}")

    # ---------- painel pré-registrado ----------
    def run(d, yv, sig, label):
        dd = d.dropna(subset=[yv, sig] + CTRL + ["ticker", "year_quarter"]).copy()
        if len(dd) < 400 or dd["ticker"].nunique() < 25:
            print(f"{label:40s} n insuficiente ({len(dd)})")
            return
        for c in [yv, sig] + CTRL:
            dd[c] = winsor(dd[c])
        iqr = dd[sig].quantile(0.75) - dd[sig].quantile(0.25)
        m = smf.ols(f"{yv} ~ {sig} + " + " + ".join(CTRL) + " + C(ticker) + C(year_quarter)",
                    data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["ticker"]})
        print(f"{label:40s} n={len(dd):>6d} fir={dd['ticker'].nunique():>5d} | "
              f"coef={m.params[sig]:+8.4f} t={m.tvalues[sig]:+6.2f} p={m.pvalues[sig]:.3f} "
              f"| IQR->y={100*m.params[sig]*iqr:+.3f}%")

    print("\n=== PAINEL H1 — AMOSTRA PRINCIPAL (só tickers fora do S&P) ===")
    for sig in ["td_b", "td_w", "td_a"]:
        for yv in ["car_m1p1", "car_m1p2", "car_m1p5"]:
            run(novo, yv, sig, f"{yv} ~ {sig} (novos)")
    print("\n=== ROBUSTEZ (universo MF completo) ===")
    for sig in ["td_b", "td_w"]:
        run(ev, "car_m1p1", sig, f"car_m1p1 ~ {sig} (todos)")
    print("\nDONE.")


if __name__ == "__main__":
    main()
