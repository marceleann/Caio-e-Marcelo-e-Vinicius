# -*- coding: utf-8 -*-
"""CAR v2 — ancoragem TIME-AWARE do dia do evento (revisão da crítica ao CAR).

Defeito corrigido: a v1 ancorava t0 na DATA da call, ignorando o HORÁRIO. Como
a maioria das calls é após o fechamento (>=16h ET), a reação ocorre no pregão
SEGUINTE; a janela [-1,+1] da v1 ficava descentrada (2 dias pré-evento + 1 de
reação), diluindo o sinal com ruído. A v2 define o dia-evento t0 como o pregão
em que o mercado PÔDE reagir:
  - call com horário >= 16:00 ET em dia útil -> t0 = pregão seguinte;
  - call com horário < 16:00 ET -> t0 = mesmo pregão (inclui pré-abertura);
  - call em dia não útil -> t0 = próximo pregão;
  - call SEM horário confiável (has_time=False) -> t0 = data (regra da v1),
    contabilizado e reportado.
Market model idêntico (estimação [t0-120, t0-21], >=60 obs; CAR = soma dos AR
em [t0-1, t0+1]). Depois re-estima o H1 (TD crua, spec Angelo) com o CAR v2 e
imprime v1 × v2 lado a lado. Uso: python scripts/sp500_car_v2.py
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
EST_LO, EST_HI, MIN_EST = -120, -21, 60
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
    se = abs(m.params["_td"] / m.tvalues["_td"]) if m.tvalues["_td"] != 0 else np.nan
    return dict(spec=label, n=len(d), n_firms=d["ticker"].nunique(),
                coef=m.params["_td"], t=m.tvalues["_td"], p=m.pvalues["_td"], se=se)


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_h2.parquet")
    meta = pd.read_parquet(ROOT / "data" / "raw" / "calls_all.parquet")[["call_id", "has_time"]]
    ev = ev.merge(meta, on="call_id", how="left")
    cdt = pd.to_datetime(ev["call_datetime"], utc=True).dt.tz_convert("US/Eastern")
    ev["hour"] = cdt.dt.hour + cdt.dt.minute / 60.0

    px = pd.concat([pd.read_parquet(p) for p in (SP / "price_shards").glob("*.parquet")],
                   ignore_index=True).sort_values(["ticker", "date"])
    mkt = px[px["ticker"] == "^GSPC"].set_index("date")["adj_close"]
    mkt_ret = mkt.pct_change()
    tick = {}
    for t, g in px[px["ticker"] != "^GSPC"].groupby("ticker"):
        g = g.dropna(subset=["adj_close"])
        tick[t] = (g["date"].to_numpy("datetime64[ns]"),
                   g["adj_close"].pct_change().to_numpy(),
                   mkt_ret.reindex(g["date"]).to_numpy())

    car2 = np.full(len(ev), np.nan)
    shifted = same = no_time = 0
    for k, row in enumerate(ev.itertuples()):
        d = tick.get(row.ticker)
        if d is None:
            continue
        dates, r, rm = d
        D = np.datetime64(row.cdate)
        idx = int(np.searchsorted(dates, D, side="left"))
        if idx >= len(dates):
            continue
        if not bool(row.has_time):
            t0 = idx; no_time += 1                    # sem horário: regra v1, contado
        elif dates[idx] != D:
            t0 = idx; shifted += 1                    # dia não útil -> próximo pregão
        elif row.hour >= 16.0:
            t0 = idx + 1; shifted += 1                # após o fechamento -> pregão seguinte
        else:
            t0 = idx; same += 1                       # antes das 16h -> mesmo pregão
        if t0 + EST_LO < 0 or t0 + 1 >= len(r):
            continue
        sl = slice(t0 + EST_LO, t0 + EST_HI + 1)
        rr, mm = r[sl], rm[sl]
        ok = np.isfinite(rr) & np.isfinite(mm)
        if ok.sum() < MIN_EST:
            continue
        b, a = np.polyfit(mm[ok], rr[ok], 1)
        ar = r[t0 - 1:t0 + 2] - (a + b * rm[t0 - 1:t0 + 2])
        if np.isfinite(ar).all():
            car2[k] = float(ar.sum())
    ev["car_v2"] = car2
    ev.to_parquet(OUTDIR / "events_sp500_car2.parquet", index=False)

    print("=== ANCORAGEM ===")
    tot = shifted + same + no_time
    print(f"eventos ancorados: {tot} | deslocados p/ pregão seguinte (pós-16h ou dia não útil): "
          f"{shifted} ({100*shifted/tot:.0f}%) | mesmo dia (<16h): {same} ({100*same/tot:.0f}%) | "
          f"sem horário (regra v1): {no_time} ({100*no_time/tot:.0f}%)")
    print(f"CAR v1: n={ev['car'].notna().sum()}, sd={100*ev['car'].std():.2f}% | "
          f"CAR v2: n={ev['car_v2'].notna().sum()}, sd={100*ev['car_v2'].std():.2f}% | "
          f"corr(v1,v2)={ev['car'].corr(ev['car_v2']):+.2f}")

    print("\n=== H1 com TD CRUA (spec Angelo) — v1 × v2 ===")
    print(f"{'CAR':10s} {'n':>6s} {'firmas':>6s} {'coef(SD)':>10s} {'t':>7s} {'p':>7s} {'SE':>8s}")
    for yv, lbl in [("car", "v1 (data)"), ("car_v2", "v2 (hora)")]:
        r = run(ev, yv, "td", lbl)
        print(f"{lbl:10s} {r['n']:>6d} {r['n_firms']:>6d} {r['coef']:>10.4f} {r['t']:>7.2f} "
              f"{r['p']:>7.3f} {r['se']:>8.5f}")
    for yv, lbl in [("car_v2", "v2, z")]:  # braço z logado (DSR)
        r = run(ev, yv, "z", lbl)
        print(f"{lbl:10s} {r['n']:>6d} {r['n_firms']:>6d} {r['coef']:>10.4f} {r['t']:>7.2f} "
              f"{r['p']:>7.3f} {r['se']:>8.5f}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
