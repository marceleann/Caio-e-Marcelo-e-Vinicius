# -*- coding: utf-8 -*-
"""CAR de ANÚNCIO (market model) ancorado na DATA DA CALL — como o Angelo mede o H1.

Correção do erro apontado: o event study antigo ancorava o CAR no decision_date
(T+1) com janela [-1,+10]. O H1 do Angelo é uma MEDIÇÃO (não um trade): CAR[-1,+1]
em torno do ANÚNCIO. A restrição T+1 vale só para o backtest, não para o teste de
hipótese. Aqui ancoramos t0 = 1º pregão >= data da call e medimos a reação.

Market model: α,β estimados em [t0-120, t0-21] (só passado, sem look-ahead na
janela do evento); AR_k = r_k - (α + β·rm_k); CAR = soma na janela.
Saída: data/interim/announcement_car.parquet [call_id, car, car_0p1, car_m1p3, n_est].
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

INT = Path(__file__).resolve().parents[1] / "data" / "interim"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
EST_LO, EST_HI, MIN_OBS = -120, -21, 60
WINDOWS = {"car": (-1, 1), "car_0p1": (0, 1), "car_m1p3": (-1, 3)}


def main():
    px = pd.read_parquet(RAW / "prices.parquet")[["date", "ticker", "ret_cc"]]
    px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None).dt.normalize()
    mkt = px[px["ticker"] == "^GSPC"][["date", "ret_cc"]].rename(columns={"ret_cc": "rm"})
    # série alinhada (data, r, rm) por ticker
    series = {}
    for tk, g in px[px["ticker"] != "^GSPC"].groupby("ticker"):
        m = g[["date", "ret_cc"]].merge(mkt, on="date", how="inner").sort_values("date")
        series[tk] = (m["date"].to_numpy(), m["ret_cc"].to_numpy(dtype=np.float64),
                      m["rm"].to_numpy(dtype=np.float64))

    ev = pd.read_parquet(PROC / "events.parquet")[["call_id", "ticker", "call_datetime"]]
    ev["cd"] = ev["call_datetime"].dt.tz_convert("US/Eastern").dt.tz_localize(None).dt.normalize()

    rows = []
    for r in ev.itertuples():
        s = series.get(r.ticker)
        if s is None:
            continue
        dates, ret, rm = s
        t0 = int(np.searchsorted(dates, np.datetime64(r.cd), side="left"))
        need_hi = t0 + max(w[1] for w in WINDOWS.values())
        if t0 + EST_LO < 0 or need_hi >= len(dates):
            continue
        est = slice(t0 + EST_LO, t0 + EST_HI + 1)
        y, x = ret[est], rm[est]
        ok = np.isfinite(y) & np.isfinite(x)
        if ok.sum() < MIN_OBS:
            continue
        y, x = y[ok], x[ok]
        vx = x.var()
        if vx <= 0:
            continue
        beta = np.cov(y, x, bias=True)[0, 1] / vx
        alpha = y.mean() - beta * x.mean()
        out = {"call_id": r.call_id, "n_est": int(ok.sum())}
        good = True
        for name, (lo, hi) in WINDOWS.items():
            ar = ret[t0 + lo: t0 + hi + 1] - (alpha + beta * rm[t0 + lo: t0 + hi + 1])
            if not np.isfinite(ar).all():
                good = False; break
            out[name] = float(ar.sum())
        if good:
            rows.append(out)
    res = pd.DataFrame(rows)
    dest = INT / "announcement_car.parquet"
    res.to_parquet(dest, index=False)
    print(f"Escrito {dest}: {len(res)} eventos com CAR de anúncio")
    print(res[["car", "car_0p1", "car_m1p3"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
