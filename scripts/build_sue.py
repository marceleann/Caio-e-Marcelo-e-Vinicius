# -*- coding: utf-8 -*-
"""Controle de SUE (surpresa de lucros) — proxy de consenso via yfinance.

Especificação exige controlar a surpresa de lucros no H1 (o CAR do anúncio é
dominado por ela; sem o controle, variância residual infla o SE e afoga o tom).
Proxy com embasamento: surpresa de CONSENSO = (EPS reportado − EPS estimado),
padrão da literatura de event study (equivalente aberto do I/B/E/S).

Forma funcional: surpresa PERCENTUAL (A−E)/max(|E|, piso) — invariante a ajuste
de split (A e E na mesma unidade), robusta a E≈0 via piso; winsorização 5/95 na
regressão. Casamento call↔anúncio: data de earnings mais próxima a ±3 dias da
data da call.

Saída: data/interim/events_sue.parquet [call_id, sue_pct, sue_days_gap]
Cache de rede: data/raw/earnings_cache.parquet
"""
from __future__ import annotations
import os, time
_CA = r"C:\Users\Marcelo\AppData\Local\Temp\claude\cacert.pem"
os.environ.setdefault("CURL_CA_BUNDLE", _CA)
os.environ.setdefault("SSL_CERT_FILE", _CA)

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, INT, PROC = ROOT / "data" / "raw", ROOT / "data" / "interim", ROOT / "data" / "processed"
E_FLOOR = 0.01          # piso do denominador (|E| em US$/ação)
MAX_GAP_D = 3           # tolerância call ↔ data de earnings


def fetch_earnings(tickers):
    cache = RAW / "earnings_cache.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import yfinance as yf
    rows = []
    for i, t in enumerate(tickers):
        sym = {"FI": "FISV"}.get(t, t)
        try:
            e = yf.Ticker(sym).get_earnings_dates(limit=100)  # Yahoo capa em 100 (~25 anos)
            if e is not None and len(e):
                d = e.reset_index()
                d.columns = [c.strip() for c in d.columns]
                d = d.rename(columns={d.columns[0]: "edate", "EPS Estimate": "eps_est",
                                      "Reported EPS": "eps_act"})
                d["edate"] = pd.to_datetime(d["edate"], utc=True).dt.tz_localize(None).dt.normalize()
                d["ticker"] = t
                rows.append(d[["ticker", "edate", "eps_est", "eps_act"]])
        except Exception as ex:
            print(f"  {t}: ERRO {str(ex)[:60]}")
        if (i + 1) % 25 == 0:
            print(f"  earnings: {i+1}/{len(tickers)}"); time.sleep(1)
    out = pd.concat(rows, ignore_index=True).dropna(subset=["eps_est", "eps_act"])
    out = out.sort_values(["ticker", "edate"]).drop_duplicates(["ticker", "edate"])
    out.to_parquet(cache, index=False)
    return out


def main():
    ev = pd.read_parquet(PROC / "events.parquet")[["call_id", "ticker", "call_datetime"]]
    ev["cdate"] = ev["call_datetime"].dt.tz_convert("US/Eastern").dt.tz_localize(None).dt.normalize()
    tickers = sorted(ev["ticker"].unique())
    earn = fetch_earnings(tickers)
    print(f"earnings com estimado+reportado: {len(earn)} linhas, "
          f"{earn['ticker'].nunique()}/{len(tickers)} tickers, "
          f"{earn['edate'].min().date()} a {earn['edate'].max().date()}")

    by_t = {t: g.sort_values("edate").reset_index(drop=True) for t, g in earn.groupby("ticker")}
    rows = []
    for r in ev.itertuples():
        g = by_t.get(r.ticker)
        if g is None:
            continue
        gaps = (g["edate"] - r.cdate).dt.days.abs()
        j = int(gaps.idxmin())
        if gaps.iloc[j] > MAX_GAP_D:
            continue
        a, e = float(g["eps_act"].iloc[j]), float(g["eps_est"].iloc[j])
        sue = (a - e) / max(abs(e), E_FLOOR)
        rows.append({"call_id": r.call_id, "sue_pct": sue, "sue_days_gap": int(gaps.iloc[j])})
    out = pd.DataFrame(rows)
    out.to_parquet(INT / "events_sue.parquet", index=False)
    print(f"\nEscrito events_sue.parquet: {len(out)}/{len(ev)} calls com SUE "
          f"({100*len(out)/len(ev):.1f}%)")
    print(f"sue_pct: mediana {out['sue_pct'].median():+.3f} | p5 {out['sue_pct'].quantile(.05):+.3f} "
          f"| p95 {out['sue_pct'].quantile(.95):+.3f} | beat rate {100*(out['sue_pct']>0).mean():.0f}%")
    print(f"gap de datas: {out['sue_days_gap'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
