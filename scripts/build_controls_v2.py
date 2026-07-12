# -*- coding: utf-8 -*-
"""Controles v2 para a replicação do Angelo: MARKET CAP real + INDUSTRY TONE.

Correções apontadas na revisão (2026-07-10):
- `size_proxy` atual é volume em dólar (LIQUIDEZ), não tamanho. O Angelo usa
  Standardized ME (market equity). Aqui: ln(mktcap) = ln(close BRUTO × shares).
  * close BRUTO (auto_adjust=False): mktcap com preço ajustado seria distorcido
    por splits/dividendos futuros.
  * shares outstanding: yfinance get_shares_full — cobertura REAL só 2015+;
    antes disso, estendemos o 1º valor conhecido p/ trás com flag
    `mktcap_extrapolated=True` (limitação declarada; regressão roda também na
    janela 2015+ onde é exato).
- `industry_tone` (Angelo, Tabela 2): média do disclosure_tone das OUTRAS
  empresas do universo (nosso universo ~ o setor) em janela de 90 dias
  ESTRITAMENTE anterior à call (PIT); NaN com <3 pares.

Saída: data/interim/events_controls_v2.parquet
  [call_id, ln_mktcap, mktcap_extrapolated, industry_tone]
Caches de rede: data/raw/shares_cache.parquet, data/raw/rawclose_cache.parquet.
"""
from __future__ import annotations
import os, time
# certificado: caminho SEM acento (curl falha com 'Itaú' no path)
_CA = r"C:\Users\Marcelo\AppData\Local\Temp\claude\cacert.pem"
os.environ.setdefault("CURL_CA_BUNDLE", _CA)
os.environ.setdefault("SSL_CERT_FILE", _CA)

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, INT, PROC = ROOT / "data" / "raw", ROOT / "data" / "interim", ROOT / "data" / "processed"
PEER_WINDOW_D, PEER_MIN = 90, 3


def fetch_shares(tickers):
    cache = RAW / "shares_cache.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import yfinance as yf
    rows = []
    for i, t in enumerate(tickers):
        for sym in ({"FI": "FISV"}.get(t, t),):
            try:
                s = yf.Ticker(sym).get_shares_full(start="2004-01-01")
                if s is not None and len(s):
                    df = s.reset_index()
                    df.columns = ["date", "shares"]
                    # tz varia por ticker (aware/naive misto) — normaliza JÁ AQUI
                    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
                    df["ticker"] = t
                    rows.append(df)
            except Exception as e:
                print(f"  shares {t}: ERRO {str(e)[:60]}")
        if (i + 1) % 25 == 0:
            print(f"  shares: {i+1}/{len(tickers)}"); time.sleep(1)
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    out.to_parquet(cache, index=False)
    return out


def fetch_rawclose(tickers):
    cache = RAW / "rawclose_cache.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import yfinance as yf
    rows = []
    for i in range(0, len(tickers), 20):
        chunk = [{"FI": "FISV"}.get(t, t) for t in tickers[i:i + 20]]
        try:
            df = yf.download(chunk, start="2004-06-01", auto_adjust=False,
                             progress=False, group_by="ticker")
            for sym in chunk:
                t = {"FISV": "FI"}.get(sym, sym)
                try:
                    c = df[sym]["Close"].dropna()
                except Exception:
                    continue
                if len(c):
                    r = c.reset_index(); r.columns = ["date", "raw_close"]; r["ticker"] = t
                    rows.append(r)
        except Exception as e:
            print(f"  rawclose chunk {i}: ERRO {str(e)[:60]}")
        time.sleep(1)
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out = out.sort_values(["ticker", "date"])
    out.to_parquet(cache, index=False)
    return out


def main():
    ev = pd.read_parquet(PROC / "events.parquet")[["call_id", "ticker", "call_datetime", "disclosure_tone"]]
    ev["cd"] = ev["call_datetime"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
    ev["cdate"] = ev["cd"].dt.normalize()
    tickers = sorted(ev["ticker"].unique())
    print(f"eventos: {len(ev)} | tickers: {len(tickers)}")

    shares = fetch_shares(tickers)
    print(f"shares: {shares['ticker'].nunique()}/{len(tickers)} tickers, "
          f"{shares['date'].min().date()} a {shares['date'].max().date()}")
    rawc = fetch_rawclose(tickers)
    print(f"raw close: {rawc['ticker'].nunique()}/{len(tickers)} tickers")

    # ---- ln_mktcap point-in-time (asof estritamente anterior à data da call) ----
    out_rows = []
    sh_g = {t: g.sort_values("date") for t, g in shares.groupby("ticker")}
    px_g = {t: g.sort_values("date") for t, g in rawc.groupby("ticker")}
    for r in ev.itertuples():
        sh, px = sh_g.get(r.ticker), px_g.get(r.ticker)
        ln_mc, extrap = np.nan, False
        if sh is not None and px is not None and len(sh) and len(px):
            i_px = int(np.searchsorted(px["date"].to_numpy(), np.datetime64(r.cdate), side="left")) - 1
            if i_px >= 0:
                close = float(px["raw_close"].iloc[i_px])
                j = int(np.searchsorted(sh["date"].to_numpy(), np.datetime64(r.cdate), side="right")) - 1
                if j >= 0:
                    n_sh = float(sh["shares"].iloc[j])
                else:  # antes da 1ª observação: estende p/ trás, com flag
                    n_sh, extrap = float(sh["shares"].iloc[0]), True
                if close > 0 and n_sh > 0:
                    ln_mc = float(np.log(close * n_sh))
        out_rows.append({"call_id": r.call_id, "ln_mktcap": ln_mc, "mktcap_extrapolated": extrap})
    ctrl = pd.DataFrame(out_rows)

    # ---- industry tone: média PIT do disclosure_tone dos PARES (90d, estrito) ----
    evs = ev.sort_values("cd").reset_index(drop=True)
    times = evs["cd"].to_numpy()
    vals = evs["disclosure_tone"].to_numpy(dtype=np.float64)
    tks = evs["ticker"].to_numpy()
    win = np.timedelta64(PEER_WINDOW_D, "D")
    lo = np.searchsorted(times, times - win, side="left")
    hi = np.searchsorted(times, times, side="left")  # ESTRITO: mesmo instante fora
    valid = np.isfinite(vals)
    cv = np.concatenate([[0.0], np.cumsum(np.where(valid, vals, 0.0))])
    cc = np.concatenate([[0], np.cumsum(valid.astype(int))])
    ind = np.full(len(evs), np.nan)
    for i in range(len(evs)):
        sl = slice(lo[i], hi[i])
        mask = tks[sl] != tks[i]
        v = vals[sl][mask]
        v = v[np.isfinite(v)]
        if len(v) >= PEER_MIN:
            ind[i] = v.mean()
    evs["industry_tone"] = ind
    ctrl = ctrl.merge(evs[["call_id", "industry_tone"]], on="call_id", how="left")

    dest = INT / "events_controls_v2.parquet"
    ctrl.to_parquet(dest, index=False)
    print(f"\nEscrito {dest}")
    print(f"ln_mktcap não-nulo: {ctrl['ln_mktcap'].notna().sum()}/{len(ctrl)} "
          f"(extrapolado p/ trás: {int(ctrl['mktcap_extrapolated'].sum())})")
    print(f"industry_tone não-nulo: {ctrl['industry_tone'].notna().sum()}/{len(ctrl)}")


if __name__ == "__main__":
    main()
