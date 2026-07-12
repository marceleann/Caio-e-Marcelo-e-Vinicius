# -*- coding: utf-8 -*-
"""Replicação S&P 500 — FASE B: dados de mercado dos ~685 tickers (rede).

Busca e cacheia em shards RESUMÍVEIS (re-execução pula o que já existe):
  1. Preços diários 2004+ com auto_adjust=False -> 'Adj Close' (CAR/retornos,
     ajustado) E 'Close' bruto (mktcap) numa única chamada. Inclui ^GSPC.
  2. Shares outstanding (get_shares_full; cobertura 2015+, flag de extrapolação
     fica para a fase C).
  3. Earnings (EPS estimado × reportado, limit=100) para o SUE.

Saídas (data/raw/sp500/): price_shards/chunk_XX.parquet,
shares_shards/block_XX.parquet, earnings_shards/block_XX.parquet.
Uso: python scripts/sp500_phase_b.py
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
RAW = ROOT / "data" / "raw"
OUT = RAW / "sp500"
PRICE_CHUNK, BLOCK = 25, 50
SYMBOL_OVERRIDES = {"FI": "FISV"}   # símbolo no Yahoo difere do canônico


def main():
    import yfinance as yf
    for d in (OUT / "price_shards", OUT / "shares_shards", OUT / "earnings_shards"):
        d.mkdir(parents=True, exist_ok=True)
    calls = pd.read_parquet(RAW / "calls_all.parquet")
    tickers = sorted(calls["ticker"].dropna().unique())
    print(f"tickers alvo: {len(tickers)} + ^GSPC", flush=True)

    # ---------- 1) preços (chunks de 25) ----------
    groups = [tickers[i:i + PRICE_CHUNK] for i in range(0, len(tickers), PRICE_CHUNK)]
    groups.append(["^GSPC"])
    t0 = time.time()
    for gi, grp in enumerate(groups):
        shard = OUT / "price_shards" / f"chunk_{gi:03d}.parquet"
        if shard.exists():
            continue
        syms = [SYMBOL_OVERRIDES.get(t, t) for t in grp]
        rows = []
        try:
            d = yf.download(syms, start="2004-06-01", auto_adjust=False,
                            progress=False, group_by="ticker", threads=True)
            for t, sym in zip(grp, syms):
                try:
                    sub = d[sym] if len(syms) > 1 else d
                    sub = sub.dropna(subset=["Close"])
                    if sub.empty:
                        continue
                    r = pd.DataFrame({
                        "date": pd.to_datetime(sub.index).tz_localize(None),
                        "ticker": t,
                        "adj_close": sub["Adj Close"].to_numpy(float),
                        "raw_close": sub["Close"].to_numpy(float),
                        "volume": sub["Volume"].to_numpy(float),
                    })
                    rows.append(r)
                except Exception:
                    continue
        except Exception as e:
            print(f"  preços chunk {gi}: ERRO {str(e)[:60]}", flush=True)
        if rows:
            pd.concat(rows, ignore_index=True).to_parquet(shard, index=False)
        else:  # marca vazio p/ não refazer eternamente
            pd.DataFrame(columns=["date", "ticker", "adj_close", "raw_close", "volume"]).to_parquet(shard, index=False)
        if (gi + 1) % 5 == 0:
            print(f"  preços: chunk {gi+1}/{len(groups)} ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(1)
    print(f"preços OK ({time.time()-t0:.0f}s)", flush=True)

    # ---------- 2) shares + 3) earnings (blocos de 50 tickers) ----------
    blocks = [tickers[i:i + BLOCK] for i in range(0, len(tickers), BLOCK)]
    for bi, blk in enumerate(blocks):
        sh_shard = OUT / "shares_shards" / f"block_{bi:03d}.parquet"
        ea_shard = OUT / "earnings_shards" / f"block_{bi:03d}.parquet"
        if sh_shard.exists() and ea_shard.exists():
            continue
        sh_rows, ea_rows = [], []
        for t in blk:
            sym = SYMBOL_OVERRIDES.get(t, t)
            tk = yf.Ticker(sym)
            try:
                s = tk.get_shares_full(start="2004-01-01")
                if s is not None and len(s):
                    d = s.reset_index(); d.columns = ["date", "shares"]
                    d["date"] = pd.to_datetime(d["date"], utc=True).dt.tz_localize(None).dt.normalize()
                    d["ticker"] = t
                    sh_rows.append(d)
            except Exception:
                pass
            try:
                e = tk.get_earnings_dates(limit=100)
                if e is not None and len(e):
                    d = e.reset_index()
                    d = d.rename(columns={d.columns[0]: "edate", "EPS Estimate": "eps_est",
                                          "Reported EPS": "eps_act"})
                    d["edate"] = pd.to_datetime(d["edate"], utc=True).dt.tz_localize(None).dt.normalize()
                    d["ticker"] = t
                    ea_rows.append(d[["ticker", "edate", "eps_est", "eps_act"]])
            except Exception:
                pass
            time.sleep(0.2)
        (pd.concat(sh_rows, ignore_index=True) if sh_rows else
         pd.DataFrame(columns=["date", "shares", "ticker"])).to_parquet(sh_shard, index=False)
        (pd.concat(ea_rows, ignore_index=True) if ea_rows else
         pd.DataFrame(columns=["ticker", "edate", "eps_est", "eps_act"])).to_parquet(ea_shard, index=False)
        print(f"  shares/earnings: bloco {bi+1}/{len(blocks)} ({time.time()-t0:.0f}s)", flush=True)

    # ---------- resumo de cobertura ----------
    px = pd.concat([pd.read_parquet(p) for p in (OUT / "price_shards").glob("*.parquet")], ignore_index=True)
    sh = pd.concat([pd.read_parquet(p) for p in (OUT / "shares_shards").glob("*.parquet")], ignore_index=True)
    ea = pd.concat([pd.read_parquet(p) for p in (OUT / "earnings_shards").glob("*.parquet")], ignore_index=True)
    print(f"\nCOBERTURA: preços {px['ticker'].nunique()}/{len(tickers)} | "
          f"shares {sh['ticker'].nunique()} | earnings {ea['ticker'].nunique()}", flush=True)
    print(f"FASE B concluída em {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
