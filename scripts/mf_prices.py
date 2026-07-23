# -*- coding: utf-8 -*-
"""Extensão small/mid caps — FASE MF-3: preços diários (yfinance, resumável).

Busca adj_close + close bruto para os tickers do universo Motley Fool que NÃO
estão nos shards do S&P 500 (estes são reaproveitados). Janela 2018-06-01 a
2023-09-01 (estimação CAPM de 100 pregões antes do 1º evento de 2019-04 +
janelas pós-evento até ~6m depois do último de 2023-02).

Mecânica idêntica à fase B do S&P: blocos de 100 tickers, 1 parquet por bloco
em data/raw/mf/price_shards (resumável; bloco existente é pulado), colunas
achatadas, tratamento de MultiIndex p/ símbolo único. Cobertura esperada
PARCIAL (small caps deslistadas sem histórico no yfinance) — % reportado e
declarado como limitação, não silenciado.
Uso: python scripts/mf_prices.py
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MF = ROOT / "data" / "interim" / "mf"
OUTD = ROOT / "data" / "raw" / "mf" / "price_shards"
START, END = "2018-06-01", "2023-09-01"
BLOCK = 100


def fetch_block(tks):
    import yfinance as yf
    df = yf.download(tks, start=START, end=END, auto_adjust=False,
                     progress=False, group_by="ticker", threads=True)
    rows = []
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "date", "adj_close", "raw_close"])
    if not isinstance(df.columns, pd.MultiIndex):     # símbolo único
        df = pd.concat({tks[0]: df}, axis=1)
    for t in tks:
        if t not in df.columns.get_level_values(0):
            continue
        sub = df[t].dropna(subset=["Adj Close"])
        if sub.empty:
            continue
        rows.append(pd.DataFrame(dict(
            ticker=t, date=sub.index,
            adj_close=sub["Adj Close"].to_numpy(),
            raw_close=sub["Close"].to_numpy())))
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "adj_close", "raw_close"])
    return pd.concat(rows, ignore_index=True)


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    cm = pd.read_parquet(MF / "calls_mf.parquet")
    sp_have = set()
    spdir = ROOT / "data" / "raw" / "sp500" / "price_shards"
    for p in spdir.glob("*.parquet"):
        sp_have.update(pd.read_parquet(p, columns=["ticker"])["ticker"].unique())
    tks = sorted(set(cm["ticker"]) - sp_have)
    print(f"tickers MF: {cm['ticker'].nunique()} | já no S&P: "
          f"{cm['ticker'].nunique() - len(tks)} | a buscar: {len(tks)}", flush=True)

    t0 = time.time()
    got = set()
    for i0 in range(0, len(tks), BLOCK):
        dest = OUTD / f"block_{i0//BLOCK:03d}.parquet"
        blk = tks[i0:i0 + BLOCK]
        if dest.exists():
            got.update(pd.read_parquet(dest, columns=["ticker"])["ticker"].unique())
            continue
        df = fetch_block(blk)
        df.to_parquet(dest, index=False)
        got.update(df["ticker"].unique())
        print(f"  bloco {i0//BLOCK + 1}/{(len(tks)+BLOCK-1)//BLOCK}: "
              f"{df['ticker'].nunique()}/{len(blk)} tickers com preço "
              f"({time.time()-t0:.0f}s)", flush=True)
        time.sleep(1.0)     # cortesia com o Yahoo
    print(f"\ncobertura: {len(got)}/{len(tks)} tickers novos com preços "
          f"({100*len(got)/max(len(tks),1):.0f}%)")
    print("DONE.")


if __name__ == "__main__":
    main()
