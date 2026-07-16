# -*- coding: utf-8 -*-
"""Industry Tone EXATO do paper: Fama-French 49 via SIC do EDGAR.

Problema detectado: o industry_tone por setor GICS (sector_cache do yfinance)
tem `sector` nulo em ~23% dos eventos (e ~95% do subset tech), custando amostra
em TODAS as regressões via dropna. O Apêndice A do Angelo define: "Industry
Tone is the average Transcript Tone for a given industry in a quarter.
Industry is based on the Fama French 49 industry classification."

Este script:
  1. busca o SIC de cada CIK em data.sec.gov/submissions (fonte primária);
  2. baixa Siccodes49 do site do Ken French e monta o mapa SIC->FF49
     (sem range correspondente -> indústria 48 "Other", convenção padrão);
  3. industry_tone_ff49 = média leave-one-out do disclosure_tone das OUTRAS
     firmas da MESMA indústria FF49 no MESMO ano-trimestre (>=3 pares —
     contemporâneo ao trimestre, como no paper; é controle, não sinal);
  4. grava a coluna em events_sp500_car3.parquet e events_sp500_paper.parquet.
Caches: data/raw/cik_sic.parquet, data/raw/Siccodes49.txt.
Uso: python scripts/sp500_ff49.py
"""
from __future__ import annotations
import io, re, time, zipfile
import numpy as np
import pandas as pd
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTDIR = ROOT / "data" / "interim" / "sp500"
UA = {"User-Agent": "Projeto academico Desafio Quant AI - contato mns2026@example.com"}
PEER_MIN = 3


def fetch_sic():
    cache = RAW / "cik_sic.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    cmap = pd.read_parquet(RAW / "cik_map.parquet")
    rows = []
    for i, r in enumerate(cmap.itertuples()):
        url = f"https://data.sec.gov/submissions/CIK{int(r.cik):010d}.json"
        try:
            j = requests.get(url, headers=UA, timeout=30).json()
            sic = j.get("sic")
            rows.append((r.ticker, int(r.cik), int(sic) if sic else None,
                         j.get("sicDescription")))
        except Exception as e:
            rows.append((r.ticker, int(r.cik), None, f"ERRO {type(e).__name__}"))
        if (i + 1) % 50 == 0:
            print(f"  SIC: {i+1}/{len(cmap)}", flush=True)
        time.sleep(0.12)   # cortesia com a SEC (~8 req/s)
    df = pd.DataFrame(rows, columns=["ticker", "cik", "sic", "sic_desc"])
    df.to_parquet(cache, index=False)
    print(f"SIC: {df['sic'].notna().sum()}/{len(df)} tickers com código", flush=True)
    return df


def fetch_ff49():
    txt_cache = RAW / "Siccodes49.txt"
    if not txt_cache.exists():
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes49.zip"
        z = zipfile.ZipFile(io.BytesIO(requests.get(url, headers=UA, timeout=60).content))
        name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
        txt_cache.write_bytes(z.read(name))
    ranges = []          # (lo, hi, ind_num, ind_abbr)
    ind_num, ind_abbr = None, None
    hdr = re.compile(r"^\s*(\d+)\s+(\S+)\s")
    rng = re.compile(r"^\s+(\d{4})-(\d{4})")
    for line in txt_cache.read_text().splitlines():
        m = hdr.match(line)
        if m and not line.startswith(" " * 6):
            ind_num, ind_abbr = int(m.group(1)), m.group(2)
            continue
        m = rng.match(line)
        if m and ind_num is not None:
            ranges.append((int(m.group(1)), int(m.group(2)), ind_num, ind_abbr))
    print(f"FF49: {len(ranges)} faixas SIC em {len(set(r[2] for r in ranges))} indústrias",
          flush=True)
    return ranges


def main():
    sic = fetch_sic()
    ranges = fetch_ff49()

    def to_ff49(code):
        if code is None or not np.isfinite(code):
            return None
        c = int(code)
        for lo, hi, num, abbr in ranges:
            if lo <= c <= hi:
                return abbr
        return "Other"    # convenção: sem faixa -> 48 Other

    sic["ff49"] = sic["sic"].map(to_ff49)
    print(sic["ff49"].value_counts().head(12).to_string(), flush=True)
    fmap = sic.set_index("ticker")["ff49"].to_dict()

    for fn in ["events_sp500_car3.parquet", "events_sp500_paper.parquet"]:
        ev = pd.read_parquet(OUTDIR / fn)
        ev["ff49"] = ev["ticker"].map(fmap)
        # média leave-one-out por (ff49, year_quarter)
        g = ev.groupby(["ff49", "year_quarter"])["disclosure_tone"]
        s, n = g.transform("sum"), g.transform("count")
        loo = (s - ev["disclosure_tone"].fillna(0)) / (n - ev["disclosure_tone"].notna())
        loo[(n - ev["disclosure_tone"].notna()) < PEER_MIN] = np.nan
        ev["industry_tone_ff49"] = loo
        ev.to_parquet(OUTDIR / fn, index=False)
        print(f"{fn}: industry_tone_ff49 {ev['industry_tone_ff49'].notna().sum()}"
              f"/{len(ev)} ({100*ev['industry_tone_ff49'].notna().mean():.0f}%) | "
              f"GICS antigo: {100*ev['industry_tone'].notna().mean():.0f}%", flush=True)
    print("DONE.")


if __name__ == "__main__":
    main()
