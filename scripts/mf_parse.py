# -*- coding: utf-8 -*-
"""Extensão small/mid caps — FASE MF-1: parser do dataset Motley Fool.

Trilho ISOLADO do pipeline S&P 500 (nada em sp500_* é tocado). Transforma
data/raw/motleyfool/motley-fool-data.pkl (18.755 calls, 2019-04 a 2023-02,
2.876 tickers) em:
  data/interim/mf/calls_mf.parquet       (metadados por call)
  data/interim/mf/utterances_mf.parquet  (uma linha por fala, ordem preservada)

Formato de origem: texto com seções "Prepared Remarks:" / "Questions & Answers:"
e cabeçalho de orador em linha própria: "Nome -- Cargo/Afiliação" ou "Operator".
Regras:
  - call_id = TICKER_YYYYQQ (da coluna q, ex. 2020-Q2 -> 2020Q2);
  - datas "Aug 27, 2020, 9:00 p.m. ET" -> datetime (ET); sem hora -> has_time=False;
  - duplicatas de (ticker, q): mantém a de transcrição mais longa;
  - datas não parseáveis: descartadas e contadas;
  - fala = linhas entre cabeçalhos de orador; seção corrente pelo marcador.
Validação impressa: nº de calls/falas, % com >=2 oradores com cabeçalho
"--", distribuição de falas por call, e 1 call de amostra reconstruída.
Uso: python scripts/mf_parse.py
"""
from __future__ import annotations
import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "interim" / "mf"

SEC_REM = re.compile(r"^\s*Prepared Remarks:?\s*$", re.I)
SEC_QA = re.compile(r"^\s*Questions\s*(?:&|and)\s*Answers:?\s*$", re.I)
# cabeçalho de orador: "Nome -- Cargo" (2+ traços) ou linhas especiais
SPK = re.compile(r"^\s*([A-Z][^\n]{1,80}?)\s*--\s*([^\n]{2,100})\s*$")
OP = re.compile(r"^\s*Operator\s*$", re.I)
DUR = re.compile(r"^\s*Duration:\s", re.I)


def parse_one(text: str):
    """-> lista de (utterance_idx, speaker, section, texto)."""
    rows = []
    section = "remarks"
    speaker = None
    buf = []
    idx = 0

    def flush():
        nonlocal idx, buf
        if speaker is not None and buf:
            txt = " ".join(x.strip() for x in buf if x.strip())
            if txt:
                rows.append((idx, speaker, section, txt))
                idx += 1
        buf = []

    for line in text.split("\n"):
        if SEC_REM.match(line):
            flush(); section = "remarks"; continue
        if SEC_QA.match(line):
            flush(); section = "qa"; continue
        if DUR.match(line):
            flush(); speaker = None; continue
        if OP.match(line):
            flush(); speaker = "Operator"; continue
        m = SPK.match(line)
        if m:
            flush()
            speaker = f"{m.group(1).strip()} -- {m.group(2).strip()}"
            continue
        buf.append(line)
    flush()
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(ROOT / "data" / "raw" / "motleyfool" / "motley-fool-data.pkl")
    n0 = len(df)

    # datas (ET; "mixed" cobre com e sem horário)
    dt = pd.to_datetime(df["date"].astype(str).str.replace(" ET", "", regex=False),
                        format="mixed", errors="coerce")
    df = df.assign(call_datetime=dt)
    df = df[df["call_datetime"].notna()].copy()
    df["has_time"] = df["date"].astype(str).str.contains(r"\d\s*[ap]\.m\.", regex=True)
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["qq"] = df["q"].astype(str).str.replace("-", "", regex=False)
    df["call_id"] = df["ticker"] + "_" + df["qq"]
    df["tlen"] = df["transcript"].str.len()
    df = (df.sort_values("tlen", ascending=False)
            .drop_duplicates("call_id", keep="first"))
    print(f"calls: {n0} brutas -> {len(df)} únicas com data "
          f"({n0 - len(df)} descartadas: data inválida/duplicata)", flush=True)

    utts, meta = [], []
    for r in df.itertuples():
        rows = parse_one(r.transcript)
        n_spk = len({s for _, s, _, _ in rows if "--" in s})
        meta.append(dict(call_id=r.call_id, ticker=r.ticker,
                         call_datetime=r.call_datetime, has_time=bool(r.has_time),
                         year=int(r.call_datetime.year),
                         quarter=int(str(r.qq)[-1]), n_utt=len(rows),
                         n_named_speakers=n_spk))
        for idx, spk, sec, txt in rows:
            utts.append((r.call_id, idx, spk, sec, txt))
    um = pd.DataFrame(utts, columns=["call_id", "utterance_idx", "speaker", "section", "text"])
    cm = pd.DataFrame(meta)
    um.to_parquet(OUT / "utterances_mf.parquet", index=False)
    cm.to_parquet(OUT / "calls_mf.parquet", index=False)

    print(f"falas: {len(um)} | calls com >=2 oradores nomeados: "
          f"{(cm['n_named_speakers'] >= 2).mean()*100:.1f}%", flush=True)
    print(f"falas por call: mediana {cm['n_utt'].median():.0f}, "
          f"p5 {cm['n_utt'].quantile(.05):.0f}, p95 {cm['n_utt'].quantile(.95):.0f}")
    print(f"% falas em Q&A: {(um['section'] == 'qa').mean()*100:.0f}%")
    print(f"% com has_time: {cm['has_time'].mean()*100:.0f}%")
    print("\n=== AMOSTRA (primeiras 6 falas de", cm['call_id'].iloc[0], ") ===")
    s = um[um["call_id"] == cm["call_id"].iloc[0]].head(6)
    for r in s.itertuples():
        print(f"[{r.section}] {r.speaker}: {r.text[:110]}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
