# -*- coding: utf-8 -*-
"""Gera data/interim/call_blocks.parquet: cada call_id -> bloco de universo.
NÃO altera nada existente; só cria um artefato de ETIQUETA para o 'com/sem'."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INT = ROOT / "data" / "interim"
calls = pd.read_parquet(INT / "calls.parquet")
uni = pd.read_parquet(INT / "universe.parquet")[["ticker", "sensitivity_block"]]

# candidato novo (decisão do Marcelo amanhã): tech "de fato" discutível
CORE_BORDERLINE = {"GRMN", "TDY", "FTV", "VNT", "ROP", "LDOS", "UIS"}

df = calls[["call_id", "ticker", "year"]].merge(uni, on="ticker", how="left")
def block_of(r):
    b = r["sensitivity_block"]
    if isinstance(b, str) and b.strip():
        return b
    if r["ticker"] in CORE_BORDERLINE:
        return "core_borderline"
    return "core"
df["block"] = df.apply(block_of, axis=1)
# universo "estrito tech" = core + core_borderline (sem os 5 blocos econômicos)
df["in_strict_tech"] = df["block"].isin(["core", "core_borderline"])
# universo "core puro" = só core (sem borderline nem blocos)
df["in_core_pure"] = df["block"].eq("core")

out = df[["call_id", "ticker", "block", "in_strict_tech", "in_core_pure"]].copy()
dest = INT / "call_blocks.parquet"
out.to_parquet(dest, index=False)
print(f"Escrito: {dest}  ({len(out)} calls)")
print("\nCalls por bloco:")
print(out.groupby("block").agg(calls=("call_id", "size"),
                               tickers=("ticker", "nunique")).to_string())
print(f"\nin_strict_tech (core+borderline): {int(out['in_strict_tech'].sum())} calls, "
      f"{out.loc[out['in_strict_tech'],'ticker'].nunique()} tickers")
print(f"in_core_pure (só core):           {int(out['in_core_pure'].sum())} calls, "
      f"{out.loc[out['in_core_pure'],'ticker'].nunique()} tickers")
print("DONE.")
