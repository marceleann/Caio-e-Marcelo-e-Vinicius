# -*- coding: utf-8 -*-
"""Reavaliação factual do universo com setor Yahoo ATUAL (sector_cache) — READ-ONLY."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INT = ROOT / "data" / "interim"
cache = pd.read_parquet(ROOT / "data" / "raw" / "sector_cache.parquet").set_index("ticker")
uni = pd.read_parquet(INT / "universe.parquet")
calls = pd.read_parquet(INT / "calls.parquet")

n_by = calls.groupby("ticker").size().rename("n_calls")
yr = calls.groupby("ticker")["year"].agg(["min", "max"])
uni = uni.merge(n_by, left_on="ticker", right_index=True, how="left")
uni = uni.merge(yr, left_on="ticker", right_index=True, how="left")
uni["n_calls"] = uni["n_calls"].fillna(0).astype(int)
uni = uni[uni["n_calls"] > 0].copy()

# setor ATUAL via cache (fallback p/ o que já havia no universe.parquet)
def sec(t):
    if t in cache.index: return cache.loc[t, "sector"]
    return None
def ind(t):
    if t in cache.index: return cache.loc[t, "industry"]
    return None
# considerar override FI->FISV
alias_price = {"FI": "FISV"}
uni["ysector"] = uni["ticker"].map(lambda t: sec(alias_price.get(t, t)))
uni["yindustry"] = uni["ticker"].map(lambda t: ind(alias_price.get(t, t)))
uni["block"] = uni["sensitivity_block"].fillna("").replace("", "core")

# setores considerados "tech de fato" para o sanity
TECH_SECTORS = {"Technology"}
TECH_COMM_INDUSTRIES = {"Internet Content & Information", "Electronic Gaming & Multimedia"}

def classify(r):
    if r["ysector"] in TECH_SECTORS: return "Tech (GICS/Yahoo)"
    if r["yindustry"] in TECH_COMM_INDUSTRIES: return "Tech (internet/games)"
    if pd.isna(r["ysector"]): return "DESLISTADO (sem Yahoo)"
    return f"NÃO-tech Yahoo: {r['ysector']}"

uni["veredito"] = uni.apply(classify, axis=1)

print("=" * 100)
print("REAVALIAÇÃO — cada ticker com call, bloco atual x setor Yahoo atual x veredito")
print("=" * 100)
for block, g in uni.groupby("block"):
    print(f"\n########## BLOCO = {block}  ({len(g)} tickers, {int(g['n_calls'].sum())} calls) ##########")
    for _, r in g.sort_values(["veredito", "n_calls"], ascending=[True, False]).iterrows():
        print(f"  {r['ticker']:6s} n={r['n_calls']:4d} [{int(r['min'])}-{int(r['max'])}]  "
              f"{str(r['ysector'] or '—')[:20]:20s} | {str(r['yindustry'] or '—')[:34]:34s} -> {r['veredito']}")

print("\n" + "=" * 100)
print("FOCO: tickers no NÚCLEO cujo setor Yahoo NÃO é 'Technology' nem internet/games")
print("(candidatos a um bloco 'borderline' — decisão do Marcelo)")
print("=" * 100)
core = uni[uni["block"] == "core"]
border = core[~core["veredito"].str.startswith("Tech")]
# separar deslistados (sem Yahoo) dos que têm setor não-tech explícito
delisted = border[border["veredito"].str.startswith("DESLISTADO")]
nontech = border[~border["veredito"].str.startswith("DESLISTADO")]
print(f"\n-- núcleo com setor Yahoo NÃO-tech explícito ({len(nontech)} tickers, {int(nontech['n_calls'].sum())} calls):")
for _, r in nontech.sort_values("n_calls", ascending=False).iterrows():
    print(f"  {r['ticker']:6s} n={r['n_calls']:4d}  {r['ysector']} | {r['yindustry']}")
print(f"\n-- núcleo DESLISTADO (sem Yahoo hoje; eram tech na época) ({len(delisted)} tickers, {int(delisted['n_calls'].sum())} calls):")
print("  " + ", ".join(sorted(delisted["ticker"].tolist())))
print(f"\n-- núcleo confirmado Tech pelo Yahoo: {int((core['veredito'].str.startswith('Tech')).sum())} tickers")

print("\nDONE.")
