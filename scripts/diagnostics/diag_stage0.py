# -*- coding: utf-8 -*-
"""Diagnóstico Stage 0 — READ-ONLY. Reproduz P1/P2, reconcilia contagens,
estima o compute do re-scoring sentence-level. Não escreve nada em disco."""
import re
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INT = ROOT / "data" / "interim"

def load(name):
    return pd.read_parquet(INT / name)

print("=" * 70)
print("[COLS] schema real das tabelas")
print("=" * 70)
scores = load("utterance_scores.parquet")
roles = load("utterances_roles.parquet")
calls = load("calls.parquet")
universe = load("universe.parquet")
for nm, df in [("utterance_scores", scores), ("utterances_roles", roles),
               ("calls", calls), ("universe", universe)]:
    print(f"\n{nm}: shape={df.shape}")
    print("  cols:", list(df.columns))

# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("[P1] saturação do FinBERT — distribuição de p_pos (utterance_scores)")
print("=" * 70)
pp = scores["p_pos"].to_numpy(dtype=np.float64)
n = len(pp)
def pct(mask): return 100.0 * mask.sum() / n
print(f"n falas pontuadas: {n}")
print(f"  p_pos < 0.01      : {pct(pp < 0.01):5.1f}%   (prompt: 53.5%)")
print(f"  p_pos > 0.99      : {pct(pp > 0.99):5.1f}%   (prompt: 21.4%)")
print(f"  0.05 < p_pos<0.95 : {pct((pp > 0.05) & (pp < 0.95)):5.1f}%   (prompt: 16.5%)")
# também p_neg para contexto
pn = scores["p_neg"].to_numpy(dtype=np.float64)
print(f"  (contexto) p_neg<0.01: {pct(pn<0.01):5.1f}% | p_neg>0.99: {pct(pn>0.99):5.1f}%")

# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("[RECON] contagens de calls / tickers / universo")
print("=" * 70)
print(f"calls.parquet: n_linhas={len(calls)}, call_id únicos={calls['call_id'].nunique()}")
for c in ("ticker", "ticker_raw"):
    if c in calls.columns:
        print(f"  {c} únicos: {calls[c].nunique()}")
rng = [c for c in ("call_datetime", "date", "datetime") if c in calls.columns]
if rng:
    col = rng[0]
    print(f"  período ({col}): {calls[col].min()} .. {calls[col].max()}")
print(f"universe.parquet: n_linhas={len(universe)}")
if "ticker" in universe.columns:
    print(f"  tickers únicos no universo: {universe['ticker'].nunique()}")
    # quantos do universo têm call no dataset
    if "ticker" in calls.columns:
        u = set(universe["ticker"].unique()); c = set(calls["ticker"].unique())
        print(f"  universo∩calls: {len(u & c)} | universo sem call: {len(u - c)} | calls fora do universo: {len(c - u)}")

# ----------------------------------------------------------------------
# Réplica das funções de identidade de gestor (roles.py), sem depender do pacote.
_QA_PREFIX_RE = re.compile(r"^[QA]\s*[-–:]\s*")
_TITLE_SUFFIX_RE = re.compile(r"\s+[-–—]\s+.*$")
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
def _clean_name(name):
    if name is None or not isinstance(name, str):
        if name is None or pd.isna(name):
            return ""
        name = str(name)
    name = _QA_PREFIX_RE.sub("", name.strip())
    name = _TITLE_SUFFIX_RE.sub("", name)
    name = _CAMEL_RE.sub(" ", name)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", name).strip().casefold()
def _match_key(name):
    cleaned = _clean_name(name)
    if not cleaned:
        return ""
    tokens = [t for t in cleaned.replace(".", " ").split() if len(t) > 1]
    if not tokens:
        return cleaned
    return f"{tokens[-1]}|{tokens[0][0]}"

print("\n" + "=" * 70)
print("[P2] Tone Distance reconstruída e correlações com volume de fala")
print("=" * 70)
MIN_TOK = 25   # config atual: features.tone_distance.min_manager_tokens
MIN_MGR = 2
# anexar speaker aos scores (scores não traz speaker; roles traz)
key_cols = ["call_id", "utterance_idx"]
if "speaker" not in scores.columns:
    sc = scores.merge(roles[key_cols + ["speaker"]], on=key_cols, how="left")
else:
    sc = scores.copy()
# só calls com Q&A detectado (mesma regra do código atual)
qa_calls = sc.loc[sc["section"] == "qa", "call_id"].unique()
mgmt = sc[(sc["role"] == "management") & (sc["call_id"].isin(qa_calls))].copy()
mgmt["_mkey"] = mgmt["speaker"].map(_match_key)
mgmt = mgmt[mgmt["_mkey"] != ""]
tok = mgmt["n_tokens"].to_numpy(dtype=np.float64)
mgmt["_wpos"] = mgmt["p_pos"].to_numpy(dtype=np.float64) * tok
mgmt["_wneg"] = mgmt["p_neg"].to_numpy(dtype=np.float64) * tok
per_mgr = (mgmt.groupby(["call_id", "_mkey"], sort=False)
           .agg(tok=("n_tokens", "sum"), wpos=("_wpos", "sum"), wneg=("_wneg", "sum"))
           .reset_index())
per_mgr = per_mgr[per_mgr["tok"] >= MIN_TOK]
per_mgr["Pos"] = per_mgr["wpos"] / per_mgr["tok"]
per_mgr["Neg"] = per_mgr["wneg"] / per_mgr["tok"]

rows = []
for call_id, sub in per_mgr.groupby("call_id", sort=False):
    nmgr = len(sub)
    if nmgr < MIN_MGR:
        continue
    pos = sub["Pos"].to_numpy(); neg = sub["Neg"].to_numpy()
    td = float(np.sqrt((pos - pos.mean())**2 + (neg - neg.mean())**2).mean())
    rows.append({"call_id": call_id, "td": td, "n_managers": nmgr,
                 "min_tok": float(sub["tok"].min()),
                 "avg_pos": float(sub["Pos"].mean())})
td_df = pd.DataFrame(rows)
print(f"calls com Q&A detectado: {len(qa_calls)}")
print(f"calls com Tone Distance (>= {MIN_MGR} gestores, >= {MIN_TOK} tokens): {len(td_df)}")
print(f"\nTone Distance: mean={td_df['td'].mean():.4f} median={td_df['td'].median():.4f} "
      f"sd={td_df['td'].std():.4f} CV={td_df['td'].std()/td_df['td'].mean():.3f}")
print("  (paper Angelo LM: mean 0.0079 sd 0.0048 CV 0.61)")
print("\ncorrelações (Pearson):")
print(f"  corr(TD, tokens do gestor mais quieto): {td_df['td'].corr(td_df['min_tok']):+.3f}   (prompt: -0.545)")
print(f"  corr(TD, nº de gestores)              : {td_df['td'].corr(td_df['n_managers']):+.3f}   (prompt: +0.157)")
print(f"  corr(TD, tom positivo médio da call)  : {td_df['td'].corr(td_df['avg_pos']):+.3f}   (prompt: +0.017)")
print("\nmédia de TD por nº de gestores (prompt: 2→0.224, 3→0.307, 4→0.314):")
g = td_df.groupby("n_managers")["td"].agg(["mean", "count"])
print(g.head(8).to_string())

# sensibilidade de min_manager_tokens (nº de calls sobreviventes) — recontagem leve
print("\nsensibilidade min_manager_tokens (calls com >=2 gestores):")
base = (mgmt.groupby(["call_id", "_mkey"], sort=False)["n_tokens"].sum().reset_index())
for thr in (25, 100, 250, 500):
    sub = base[base["n_tokens"] >= thr]
    cnt = sub.groupby("call_id").size()
    print(f"  >= {thr:4d} tokens: {(cnt >= 2).sum()} calls")

# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("[STAGE2] estimativa de compute do re-scoring sentence-level")
print("=" * 70)
# forward passes ATUAIS = nº de chunks já pontuados
cur_fwd = int(scores["n_chunks"].sum()) if "n_chunks" in scores.columns else len(scores)
print(f"falas pontuadas: {len(scores)} | forward passes ATUAIS (n_chunks): {cur_fwd}")
# contar sentenças nas falas elegíveis (role mgmt/analyst, n_chars>=20)
_SENT = re.compile(r"[.!?]+(?:\s|$)")
elig = roles.copy()
if "n_chars" in elig.columns:
    elig = elig[elig["n_chars"] >= 20]
elig = elig[elig["role"].isin(["management", "analyst"])]
elig = elig[elig["text"].astype(str).str.strip() != ""]
def n_sent(t):
    t = str(t)
    k = len(_SENT.findall(t))
    return k if k > 0 else 1
sent_counts = elig["text"].map(n_sent)
tot_sent = int(sent_counts.sum())
print(f"falas elegíveis: {len(elig)} | sentenças estimadas: {tot_sent}")
print(f"  média sentenças/fala: {sent_counts.mean():.1f} | mediana: {sent_counts.median():.0f} | máx: {sent_counts.max()}")
print(f"multiplicador de forward passes (sentenças / chunks atuais): {tot_sent / cur_fwd:.1f}x")
print("  -> runtime real depende do device; próximo passo é micro-benchmark (score de N sentenças, medir).")
print("\nDONE.")
