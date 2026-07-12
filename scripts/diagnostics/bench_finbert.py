# -*- coding: utf-8 -*-
"""Micro-benchmark do FinBERT em nível de SENTENÇA — mede throughput real
para estimar o runtime do Stage 2. Read-only (não escreve scores)."""
import re, time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INT = ROOT / "data" / "interim"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL = "yiyanghkust/finbert-tone"
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | CUDA disponível: {torch.cuda.is_available()} | device usado: {dev}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"CPU threads: {torch.get_num_threads()}")

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
print(f"modelo carregado em {time.time()-t0:.1f}s")

# extrair sentenças reais das falas elegíveis
roles = pd.read_parquet(INT / "utterances_roles.parquet")
elig = roles[(roles["n_chars"] >= 20) & (roles["role"].isin(["management", "analyst"]))]
_SPLIT = re.compile(r"(?<=[.!?])\s+")
sents = []
for t in elig["text"].head(5000):
    for s in _SPLIT.split(str(t)):
        s = s.strip()
        if len(s) >= 3:
            sents.append(s)
        if len(sents) >= 6000:
            break
    if len(sents) >= 6000:
        break
print(f"sentenças coletadas p/ benchmark: {len(sents)}")
lens = [len(tok.encode(s, add_special_tokens=True)) for s in sents]
print(f"tokens/sentença: média {np.mean(lens):.1f} | mediana {np.median(lens):.0f} | p95 {np.percentile(lens,95):.0f} | máx {max(lens)}")

def run(texts, batch_size, max_len=512, sort=True):
    order = np.argsort([len(x) for x in texts]) if sort else np.arange(len(texts))
    texts = [texts[i] for i in order]
    t = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b = texts[i:i+batch_size]
            enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(dev)
            _ = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
    return time.time() - t

# aquecer
_ = run(sents[:64], 16)
for bs in (16, 32, 64):
    N = 2000
    dt = run(sents[:N], bs)
    rate = N / dt
    print(f"batch={bs:3d}: {N} sentenças em {dt:5.1f}s -> {rate:6.1f} sent/s")

# extrapolação com o melhor rate (batch=32 costuma ser bom)
best_dt = run(sents[:2000], 32)
rate = 2000 / best_dt
TOTAL = 2_384_498
horas = TOTAL / rate / 3600
print("\n" + "=" * 60)
print(f"EXTRAPOLAÇÃO Stage 2 (device={dev}):")
print(f"  rate ~{rate:.0f} sent/s -> {TOTAL} sentenças ≈ {horas:.1f} h")
print("=" * 60)
print("DONE.")
