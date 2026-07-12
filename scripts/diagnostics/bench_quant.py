# -*- coding: utf-8 -*-
"""Testa FinBERT quantizado int8 (dynamic) — throughput E concordância vs fp32.
Read-only. Decide se dá para usar quantização no Stage 2."""
import re, time
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    ROOT = Path(__file__).resolve().parents[2]
    INT = ROOT / "data" / "interim"
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(10)
    MODEL = "yiyanghkust/finbert-tone"
    tok = AutoTokenizer.from_pretrained(MODEL)
    m_fp32 = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    # ordem de labels por NOME (nunca por índice)
    id2label = {int(k): str(v).casefold() for k, v in m_fp32.config.id2label.items()}
    keys = {"neg": "negative", "neu": "neutral", "pos": "positive"}
    order = {}
    for i, lab in id2label.items():
        for k, sub in keys.items():
            if sub in lab: order[k] = i
    print("id2label:", m_fp32.config.id2label, "-> order(neg,neu,pos):",
          (order["neg"], order["neu"], order["pos"]))
    i_neg, i_neu, i_pos = order["neg"], order["neu"], order["pos"]

    m_q = torch.quantization.quantize_dynamic(m_fp32, {torch.nn.Linear}, dtype=torch.qint8).eval()

    roles = pd.read_parquet(INT / "utterances_roles.parquet")
    elig = roles[(roles["n_chars"] >= 20) & (roles["role"].isin(["management", "analyst"]))]
    _SPLIT = re.compile(r"(?<=[.!?])\s+")
    sents = []
    for t in elig["text"].head(4000):
        for s in _SPLIT.split(str(t)):
            s = s.strip()
            if len(s) >= 3: sents.append(s)
        if len(sents) >= 1200: break
    sents = sents[:1000]
    print(f"sentenças p/ teste: {len(sents)}")

    def score(model, texts, bs=32):
        order_idx = np.argsort([len(x) for x in texts])
        texts_s = [texts[i] for i in order_idx]
        out = np.empty((len(texts), 3))
        t = time.time()
        with torch.no_grad():
            for i in range(0, len(texts_s), bs):
                b = texts_s[i:i+bs]
                enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=128)
                p = torch.softmax(model(**enc).logits, dim=-1).numpy()
                out[i:i+len(b), 0] = p[:, i_neg]; out[i:i+len(b), 1] = p[:, i_neu]; out[i:i+len(b), 2] = p[:, i_pos]
        dt = time.time() - t
        # desfazer a ordenação
        inv = np.empty_like(order_idx); inv[order_idx] = np.arange(len(order_idx))
        return out[inv], dt

    _ = score(m_fp32, sents[:64]); _ = score(m_q, sents[:64])
    p32, dt32 = score(m_fp32, sents)
    pq, dtq = score(m_q, sents)
    print(f"\nfp32: {len(sents)/dt32:6.1f} sent/s")
    print(f"int8: {len(sents)/dtq:6.1f} sent/s   (speedup {dt32/dtq:.2f}x)")
    # concordância
    dpos = np.abs(p32[:, 2] - pq[:, 2])
    corr = np.corrcoef(p32[:, 2], pq[:, 2])[0, 1]
    argmatch = (p32.argmax(1) == pq.argmax(1)).mean()
    print(f"\nconcordância fp32 vs int8:")
    print(f"  corr(p_pos): {corr:.4f} | mean|Δp_pos|: {dpos.mean():.4f} | max|Δp_pos|: {dpos.max():.4f}")
    print(f"  argmax igual: {100*argmatch:.1f}%")
    TOTAL = 2_384_498
    print(f"\nStage 2 estimado: fp32 ≈ {TOTAL/(len(sents)/dt32)/3600:.1f}h | int8 ≈ {TOTAL/(len(sents)/dtq)/3600:.1f}h")
    print("DONE.")

if __name__ == "__main__":
    main()
