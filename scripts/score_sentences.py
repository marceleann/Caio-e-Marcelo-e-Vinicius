# -*- coding: utf-8 -*-
"""Stage 2 — scoring de tom em nível de SENTENÇA com FinBERT.

Por que existe (diagnóstico P1): pontuar a fala INTEIRA satura o FinBERT
(p_pos<0.01 em 53,5% das falas; só 16,5% na banda intermediária). O FinBERT foi
fine-tunado em SENTENÇAS de relatórios financeiros — então pontuamos sentença a
sentença e agregamos ao gestor por tokens (Stage 3). Isso também é a ponte
conceitual exata com o Loughran-McDonald do Angelo (fração de sentenças
positivas ~ fração de palavras positivas).

Saída: um shard parquet POR CALL em data/interim/sentence_scores_shards/, com
uma linha por sentença: [call_id, utterance_idx, sentence_idx, role, section,
speaker, n_tokens, p_neg, p_neu, p_pos]. Resumível (calls já pontuadas são
puladas). Ordem de processamento = embaralhamento determinístico (seed 42), para
que qualquer prefixo seja representativo (piloto) e o run completo reaproveite.

Uso:
  python scripts/score_sentences.py pilot 900   # primeiras 900 calls (embaralhadas)
  python scripts/score_sentences.py full         # todas as calls restantes
"""
from __future__ import annotations
import os, sys, time, logging
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # não ir à rede (modelo em cache)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "10")

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonediv.nlp.segment import split_sentences, chunk_text  # segmentador do repo

INT = ROOT / "data" / "interim"
OUTDIR = INT / "sentence_scores_shards"
MODEL = "yiyanghkust/finbert-tone"
BATCH = 32
MAXLEN = 512
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("stage2")


def resolve_order(id2label):
    keys = {"neg": "negative", "neu": "neutral", "pos": "positive"}
    order = {}
    for i, lab in id2label.items():
        norm = str(lab).casefold()
        for k, sub in keys.items():
            if sub in norm:
                order[k] = int(i)
    assert set(order) == {"neg", "neu", "pos"}, f"labels não resolvidos: {id2label}"
    return order["neg"], order["neu"], order["pos"]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    n_calls = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    OUTDIR.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(10)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    i_neg, i_neu, i_pos = resolve_order(dict(model.config.id2label))
    log.info("FinBERT pronto (CPU). order(neg,neu,pos)=%s", (i_neg, i_neu, i_pos))

    def measure(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=True))

    # falas elegíveis (mesmo filtro do scorer.filter_eligible)
    roles = pd.read_parquet(INT / "utterances_roles.parquet")
    elig = roles[
        roles["role"].isin(["management", "analyst"])
        & (roles["n_chars"] >= 20)
        & roles["text"].astype(str).str.strip().ne("")
    ].copy()

    # ordem determinística das calls
    all_calls = np.sort(elig["call_id"].unique())
    rng = np.random.default_rng(SEED)
    shuffled = all_calls.copy()
    rng.shuffle(shuffled)
    if mode == "pilot":
        target = shuffled[:n_calls]
    else:
        target = shuffled
    # pular já feitas
    todo = [c for c in target if not (OUTDIR / f"call_{c}.parquet").exists()]
    log.info("modo=%s | calls alvo=%d | já feitas=%d | a fazer=%d",
             mode, len(target), len(target) - len(todo), len(todo))

    elig_g = {cid: g for cid, g in elig.groupby("call_id", sort=False)}

    def score_batch(texts):
        out = np.empty((len(texts), 3), dtype=np.float64)
        ntok = np.empty(len(texts), dtype=np.int64)
        order_idx = np.argsort([len(x) for x in texts])
        st = [texts[i] for i in order_idx]
        res = np.empty((len(texts), 3)); rtok = np.empty(len(texts), dtype=np.int64)
        with torch.no_grad():
            for i in range(0, len(st), BATCH):
                b = st[i:i + BATCH]
                enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN)
                p = torch.softmax(model(**enc).logits, dim=-1).numpy()
                res[i:i + len(b), 0] = p[:, i_neg]
                res[i:i + len(b), 1] = p[:, i_neu]
                res[i:i + len(b), 2] = p[:, i_pos]
                rtok[i:i + len(b)] = enc["attention_mask"].sum(1).numpy()
        inv = np.empty_like(order_idx); inv[order_idx] = np.arange(len(order_idx))
        return res[inv], rtok[inv]

    t_start = time.time()
    done_sent = 0
    for k, cid in enumerate(todo):
        g = elig_g.get(cid)
        if g is None:
            continue
        rows_meta = []   # (utt_idx, sent_idx, role, section, speaker)
        texts = []
        for _, r in g.iterrows():
            sents = split_sentences(str(r["text"]))
            units = []
            for s in sents:
                # sentença normal (p95≈49 tokens) entra direta; só a rara sentença
                # gigante (>1200 chars ~ >300 tokens) paga um measure() e é hard-splitada
                # se de fato passar de 512 — evita tokenizar 2x o caso comum.
                if len(s) > 1200 and measure(s) > MAXLEN:
                    units.extend(chunk_text(s, MAXLEN, measure))
                else:
                    units.append(s)
            for si, u in enumerate(units):
                rows_meta.append((r["utterance_idx"], si, r["role"], r["section"], r["speaker"]))
                texts.append(u)
        if not texts:
            continue
        probs, ntok = score_batch(texts)
        df = pd.DataFrame(rows_meta, columns=["utterance_idx", "sentence_idx", "role", "section", "speaker"])
        df.insert(0, "call_id", cid)
        df["n_tokens"] = ntok
        df["p_neg"], df["p_neu"], df["p_pos"] = probs[:, 0], probs[:, 1], probs[:, 2]
        df.to_parquet(OUTDIR / f"call_{cid}.parquet", index=False)  # commit atômico da call
        done_sent += len(texts)
        if (k + 1) % 20 == 0 or k == len(todo) - 1:
            el = time.time() - t_start
            rate = done_sent / el if el > 0 else 0
            eta = (len(todo) - (k + 1)) * (el / (k + 1)) / 3600 if k + 1 else 0
            log.info("call %d/%d | %d sent | %.1f sent/s | ETA restante ~%.1f h",
                     k + 1, len(todo), done_sent, rate, eta)
    log.info("FIM modo=%s: %d calls novas, %d sentenças, %.1f min",
             mode, len(todo), done_sent, (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
