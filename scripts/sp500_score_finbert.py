# -*- coding: utf-8 -*-
"""FinBERT sentença-a-sentença para TODO o S&P 500 (braço FinBERT do projeto).

Decisão de desenho (Marcelo): replicação exata do Angelo com UM desvio
intencional — o tom medido por FinBERT em nível de sentença (nosso diferencial
GenAI), no lugar do dicionário LM. Este script pontua as ~27,7k calls que
faltam (5.452 do universo tech já estão em data/interim/sentence_scores_shards
e são reaproveitadas — mesmo segmentador, mesmo modelo, mesma convenção).

Mecânica:
  - calls do HF em ordem EMBARALHADA determinística (seed 42): qualquer
    prefixo é amostra representativa; o run é retomável (1 parquet por call,
    commit atômico; calls existentes são puladas);
  - papéis por fala vêm de speaker_counts.parquet (cache LLM + determinístico
    validado — os mesmos do braço LM, garantindo comparabilidade);
  - elegíveis: manager/analyst com >=20 chars; sentenças via split_sentences
    do repo; FinBERT-tone (yiyanghkust) em CPU, batch por comprimento.
Saída: data/interim/sentence_scores_shards/call_<id>.parquet
       [call_id, utterance_idx, sentence_idx, role, section, mk, n_tokens,
        p_neg, p_neu, p_pos]
Uso: python scripts/sp500_score_finbert.py [max_horas]   (default 12h por sessão)
"""
from __future__ import annotations
import os, re, sys, time, logging, unicodedata
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "10")

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonediv.config import load_config
from tonediv.data import transcripts as tr
from tonediv.nlp.segment import split_sentences, chunk_text

INT = ROOT / "data" / "interim"
OUTDIR = INT / "sentence_scores_shards"
MODEL = "yiyanghkust/finbert-tone"
BATCH, MAXLEN, SEED = 32, 512, 42
MAX_HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("sp500-finbert")

_QAp = re.compile(r"^[QA]\s*[-–:]\s*"); _TS = re.compile(r"\s+[-–—]\s+.*$"); _CM = re.compile(r"(?<=[a-z])(?=[A-Z])")
def _clean(n):
    if not isinstance(n, str): return ""
    n = _QAp.sub("", n.strip()); n = _TS.sub("", n); n = _CM.sub(" ", n)
    n = unicodedata.normalize("NFKD", n); n = "".join(c for c in n if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", n).strip().casefold()
def mkey(n):
    c = _clean(n)
    if not c: return ""
    t = [x for x in c.replace(".", " ").split() if len(x) > 1]
    return f"{t[-1]}|{t[0][0]}" if t else c


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
    t00 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "config.yaml"))

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(10)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    i_neg, i_neu, i_pos = resolve_order(dict(model.config.id2label))
    log.info("FinBERT pronto (CPU).")

    roles = pd.read_parquet(INT / "sp500" / "speaker_counts.parquet",
                            columns=["call_id", "utterance_idx", "mk", "role_eff", "section"])
    roles = roles[roles["role_eff"].isin(["manager", "analyst"])]
    rkey = {}
    for r in roles.itertuples():
        rkey[(r.call_id, r.utterance_idx)] = (r.mk, r.role_eff, r.section)
    log.info("papéis: %d falas elegíveis", len(rkey))

    calls = tr.prepare_calls(tr.load_hf_transcripts(cfg), cfg)
    done = {p.stem[5:] for p in OUTDIR.glob("call_*.parquet")}
    order = np.argsort(calls["call_id"].to_numpy())
    rng = np.random.default_rng(SEED)
    rng.shuffle(order)
    calls = calls.iloc[order].reset_index(drop=True)
    todo_mask = ~calls["call_id"].isin(done)
    calls = calls[todo_mask].reset_index(drop=True)
    log.info("calls: %d já pontuadas | %d a fazer", len(done), len(calls))

    def measure(text): return len(tok.encode(text, add_special_tokens=True))

    def score_batch(texts):
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

    n_done = n_sent = 0
    t0 = time.time()
    for i0 in range(0, len(calls), 200):     # sub-blocos p/ não estourar memória
        chunk = calls.iloc[i0:i0 + 200]
        utts = tr.build_utterances(chunk)
        for cid, g in utts.groupby("call_id", sort=False):
            if (time.time() - t0) > MAX_HOURS * 3600:
                log.info("limite de %.1fh atingido — %d calls nesta sessão", MAX_HOURS, n_done)
                return
            rows_meta, texts = [], []
            for r in g.itertuples():
                key = rkey.get((cid, r.utterance_idx))
                if key is None:
                    continue
                mk_, role_, sec_ = key
                txt = str(r.text)
                if len(txt.strip()) < 20:
                    continue
                for s in split_sentences(txt):
                    units = chunk_text(s, MAXLEN, measure) \
                        if (len(s) > 1200 and measure(s) > MAXLEN) else [s]
                    for u in units:
                        rows_meta.append((r.utterance_idx, len(rows_meta), role_, sec_, mk_))
                        texts.append(u)
            if not texts:
                continue
            probs, ntok = score_batch(texts)
            df = pd.DataFrame(rows_meta, columns=["utterance_idx", "sentence_idx",
                                                  "role", "section", "mk"])
            df.insert(0, "call_id", cid)
            df["n_tokens"] = ntok
            df["p_neg"], df["p_neu"], df["p_pos"] = probs[:, 0], probs[:, 1], probs[:, 2]
            df.to_parquet(OUTDIR / f"call_{cid}.parquet", index=False)
            n_done += 1; n_sent += len(texts)
            if n_done % 20 == 0:
                el = time.time() - t0
                eta_h = (len(calls) - n_done) * (el / n_done) / 3600
                log.info("call %d/%d | %.1f sent/s | ETA total ~%.1f h",
                         n_done, len(calls), n_sent / el, eta_h)
    log.info("FIM: %d calls, %d sentenças, %.1f min", n_done, n_sent, (time.time() - t00) / 60)


if __name__ == "__main__":
    main()
