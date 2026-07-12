"""Scoring de tom com FinBERT: distribuições [neg, neu, pos] por fala.

Por que existe:
    É o sensor do projeto — converte texto em distribuições de sentimento que
    alimentam todas as features. Duas decisões críticas moram aqui:

    1. **Mapeamento de labels em RUNTIME (ADR-007 / armadilha nº 3):** a ordem
       das classes de um modelo HuggingFace NÃO é garantida; hardcodar índices
       pode inverter positivo↔negativo silenciosamente — o pior tipo de bug,
       pois o pipeline roda "com sucesso" e produz sinal com o sinal trocado.
       :func:`resolve_label_order` lê ``model.config.id2label`` e mapeia por
       NOME, falhando alto se as três classes não forem identificáveis. É pura
       e testada offline com dicionários falsos + frases-sentinela quando o
       modelo está disponível (tests/test_scorer_labels.py).

    2. **Chunking com o tokenizer real:** falas acima de 512 tokens são
       divididas por :mod:`tonediv.nlp.segment` usando a contagem do próprio
       tokenizer do FinBERT (não aproximação por palavras) e agregadas por
       média ponderada pelo nº de tokens de cada chunk — chunks maiores
       carregam mais texto, logo mais peso.

    Ordem CANÔNICA de saída em todo o projeto: ``[p_neg, p_neu, p_pos]``.

Dependências pesadas (torch/transformers) importadas tardiamente, só no I/O
real — a lógica pura permanece testável sem elas (regra nº 10).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tonediv.config import Config
from tonediv.nlp.segment import chunk_text

logger = logging.getLogger(__name__)

# Substrings que identificam cada classe no id2label (casefold).
_LABEL_KEYS: dict[str, str] = {"neg": "negative", "neu": "neutral", "pos": "positive"}


@dataclass(frozen=True)
class ScorerBundle:
    """Modelo carregado + metadados necessários para pontuar.

    Attributes:
        tokenizer: Tokenizer do FinBERT.
        model: Modelo de classificação (em modo eval, no device escolhido).
        label_order: Índices (i_neg, i_neu, i_pos) nos logits do modelo.
        device: Device efetivo ("cuda" ou "cpu").
    """

    tokenizer: Any
    model: Any
    label_order: tuple[int, int, int]
    device: str


def resolve_label_order(id2label: dict[int, str]) -> tuple[int, int, int]:
    """Resolve os índices (neg, neu, pos) a partir do ``id2label`` do modelo.

    Mapeia por NOME (substring casefold: "negative"/"neutral"/"positive"),
    nunca por posição. Função PURA — o teste offline cobre ordens embaralhadas,
    classes faltantes e duplicadas sem baixar o modelo.

    Args:
        id2label: Dicionário índice->rótulo de ``model.config.id2label``.

    Returns:
        Tupla ``(i_neg, i_neu, i_pos)``.

    Raises:
        ValueError: Se alguma das três classes não for encontrada ou se um
            rótulo casar com mais de uma classe (mapeamento ambíguo).
    """
    found: dict[str, int] = {}
    for idx, label in id2label.items():
        norm = str(label).casefold()
        matches = [k for k, sub in _LABEL_KEYS.items() if sub in norm]
        if len(matches) > 1:
            raise ValueError(f"Rótulo ambíguo no id2label: {label!r} casa com {matches}.")
        if matches:
            key = matches[0]
            if key in found:
                raise ValueError(f"Classe '{key}' duplicada no id2label: {id2label}.")
            found[key] = int(idx)
    missing = [k for k in ("neg", "neu", "pos") if k not in found]
    if missing:
        raise ValueError(f"Classes ausentes no id2label {id2label}: {missing}.")
    return found["neg"], found["neu"], found["pos"]


def _resolve_device(requested: str) -> str:
    """Resolve o device: "auto" vira cuda se disponível, senão cpu."""
    import torch  # import tardio: dependência pesada

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_finbert(cfg: Config) -> ScorerBundle:
    """Carrega o FinBERT e valida o mapeamento de labels em runtime.

    Returns:
        :class:`ScorerBundle` pronto para :func:`score_texts`.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = cfg.finbert.model_name
    logger.info("Carregando FinBERT (%s)...", name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    order = resolve_label_order(dict(model.config.id2label))
    device = _resolve_device(cfg.finbert.device)
    model.to(device)
    model.eval()
    logger.info("FinBERT pronto: device=%s, label_order(neg,neu,pos)=%s", device, order)
    return ScorerBundle(tokenizer=tokenizer, model=model, label_order=order, device=device)


def score_texts(
    bundle: ScorerBundle, texts: list[str], batch_size: int, max_tokens: int
) -> np.ndarray:
    """Pontua uma lista de textos, retornando probabilidades [neg, neu, pos].

    Args:
        bundle: Saída de :func:`load_finbert`.
        texts: Textos JÁ dentro do limite de tokens (chunks). ``truncation``
            fica ligado mesmo assim como cinto de segurança.
        batch_size: Tamanho do lote (config; ajustar à memória disponível).
        max_tokens: Limite do encoder (config).

    Returns:
        Array ``(n, 3)`` na ordem canônica ``[p_neg, p_neu, p_pos]``.
    """
    import torch

    i_neg, i_neu, i_pos = bundle.label_order
    n_batches = max(1, (len(texts) + batch_size - 1) // batch_size)
    out = np.empty((len(texts), 3), dtype=np.float64)
    with torch.no_grad():
        for b, start in enumerate(range(0, len(texts), batch_size)):
            batch = texts[start : start + batch_size]
            enc = bundle.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens
            ).to(bundle.device)
            probs = torch.softmax(bundle.model(**enc).logits, dim=-1).cpu().numpy()
            # Reordena da ordem do MODELO para a ordem CANÔNICA do projeto.
            out[start : start + len(batch), 0] = probs[:, i_neg]
            out[start : start + len(batch), 1] = probs[:, i_neu]
            out[start : start + len(batch), 2] = probs[:, i_pos]
            # Progresso a cada ~200 lotes: a etapa leva HORAS em CPU e o
            # operador precisa distinguir "andando" de "travado".
            if b % 200 == 0:
                logger.info(
                    "Scoring: lote %d/%d (%.1f%%)", b + 1, n_batches, 100 * (b + 1) / n_batches
                )
    return out


def aggregate_chunk_probs(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Agrega distribuições de chunks numa distribuição por fala (média ponderada).

    Pesos = nº de tokens de cada chunk: um chunk com 400 tokens representa mais
    texto (e mais informação de tom) que um de 40. Função pura, testada com
    números feitos à mão (tests/test_scorer_labels.py).

    Args:
        probs: Array ``(k, 3)`` dos chunks de UMA fala.
        weights: Array ``(k,)`` de pesos positivos.

    Returns:
        Vetor ``(3,)`` normalizado (soma 1 a menos de erro numérico).
    """
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Pesos de agregação devem somar > 0.")
    return (probs * weights[:, None]).sum(axis=0) / total


def filter_eligible(utterances: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Seleciona as falas que entram no scoring (regra de elegibilidade).

    Elegibilidade: ``role`` em {management, analyst}, comprimento ≥
    ``quality.min_chars_utterance`` e texto não-vazio. Falas curtas ("Thank
    you.") são ruído de cortesia; texto whitespace-only produziria 0 chunks e
    derrubaria a agregação DEPOIS de horas de scoring (revisão multi-agente).
    """
    return utterances[
        utterances["role"].isin(["management", "analyst"])
        & (utterances["n_chars"] >= cfg.quality.min_chars_utterance)
        & utterances["text"].astype(str).str.strip().ne("")
    ].reset_index(drop=True)


def _score_eligible(eligible: pd.DataFrame, cfg: Config, bundle: ScorerBundle) -> pd.DataFrame:
    """Pontua um conjunto JÁ filtrado de falas elegíveis (chunk -> score -> agrega).

    Puro em relação ao I/O; o índice de ``eligible`` deve estar resetado (spans
    são posicionais).
    """
    fb = cfg.finbert
    if eligible.empty:
        return _empty_scores()

    def measure(text: str) -> int:
        return len(bundle.tokenizer.encode(text, add_special_tokens=True))

    all_chunks: list[str] = []
    spans: list[tuple[int, int]] = []  # [início, fim) dos chunks de cada fala
    for text in eligible["text"]:
        chunks = chunk_text(str(text), fb.max_tokens, measure)
        spans.append((len(all_chunks), len(all_chunks) + len(chunks)))
        all_chunks.extend(chunks)

    chunk_tokens = np.array([measure(c) for c in all_chunks], dtype=np.float64)
    # ORDENAÇÃO POR COMPRIMENTO antes do batching: cada lote é padded até o seu
    # texto mais longo — com lotes de tamanhos mistos, textos de 30 tokens
    # pagam o custo de 500. Agrupar comprimentos parecidos reduz o compute em
    # CPU tipicamente 2–4x nesta base (mediana curta, cauda longa). A ordem
    # original é RESTAURADA por índice invertido; teste com modelo fake
    # sensível ao comprimento prova a restauração (test_scorer_labels.py).
    order = np.argsort(chunk_tokens, kind="stable")
    sorted_chunks = [all_chunks[i] for i in order]
    sorted_probs = score_texts(bundle, sorted_chunks, fb.batch_size, fb.max_tokens)
    chunk_probs = np.empty_like(sorted_probs)
    chunk_probs[order] = sorted_probs

    rows = np.empty((len(eligible), 3), dtype=np.float64)
    n_tokens = np.empty(len(eligible), dtype=np.int64)
    n_chunks = np.empty(len(eligible), dtype=np.int64)
    for i, (start, end) in enumerate(spans):
        rows[i] = aggregate_chunk_probs(chunk_probs[start:end], chunk_tokens[start:end])
        n_tokens[i] = int(chunk_tokens[start:end].sum())
        n_chunks[i] = end - start

    out = eligible[["call_id", "utterance_idx", "role", "section"]].copy()
    out["p_neg"], out["p_neu"], out["p_pos"] = rows[:, 0], rows[:, 1], rows[:, 2]
    out["net_tone"] = out["p_pos"] - out["p_neg"]
    out["n_tokens"], out["n_chunks"] = n_tokens, n_chunks
    return out


_SCORE_COLS = [
    "call_id",
    "utterance_idx",
    "role",
    "section",
    "p_neg",
    "p_neu",
    "p_pos",
    "net_tone",
    "n_tokens",
    "n_chunks",
]


def _empty_scores() -> pd.DataFrame:
    """DataFrame vazio com o schema de saída do scoring."""
    return pd.DataFrame(columns=_SCORE_COLS)


def score_utterances(utterances: pd.DataFrame, cfg: Config, bundle: ScorerBundle) -> pd.DataFrame:
    """Pontua o tom de cada fala elegível (in-memory, sem checkpoint).

    Args:
        utterances: Tabela anotada por :func:`tonediv.nlp.roles.infer_roles`.
        cfg: Configuração.
        bundle: FinBERT carregado.

    Returns:
        DataFrame ``[call_id, utterance_idx, role, section, p_neg, p_neu,
        p_pos, net_tone, n_tokens, n_chunks]`` — uma linha por fala pontuada.
    """
    eligible = filter_eligible(utterances, cfg)
    logger.info("Scoring: %d falas elegíveis de %d.", len(eligible), len(utterances))
    return _score_eligible(eligible, cfg, bundle)


def score_utterances_checkpointed(
    utterances: pd.DataFrame, cfg: Config, bundle: ScorerBundle, checkpoint_dir: Path
) -> pd.DataFrame:
    """Pontua em FATIAS com checkpoint em disco — um restart retoma de onde parou.

    O scoring de ~300k falas leva horas em CPU e a máquina pode dormir/reiniciar
    no meio (aconteceu 3x). Cada fatia de ``finbert.checkpoint_shard_utterances``
    falas é salva em ``checkpoint_dir/shard_XXXXX.parquet`` assim que concluída;
    numa nova execução, fatias já existentes são LIDAS do disco em vez de
    reprocessadas. Como ``eligible`` é derivado deterministicamente da mesma
    tabela, as fronteiras das fatias são estáveis entre execuções.

    Args:
        utterances: Tabela anotada de falas.
        cfg: Configuração (tamanho da fatia em ``finbert``).
        bundle: FinBERT carregado.
        checkpoint_dir: Diretório dos parquets de fatia (criado se não existir).

    Returns:
        A tabela de scores completa (concatenação de todas as fatias).
    """
    eligible = filter_eligible(utterances, cfg)
    shard_size = cfg.finbert.checkpoint_shard_utterances
    n_shards = (len(eligible) + shard_size - 1) // shard_size
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Scoring: %d falas elegíveis de %d em %d fatias de %d (checkpoint em %s).",
        len(eligible),
        len(utterances),
        n_shards,
        shard_size,
        checkpoint_dir,
    )

    parts: list[pd.DataFrame] = []
    for i in range(n_shards):
        path = checkpoint_dir / f"shard_{i:05d}.parquet"
        if path.exists():
            logger.info("Fatia %d/%d já pontuada — retomando do disco.", i + 1, n_shards)
            parts.append(pd.read_parquet(path))
            continue
        shard = eligible.iloc[i * shard_size : (i + 1) * shard_size].reset_index(drop=True)
        scored = _score_eligible(shard, cfg, bundle)
        scored.to_parquet(path, index=False)  # commit atômico da fatia
        logger.info("Fatia %d/%d pontuada (%d falas) e salva.", i + 1, n_shards, len(scored))
        parts.append(scored)

    return pd.concat(parts, ignore_index=True) if parts else _empty_scores()
