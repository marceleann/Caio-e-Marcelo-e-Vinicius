"""Testes do scorer (Fase 2): mapeamento de labels e agregação (armadilha nº 3).

A parte OFFLINE (sempre roda) cobre a lógica pura: resolve_label_order com
ordens embaralhadas/faltantes e a agregação ponderada com números à mão.
A parte com MODELO (frases-sentinela) roda só quando torch/transformers estão
instalados e o modelo está acessível — senão é pulada com motivo explícito.
"""

import numpy as np
import pytest

from tonediv.config import load_config
from tonediv.nlp.scorer import aggregate_chunk_probs, resolve_label_order

CFG = load_config()


# ---------------------------------------------------------------------------
# Offline: resolve_label_order (o guarda real contra a inversão silenciosa)
# ---------------------------------------------------------------------------
def test_label_order_standard_finbert_layout():
    # Layout real do yiyanghkust/finbert-tone: 0=Neutral, 1=Positive, 2=Negative.
    order = resolve_label_order({0: "Neutral", 1: "Positive", 2: "Negative"})
    assert order == (2, 0, 1)  # (i_neg, i_neu, i_pos)


def test_label_order_any_permutation():
    order = resolve_label_order({0: "negative", 1: "neutral", 2: "positive"})
    assert order == (0, 1, 2)
    order = resolve_label_order({0: "POSITIVE", 1: "NEGATIVE", 2: "NEUTRAL"})
    assert order == (1, 2, 0)


def test_label_order_missing_class_raises():
    with pytest.raises(ValueError, match="ausentes"):
        resolve_label_order({0: "positive", 1: "negative"})


def test_label_order_duplicate_class_raises():
    with pytest.raises(ValueError, match="duplicada"):
        resolve_label_order({0: "positive", 1: "very positive", 2: "negative", 3: "neutral"})


def test_label_order_unrelated_labels_raises():
    with pytest.raises(ValueError):
        resolve_label_order({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


# ---------------------------------------------------------------------------
# Offline: agregação ponderada (números verificáveis no papel)
# ---------------------------------------------------------------------------
def test_aggregate_weighted_mean_by_hand():
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.2, 0.6]])
    weights = np.array([3.0, 1.0])
    # À mão: neg = (0.8*3 + 0.2*1)/4 = 0.65; neu = 0.125; pos = 0.225.
    agg = aggregate_chunk_probs(probs, weights)
    np.testing.assert_allclose(agg, [0.65, 0.125, 0.225])
    assert abs(agg.sum() - 1.0) < 1e-12


def test_aggregate_zero_weights_raises():
    with pytest.raises(ValueError):
        aggregate_chunk_probs(np.array([[0.3, 0.3, 0.4]]), np.array([0.0]))


# ---------------------------------------------------------------------------
# Reordenação canônica com modelo FAKE (exige torch, mas NÃO baixa nada):
# prova que score_texts converte o layout do modelo p/ [neg, neu, pos].
# ---------------------------------------------------------------------------
def test_score_texts_reorders_model_layout_to_canonical():
    torch = pytest.importorskip("torch", reason="torch não instalado neste ambiente")
    from tonediv.nlp.scorer import ScorerBundle, score_texts

    class _FakeEncoding(dict):
        def to(self, device):  # noqa: ANN001, ANN202 - duck-typing do BatchEncoding
            return self

    class _FakeTokenizer:
        def __call__(self, batch, **kwargs):  # noqa: ANN003, ANN001, ANN204
            return _FakeEncoding(input_ids=torch.zeros((len(batch), 4), dtype=torch.long))

    class _FakeOutput:
        # Layout do FinBERT real: 0=Neutral, 1=Positive, 2=Negative.
        # Logits DISTINTOS por classe: neu=1 < pos=2 < neg=3. Massa distinta em
        # cada saída mata mutantes que um logit único deixava passar (ex.:
        # probs[:, ::-1], que acerta neg mas troca neu<->pos — pego na 2ª
        # rodada da revisão multi-agente).
        logits = torch.tensor([[1.0, 2.0, 3.0]])

    class _FakeModel:
        def __call__(self, **enc):  # noqa: ANN003, ANN204
            return _FakeOutput()

    bundle = ScorerBundle(
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
        label_order=resolve_label_order({0: "Neutral", 1: "Positive", 2: "Negative"}),
        device="cpu",
    )
    probs = score_texts(bundle, ["dummy"], batch_size=8, max_tokens=512)
    # softmax([1,2,3]) = [0.0900, 0.2447, 0.6652] nos índices do MODELO
    # (neu, pos, neg) -> canônico [neg, neu, pos] = [0.6652, 0.0900, 0.2447].
    import numpy as np

    np.testing.assert_allclose(probs[0], [0.66524, 0.09003, 0.24473], atol=1e-4)


# ---------------------------------------------------------------------------
# Ordenação por comprimento no score_utterances: a ordem original é restaurada
# (modelo fake sensível ao COMPRIMENTO — permutação errada muda o resultado)
# ---------------------------------------------------------------------------
def test_score_utterances_restores_original_order_after_length_sort():
    torch = pytest.importorskip("torch", reason="torch não instalado neste ambiente")
    import pandas as pd

    from tonediv.config import load_config
    from tonediv.nlp.scorer import ScorerBundle, score_utterances

    class _Enc(dict):
        def to(self, device):  # noqa: ANN001, ANN202
            return self

    class _LenTokenizer:
        """Tokeniza por palavras; input_ids carregam o COMPRIMENTO do texto."""

        def encode(self, text, add_special_tokens=True):  # noqa: ANN001, ANN201, ARG002
            return list(range(len(text.split()) + 2))

        def __call__(self, batch, **kwargs):  # noqa: ANN003, ANN001, ANN204
            lens = torch.tensor([[float(len(t.split()))] for t in batch])
            return _Enc(input_ids=lens)

    class _LenModel:
        """Logit de NEGATIVE proporcional ao comprimento -> p_neg cresce com o texto."""

        def __call__(self, **enc):  # noqa: ANN003, ANN204
            # Escala 0.1 evita saturação do softmax (logit 30 e 20 dariam
            # ambos p~1.0 e o teste perderia o poder de distinguir).
            lens = enc["input_ids"][:, 0] * 0.1
            zeros = torch.zeros((len(lens), 2))
            out = type("O", (), {})()
            out.logits = torch.cat([zeros, lens.unsqueeze(1)], dim=1)  # [neu, pos, NEG]
            return out

    cfg = load_config()
    utts = pd.DataFrame(
        {
            "call_id": ["C1"] * 3,
            "utterance_idx": [0, 1, 2],
            "role": ["management", "analyst", "management"],
            "section": ["remarks", "qa", "qa"],
            # Comprimentos DECRESCENTES: a ordenação interna inverte; se a
            # restauração falhar, os p_neg saem trocados entre as falas.
            "text": [
                "alpha " * 30,  # 30 palavras -> maior p_neg
                "beta " * 20,
                "gamma " * 10,  # 10 palavras -> menor p_neg
            ],
            "n_chars": [180, 100, 60],
        }
    )
    bundle = ScorerBundle(
        tokenizer=_LenTokenizer(),
        model=_LenModel(),
        label_order=resolve_label_order({0: "Neutral", 1: "Positive", 2: "Negative"}),
        device="cpu",
    )
    out = score_utterances(utts, cfg, bundle).set_index("utterance_idx")
    assert out.loc[0, "p_neg"] > out.loc[1, "p_neg"] > out.loc[2, "p_neg"]


# ---------------------------------------------------------------------------
# Com modelo (pulado se indisponível): frases-sentinela do config
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_sentinel_phrases_with_real_model():
    pytest.importorskip("torch", reason="torch não instalado neste ambiente")
    pytest.importorskip("transformers", reason="transformers não instalado")
    from tonediv.nlp.scorer import load_finbert, score_texts

    try:
        bundle = load_finbert(CFG)
    except OSError as exc:  # sem rede/cache do modelo
        pytest.skip(f"FinBERT indisponível offline: {exc}")

    probs = score_texts(
        bundle,
        [CFG.finbert.sentinel_positive, CFG.finbert.sentinel_negative],
        batch_size=2,
        max_tokens=CFG.finbert.max_tokens,
    )
    p_pos_sentinela_positiva = probs[0, 2]
    p_neg_sentinela_positiva = probs[0, 0]
    p_pos_sentinela_negativa = probs[1, 2]
    p_neg_sentinela_negativa = probs[1, 0]
    # "Revenue grew strongly..." deve dar pos > neg; a negativa, o inverso.
    assert p_pos_sentinela_positiva > p_neg_sentinela_positiva
    assert p_neg_sentinela_negativa > p_pos_sentinela_negativa


# ---------------------------------------------------------------------------
# Checkpointing do scoring (resume): a 2ª execução lê as fatias do disco
# ---------------------------------------------------------------------------
def test_checkpointed_scoring_resumes_without_rescoring(tmp_path):
    torch = pytest.importorskip("torch", reason="torch não instalado")
    import dataclasses

    import pandas as pd

    from tonediv.nlp.scorer import ScorerBundle, score_utterances_checkpointed

    class _Enc(dict):
        def to(self, device):  # noqa: ANN001, ANN202
            return self

    class _Tok:
        def encode(self, text, add_special_tokens=True):  # noqa: ANN001, ANN201, ARG002
            return list(range(len(text.split()) + 2))

        def __call__(self, batch, **kw):  # noqa: ANN003, ANN001, ANN204
            return _Enc(input_ids=torch.zeros((len(batch), 4), dtype=torch.long))

    class _Model:
        def __call__(self, **enc):  # noqa: ANN003, ANN204
            n = enc["input_ids"].shape[0]
            out = type("O", (), {})()
            out.logits = torch.ones((n, 3))
            return out

    class _Raise:  # invocá-lo é falha: prova que o resume não re-pontuou
        def __call__(self, *a, **k):  # noqa: ANN002, ANN003, ANN204
            raise AssertionError("modelo/tokenizer NÃO deveria ser chamado no resume")

        encode = __call__

    order = resolve_label_order({0: "Neutral", 1: "Positive", 2: "Negative"})
    utts = pd.DataFrame(
        {
            "call_id": [f"C{i // 2}" for i in range(7)],
            "utterance_idx": list(range(7)),
            "role": ["management", "analyst"] * 3 + ["management"],
            "section": ["qa"] * 7,
            "text": [f"palavra {i} tom neutro do teste" for i in range(7)],
            "n_chars": [50] * 7,
        }
    )
    cfg = dataclasses.replace(
        CFG, finbert=dataclasses.replace(CFG.finbert, checkpoint_shard_utterances=3)
    )
    ckpt = tmp_path / "shards"

    bundle = ScorerBundle(tokenizer=_Tok(), model=_Model(), label_order=order, device="cpu")
    out1 = score_utterances_checkpointed(utts, cfg, bundle, ckpt)
    assert len(out1) == 7
    assert len(list(ckpt.glob("shard_*.parquet"))) == 3  # ceil(7/3)

    # 2ª execução com modelo que EXPLODE se chamado: tudo vem do disco.
    bundle_raise = ScorerBundle(tokenizer=_Raise(), model=_Raise(), label_order=order, device="cpu")
    out2 = score_utterances_checkpointed(utts, cfg, bundle_raise, ckpt)
    pd.testing.assert_frame_equal(out1, out2)
