"""Testes da segmentação e chunking (Fase 2) — medida de palavras, sem modelo."""

from tonediv.nlp.segment import chunk_text, split_sentences, word_measure


def test_split_sentences_basic():
    text = "Revenue grew. Margins expanded! Will it last? We think so."
    assert split_sentences(text) == [
        "Revenue grew.",
        "Margins expanded!",
        "Will it last?",
        "We think so.",
    ]


def test_split_sentences_no_terminal_punctuation():
    assert split_sentences("um texto sem pontuacao terminal") == ["um texto sem pontuacao terminal"]


def test_chunk_respects_limit_and_covers_all_text():
    # 6 sentenças de 4 palavras; limite de 10 palavras -> 2 sentenças por chunk.
    text = " ".join(["Aaa bbb ccc ddd." for _ in range(6)])
    chunks = chunk_text(text, max_tokens=10, measure=word_measure)
    assert len(chunks) == 3
    assert all(word_measure(c) <= 10 for c in chunks)
    # Cobertura: nenhuma palavra perdida.
    assert sum(word_measure(c) for c in chunks) == word_measure(text)


def test_single_short_text_is_one_chunk():
    assert chunk_text("Curto e direto.", max_tokens=512, measure=word_measure) == [
        "Curto e direto."
    ]


def test_oversized_sentence_hard_split():
    # Uma "sentença" de 25 palavras sem pontuação, limite 10 -> 3 pedaços.
    text = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_text(text, max_tokens=10, measure=word_measure)
    assert len(chunks) == 3
    assert all(word_measure(c) <= 10 for c in chunks)
    assert " ".join(chunks) == text  # sem perda nem duplicação


def test_real_tokenizer_contract_chunks_within_limit():
    """Contrato chunk<=512 com o tokenizer REAL (descarte derrubado 2x1 na revisão).

    Usa só o tokenizer (leve, já em cache local) — não carrega o modelo.
    Pulado se indisponível offline.
    """
    import pytest

    transformers = pytest.importorskip("transformers", reason="transformers ausente")
    from tonediv.config import load_config

    cfg = load_config()
    try:
        tok = transformers.AutoTokenizer.from_pretrained(cfg.finbert.model_name)
    except OSError as exc:
        pytest.skip(f"tokenizer indisponível offline: {exc}")

    def measure(t: str) -> int:
        return len(tok.encode(t, add_special_tokens=True))

    # Fala longa realista: 120 sentenças + uma "palavra" gigante no meio.
    text = " ".join(
        ["Revenue grew strongly and margins expanded across all product segments."] * 120
    )
    text += " " + "supercalifragilistic" * 60 + ". We remain confident in our outlook."
    chunks = chunk_text(text, cfg.finbert.max_tokens, measure)
    assert len(chunks) >= 2
    assert all(measure(c) <= cfg.finbert.max_tokens for c in chunks)


def test_giant_single_word_never_exceeds_limit():
    # Buraco reproduzido na 2ª revisão: "palavra" única gigante (URL colada)
    # gerava chunk acima do limite. Medida por caracteres torna o caso testável:
    # cada char = 1 token; palavra de 100 chars com limite 16 -> só bisseção resolve.
    def char_measure(s: str) -> int:
        return len(s)

    word = "x" * 100
    chunks = chunk_text(word, max_tokens=16, measure=char_measure)
    assert all(char_measure(c) <= 16 for c in chunks)
    assert "".join(c.replace(" ", "") for c in chunks) == word  # sem perda
