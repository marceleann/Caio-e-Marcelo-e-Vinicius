"""Segmentação de texto: sentenças e chunking limitado a N tokens.

Por que existe:
    O encoder do FinBERT aceita no máximo 512 tokens; falas longas (comuns em
    prepared remarks) precisam ser divididas SEM cortar sentenças no meio —
    cortes arbitrários mutilam o contexto sintático de que o modelo depende
    para julgar o tom. Este módulo divide em sentenças (regex leve) e empacota
    sentenças consecutivas em chunks que respeitam o limite.

    A contagem de tokens é INJETADA (``measure``): o scorer passa a contagem
    real do tokenizer do FinBERT; os testes passam contagem de palavras. Assim
    a lógica de empacotamento é 100% testável offline, sem baixar modelo
    (separação biblioteca pura × dependência pesada, regra nº 10).

Limitação declarada: o split por regex erra em abreviações raras ("Inc.",
"U.S.") — aceitável, pois o custo de um split extra é só um chunk a mais, nunca
perda de texto. Documentado aqui para a defesa.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Fim de sentença: pontuação terminal seguida de espaço + letra maiúscula/dígito.
# Exigir a maiúscula seguinte reduz falsos splits em abreviações ("vs. the").
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

# Medida de tokens: recebe um texto, devolve o nº de tokens no vocabulário alvo.
TokenMeasure = Callable[[str], int]


def word_measure(text: str) -> int:
    """Medida de fallback baseada em palavras (para testes e estimativas).

    Não substitui o tokenizer real no scoring — subconta tokens de subpalavra —
    mas preserva a propriedade que os testes exercitam: monotonicidade e
    aditividade aproximada por concatenação.
    """
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    """Divide um texto em sentenças por regex leve.

    Args:
        text: Texto de uma fala.

    Returns:
        Lista de sentenças não vazias, na ordem original. Texto sem pontuação
        terminal retorna lista de um elemento (a fala inteira).
    """
    parts = [p.strip() for p in _SENTENCE_END.split(text)]
    return [p for p in parts if p]


def _split_giant_word(word: str, max_tokens: int, measure: TokenMeasure) -> list[str]:
    """Bisseção por caracteres de UMA palavra que sozinha excede o limite.

    Caso extremo real: uma "palavra" sem espaços de milhares de caracteres
    (URL, tabela colada). Sem esta divisão, ``_hard_split`` emitiria um pedaço
    acima do limite — falsificando a garantia "chunk <= max_tokens" (buraco
    reproduzido na 2ª rodada da revisão multi-agente). A truncation do encode
    continuaria como cinto, mas descartaria texto silenciosamente.
    """
    if measure(word) <= max_tokens or len(word) <= 1:
        return [word]
    mid = len(word) // 2
    return _split_giant_word(word[:mid], max_tokens, measure) + _split_giant_word(
        word[mid:], max_tokens, measure
    )


def _hard_split(sentence: str, max_tokens: int, measure: TokenMeasure) -> list[tuple[str, int]]:
    """Divide UMA sentença que sozinha excede o limite (caso raro).

    Corta por palavras, acumulando MEDIDAS por palavra (uma chamada a
    ``measure`` por palavra, nunca sobre a concatenação crescente). Palavra
    que SOZINHA excede o limite passa por :func:`_split_giant_word`. Perde-se
    coesão sintática só nesses casos extremos — preferível a truncar
    silenciosamente, que descartaria texto do sinal.

    Returns:
        Lista de pares ``(pedaço, medida_aproximada)``.
    """
    pieces: list[tuple[str, int]] = []
    current: list[str] = []
    current_len = 0
    words = [w for raw in sentence.split() for w in _split_giant_word(raw, max_tokens, measure)]
    for word in words:
        w_len = measure(word)
        if current and current_len + w_len > max_tokens:
            pieces.append((" ".join(current), current_len))
            current, current_len = [word], w_len
        else:
            current.append(word)
            current_len += w_len
    if current:
        pieces.append((" ".join(current), current_len))
    return pieces


def chunk_text(text: str, max_tokens: int, measure: TokenMeasure) -> list[str]:
    """Empacota o texto em chunks de até ``max_tokens``, sem cortar sentenças.

    Estratégia gulosa com SOMA de medidas por sentença: ``measure`` é chamada
    UMA vez por sentença (O(n) chamadas ao tokenizer), não sobre a concatenação
    crescente (que seria O(n²) — custo real, pois este é o gargalo da etapa
    mais cara do pipeline; achado da revisão multi-agente). A soma superestima
    levemente a medida real do chunk quando ``measure`` inclui tokens especiais
    ([CLS]/[SEP] contados por sentença) — erro CONSERVADOR: chunks saem menores
    que o limite, nunca maiores; a truncation no encode é o cinto de segurança.

    Sem sobreposição entre chunks (decisão fixa, ADR-015): simples e sem dupla
    contagem de texto na agregação ponderada.

    Args:
        text: Texto completo da fala.
        max_tokens: Limite de tokens por chunk (512 para o FinBERT, via config).
        measure: Função texto -> nº de tokens (tokenizer real no scoring).

    Returns:
        Lista de chunks cobrindo todo o texto; vazia para texto vazio/só
        espaços (o scorer filtra esses casos ANTES de chegar aqui).
    """
    measured: list[tuple[str, int]] = []
    for sentence in split_sentences(text):
        s_len = measure(sentence)
        if s_len > max_tokens:
            measured.extend(_hard_split(sentence, max_tokens, measure))
        else:
            measured.append((sentence, s_len))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence, s_len in measured:
        if current and current_len + s_len > max_tokens:
            chunks.append(" ".join(current))
            current, current_len = [sentence], s_len
        else:
            current.append(sentence)
            current_len += s_len
    if current:
        chunks.append(" ".join(current))
    return chunks
