"""Testes das transformações puras de transcrições (Fase 1)."""

import json

import pandas as pd

from tonediv.config import load_config
from tonediv.data.transcripts import (
    _dedupe_ids,
    _extract_utterance,
    _parse_structured_content,
    build_utterances,
    calls_metadata,
    filter_quality,
    normalize_calls,
)

CFG = load_config()


def _raw_df():
    return pd.DataFrame(
        {
            "ticker": ["aapl", "aapl", "msft"],
            "datetime": ["2020-01-15 17:30:00", "2020-04-30 08:00:00", "2020-01-16 00:00:00"],
            "structured_content": [
                [
                    {"speaker": "Operator", "text": "Welcome."},
                    {"speaker": "CEO", "text": "Revenue grew strongly and margins expanded."},
                ],
                [{"speaker": "CFO", "text": "Guidance is stable."}],
                [],
            ],
            "year": [2020, 2020, 2020],
            "quarter": [1, 2, 1],
        }
    )


def test_normalize_uppercases_and_flags_midnight_as_no_time():
    calls = normalize_calls(_raw_df(), CFG)
    assert list(calls["ticker"]) == ["AAPL", "AAPL", "MSFT"]
    # 17:30 e 08:00 têm horário; 00:00:00 é tratado como ausente/suspeito.
    assert list(calls["has_time"]) == [True, True, False]
    assert calls["call_id"].is_unique


def test_dedupe_ids_suffixes_duplicates():
    out = _dedupe_ids(pd.Series(["A", "A", "B", "A"]))
    assert list(out) == ["A", "A_2", "B", "A_3"]


def test_extract_utterance_synonyms_and_empty():
    assert _extract_utterance({"name": "X", "content": "hi"}) == ("X", "hi")
    assert _extract_utterance({"speaker": "Y", "text": "   "}) is None  # texto vazio -> ignorado
    assert _extract_utterance({"foo": 1}) is None


def test_parse_structured_content_formats():
    items = [{"speaker": "A", "text": "one"}, {"speaker": "B", "text": "two"}]
    assert _parse_structured_content(items) == [("A", "one"), ("B", "two")]
    assert _parse_structured_content(json.dumps(items)) == [("A", "one"), ("B", "two")]
    assert _parse_structured_content("texto puro sem estrutura") == []  # cai na qualidade
    assert _parse_structured_content({"speaker": "A", "text": "x"}) == []  # dict solto, não lista


def test_build_utterances_preserves_order():
    calls = normalize_calls(_raw_df(), CFG)
    utts = build_utterances(calls)
    first_call = utts[utts["call_id"] == calls["call_id"].iloc[0]]
    assert list(first_call["utterance_idx"]) == [0, 1]
    assert list(first_call["speaker"]) == ["Operator", "CEO"]


def test_filter_quality_drops_short_and_few():
    calls_meta = pd.DataFrame(
        {"call_id": ["A", "B", "C"], "n_chars_total": [600, 100, 700], "n_utterances": [6, 6, 2]}
    )
    utts = pd.DataFrame({"call_id": ["A", "B", "C"], "utterance_idx": [0, 0, 0]})
    kept, kept_utts, report = filter_quality(calls_meta, utts, CFG)  # limiares: 500 chars, 5 falas
    assert list(kept["call_id"]) == ["A"]
    assert report.n_dropped_short == 1  # B
    assert report.n_dropped_few_utts == 1  # C (poucas falas, mas texto ok)
    assert set(kept_utts["call_id"]) == {"A"}


def test_calls_metadata_counts():
    calls = normalize_calls(_raw_df(), CFG)
    utts = build_utterances(calls)
    meta = calls_metadata(calls, utts)
    assert "_content" not in meta.columns
    aapl_q1 = meta[meta["call_id"] == calls["call_id"].iloc[0]].iloc[0]
    assert aapl_q1["n_utterances"] == 2
