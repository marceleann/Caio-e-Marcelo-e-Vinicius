"""Testes de vazamento point-in-time (Fase 3) — inclui a armadilha nº 2.

O teste central prova que a guarda compara TIMESTAMPS COMPLETOS: uma call às
19h com decisão (fabricada) no open do MESMO dia DEVE disparar LookaheadError.
Guardas que normalizam para data deixam esse caso passar — foi o defeito do
protótipo anterior.
"""

import numpy as np
import pandas as pd
import pytest

from tonediv.align.pit import (
    LookaheadError,
    align_decisions,
    assert_no_lookahead,
    attach_forward_returns,
    decision_intent,
    decision_open_timestamp,
    merge_asof_backward,
)
from tonediv.config import load_config

CFG = load_config()
TZ = CFG.sample.timezone


def _calls(rows):
    ts = pd.to_datetime([r[0] for r in rows]).tz_localize(TZ)
    return pd.DataFrame(
        {
            "call_id": [f"C{i}" for i in range(len(rows))],
            "ticker": [r[1] for r in rows],
            "call_datetime": ts,
            "has_time": [r[2] for r in rows],
        }
    )


def _prices(ticker="AAA", start="2020-01-06", days=30):
    # Pregões seguidos (dias úteis) com open=100+i, close=101+i -> retornos à mão.
    dates = pd.bdate_range(start, periods=days)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "open": 100.0 + np.arange(days),
            "close": 101.0 + np.arange(days),
        }
    )


# ---------------------------------------------------------------------------
# Regras de intenção (ADR-002, com buffer de duração da call)
# ---------------------------------------------------------------------------
def test_decision_rules_by_time_of_day():
    calls = _calls(
        [
            ("2020-01-07 19:00", "AAA", True),  # pós-fechamento -> next_open
            ("2020-01-07 08:00", "AAA", True),  # termina 09:15 < open -> same_open
            ("2020-01-07 12:00", "AAA", True),  # durante pregão -> next_open
            ("2020-01-07 11:00", "AAA", False),  # sem horário    -> next_open
        ]
    )
    out = decision_intent(calls, CFG.pit)
    assert list(out["decision_rule"]) == ["next_open", "same_open", "next_open", "next_open"]
    assert out.loc[0, "intended_date"] == pd.Timestamp("2020-01-08")
    assert out.loc[1, "intended_date"] == pd.Timestamp("2020-01-07")


def test_call_still_running_at_open_goes_next_day():
    # Achado 3/3 da revisão: call 09:00 TERMINA ~10:15 (>= open). O Q&A — fonte
    # da tese — ainda não tinha sido falado às 09:30: decide no pregão seguinte.
    calls = _calls([("2020-01-07 09:00", "AAA", True), ("2020-01-07 08:30", "AAA", True)])
    out = decision_intent(calls, CFG.pit)
    assert list(out["decision_rule"]) == ["next_open", "next_open"]


def test_exact_open_boundary_is_next_open():
    # Mata o mutante `<` -> `<=` no cutoff: exatamente no corte NÃO é same_open.
    cutoff = "2020-01-07 08:15"  # open 09:30 − buffer 75min
    out = decision_intent(_calls([(cutoff, "AAA", True)]), CFG.pit)
    assert out.loc[0, "decision_rule"] == "next_open"


def test_midnight_time_is_suspicious_fallback():
    calls = _calls([("2020-01-07 00:00", "AAA", True)])  # 00:00:00 na lista suspeita
    out = decision_intent(calls, CFG.pit)
    assert out.loc[0, "decision_rule"] == "next_open"


def test_nat_call_datetime_yields_no_timestamp_rule():
    # NaT não pode crashar nem decidir: regra própria + intended NaT.
    calls = _calls([("2020-01-07 19:00", "AAA", True)])
    calls.loc[0, "call_datetime"] = pd.NaT
    out = decision_intent(calls, CFG.pit)
    assert out.loc[0, "decision_rule"] == "no_timestamp"
    assert pd.isna(out.loc[0, "intended_date"])


# ---------------------------------------------------------------------------
# ARMADILHA Nº 2: a guarda pega decisão no mesmo dia de call pós-fechamento
# ---------------------------------------------------------------------------
def test_guard_catches_same_day_decision_for_after_close_call():
    call_ts = pd.Series(pd.to_datetime(["2020-01-07 19:00"])).dt.tz_localize(TZ)
    # Decisão FABRICADA no open do MESMO dia: 09:30 < 19:00 -> vazamento.
    decision_ts = pd.Series(pd.to_datetime(["2020-01-07 09:30"])).dt.tz_localize(TZ)
    with pytest.raises(LookaheadError):
        assert_no_lookahead(call_ts, decision_ts)


def test_guard_passes_next_day_decision():
    call_ts = pd.Series(pd.to_datetime(["2020-01-07 19:00"])).dt.tz_localize(TZ)
    decision_ts = pd.Series(pd.to_datetime(["2020-01-08 09:30"])).dt.tz_localize(TZ)
    assert_no_lookahead(call_ts, decision_ts)  # não levanta


def test_guard_catches_equal_timestamps():
    ts = pd.Series(pd.to_datetime(["2020-01-07 09:30"])).dt.tz_localize(TZ)
    with pytest.raises(LookaheadError):
        assert_no_lookahead(ts, ts.copy())  # igualdade também é violação


def test_full_pipeline_never_leaks():
    # Ponta a ponta: intenção -> alinhamento -> guarda, para os 4 tipos de call.
    calls = _calls(
        [
            ("2020-01-07 19:00", "AAA", True),
            ("2020-01-07 08:00", "AAA", True),
            ("2020-01-07 12:00", "AAA", True),
            ("2020-01-07 11:00", "AAA", False),
        ]
    )
    events = align_decisions(decision_intent(calls, CFG.pit), _prices())
    open_ts = decision_open_timestamp(events, CFG.pit, TZ)
    assert_no_lookahead(events["call_datetime"], open_ts)  # não levanta


# ---------------------------------------------------------------------------
# Alinhamento a pregões reais
# ---------------------------------------------------------------------------
def test_friday_evening_call_decides_monday():
    calls = _calls([("2020-01-10 18:00", "AAA", True)])  # sexta pós-fechamento
    events = align_decisions(decision_intent(calls, CFG.pit), _prices())
    assert events.loc[0, "decision_date"] == pd.Timestamp("2020-01-13")  # segunda


def test_ticker_without_prices_gets_nat():
    calls = _calls([("2020-01-07 19:00", "ZZZ", True)])
    events = align_decisions(decision_intent(calls, CFG.pit), _prices("AAA"))
    assert pd.isna(events.loc[0, "decision_date"])


# ---------------------------------------------------------------------------
# Retornos futuros: entrada open d0, saída close d_{H-1} (à mão)
# ---------------------------------------------------------------------------
def test_forward_returns_hand_computed():
    calls = _calls([("2020-01-07 19:00", "AAA", True)])  # decide 2020-01-08
    prices = _prices()  # 01-08 é o 3º pregão (i=2): open=102; close d0+2 (i=4)=105
    events = align_decisions(decision_intent(calls, CFG.pit), prices)
    out = attach_forward_returns(events, prices, horizons=(3,))
    assert out.loc[0, "entry_open"] == pytest.approx(102.0)
    assert out.loc[0, "ret_3"] == pytest.approx(105.0 / 102.0 - 1.0, abs=1e-12)


def test_forward_returns_nan_when_series_ends():
    calls = _calls([("2020-01-07 19:00", "AAA", True)])
    prices = _prices(days=4)  # decisão em i=2; H=10 estoura a série
    events = align_decisions(decision_intent(calls, CFG.pit), prices)
    out = attach_forward_returns(events, prices, horizons=(10,))
    assert np.isnan(out.loc[0, "ret_10"])


# ---------------------------------------------------------------------------
# merge_asof backward ESTRITO: nem futuro, nem o PRÓPRIO dia (painéis EOD)
# ---------------------------------------------------------------------------
def test_merge_asof_backward_strictly_before():
    events = pd.DataFrame({"ticker": ["AAA"], "date": [pd.Timestamp("2020-01-08")]})
    panel = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "date": pd.to_datetime(["2020-01-06", "2020-01-08", "2020-01-09"]),
            # 2.0 é EOD do PRÓPRIO dia da decisão (conhecido só ~6,5h após o
            # open) e 999 é futuro: nenhum dos dois pode aparecer. Este teste
            # também mata o mutante direction="forward" (pegaria 2.0 ou 999).
            "adv": [1.0, 2.0, 999.0],
        }
    )
    out = merge_asof_backward(events, panel, on="date", by="ticker", cols=["adv"])
    assert out.loc[0, "adv"] == pytest.approx(1.0)  # último valor ESTRITAMENTE < 01-08


def test_merge_asof_backward_preserves_nat_rows():
    events = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": [pd.Timestamp("2020-01-08"), pd.NaT],  # NaT = sem pregão de decisão
        }
    )
    panel = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": pd.to_datetime(["2020-01-06"]),
            "adv": [1.0],
        }
    )
    out = merge_asof_backward(events, panel, on="date", by="ticker", cols=["adv"])
    assert len(out) == 2  # linha NaT preservada, não descartada nem crash
    assert out.loc[0, "adv"] == pytest.approx(1.0)
    assert np.isnan(out.loc[1, "adv"])
    assert list(out["ticker"]) == ["AAA", "BBB"]  # ordem original preservada


def test_merge_asof_backward_mismatched_by_dtype():
    # Bug da 1ª execução real (scripts 06-09): eventos vêm do parquet com
    # ``ticker`` StringDtype e o painel de preços com ``ticker`` object;
    # pd.merge_asof exige a chave ``by`` com dtype IDÊNTICO e levantava
    # MergeError. A função deve alinhar os dtypes e casar normalmente.
    events = pd.DataFrame(
        {"ticker": pd.array(["AAA"], dtype="string"), "date": [pd.Timestamp("2020-01-08")]}
    )
    panel = pd.DataFrame(
        {
            "ticker": np.array(["AAA"], dtype=object),
            "date": pd.to_datetime(["2020-01-06"]),
            "adv": [1.0],
        }
    )
    assert events["ticker"].dtype == "string" and panel["ticker"].dtype == object
    out = merge_asof_backward(events, panel, on="date", by="ticker", cols=["adv"])
    assert out.loc[0, "adv"] == pytest.approx(1.0)  # não crasha e casa corretamente


# ---------------------------------------------------------------------------
# Casos da auditoria: NaT no attach, borda t−90d, invariância à ordem
# ---------------------------------------------------------------------------
def test_attach_forward_returns_handles_nat_decision():
    calls = _calls([("2020-01-07 19:00", "ZZZ", True)])  # sem preços -> NaT
    events = align_decisions(decision_intent(calls, CFG.pit), _prices("AAA"))
    out = attach_forward_returns(events, _prices("AAA"), horizons=(3,))
    assert np.isnan(out.loc[0, "entry_open"])
    assert np.isnan(out.loc[0, "ret_3"])


def test_align_decisions_invariant_to_input_order():
    calls = _calls(
        [
            ("2020-01-07 19:00", "AAA", True),
            ("2020-01-08 08:00", "AAA", True),
            ("2020-01-10 12:00", "AAA", True),
        ]
    )
    shuffled = calls.sample(frac=1.0, random_state=3)
    a = align_decisions(decision_intent(calls, CFG.pit), _prices())
    b = align_decisions(decision_intent(shuffled, CFG.pit), _prices()).sort_index()
    pd.testing.assert_series_equal(a["decision_date"], b["decision_date"])
