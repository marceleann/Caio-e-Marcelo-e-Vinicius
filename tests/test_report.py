"""Testes do módulo de relatório/tearsheet (Fase 5)."""

import numpy as np
import pandas as pd
import pytest

from tonediv.backtest.report import (
    benchmark_buy_hold,
    drawdown_series,
    equity_curve,
    excess_over_benchmark,
    relative_capital_table,
)


def test_equity_curve_compounds():
    r = pd.Series([0.1, -0.1, 0.05])
    curve = equity_curve(r)
    assert curve.iloc[-1] == pytest.approx(1.1 * 0.9 * 1.05, abs=1e-12)


def test_equity_curve_treats_nan_as_flat():
    r = pd.Series([0.1, np.nan, 0.1])
    curve = equity_curve(r)
    assert curve.iloc[-1] == pytest.approx(1.1 * 1.0 * 1.1, abs=1e-12)


def test_drawdown_by_hand():
    r = pd.Series([0.2, -0.5, 0.1])  # 1.0 -> 1.2 -> 0.6 -> 0.66; pico 1.2 -> DD -0.5
    dd = drawdown_series(r)
    assert dd.min() == pytest.approx(-0.5, abs=1e-9)
    assert dd.iloc[0] == pytest.approx(0.0)


def _prices():
    dates = pd.bdate_range("2020-01-01", periods=10)
    return pd.DataFrame({"ticker": "QQQ", "date": dates, "ret_cc": np.full(10, 0.01)})


def test_benchmark_present_aligns_and_returns():
    idx = pd.bdate_range("2020-01-01", periods=10)
    out = benchmark_buy_hold(_prices(), ("QQQ",), idx)
    assert "QQQ" in out
    assert out["QQQ"].iloc[0] == pytest.approx(0.01)


def test_absent_benchmark_omitted_not_zeroed(caplog):
    # O bug pego no smoke test: benchmark ausente NÃO pode virar linha de zeros.
    idx = pd.bdate_range("2020-01-01", periods=10)
    import logging

    with caplog.at_level(logging.WARNING):
        out = benchmark_buy_hold(_prices(), ("QQQ", "SPY"), idx)
    assert "QQQ" in out
    assert "SPY" not in out  # ausente -> omitido, não zerado
    assert any("SPY" in rec.message for rec in caplog.records)


def test_relative_capital_table():
    idx = pd.bdate_range("2020-01-01", periods=253)  # ~1 ano
    curve = pd.Series(np.linspace(1.0, 1.2, len(idx)), index=idx)
    table = relative_capital_table({"strat": curve}).set_index("serie")
    assert table.loc["strat", "retorno_total"] == pytest.approx(0.2, abs=1e-6)
    assert table.loc["strat", "cagr"] == pytest.approx(0.2, abs=0.02)


def test_excess_over_benchmark():
    idx = pd.bdate_range("2020-01-01", periods=100)
    strat = pd.Series(np.full(100, 0.002), index=idx)
    bench = pd.Series(np.full(100, 0.001), index=idx)
    out = excess_over_benchmark(strat, bench, 252)
    assert out["excess_annual"] == pytest.approx(0.001 * 252, abs=1e-9)
    assert np.isnan(out["information_ratio"])  # excesso constante -> TE 0
