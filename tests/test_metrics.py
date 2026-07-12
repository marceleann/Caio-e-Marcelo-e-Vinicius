"""Testes das métricas (Fase 4) — valores verificáveis no papel."""

import numpy as np
import pandas as pd
import pytest

from tonediv.backtest.costs import apply_costs, total_cost_drag
from tonediv.backtest.metrics import (
    annual_turnover,
    deflated_sharpe_ratio,
    ic_by_period,
    ic_ir,
    max_drawdown,
    newey_west_tstat,
    rank_ic,
    sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Custos
# ---------------------------------------------------------------------------
def test_apply_costs_by_hand():
    gross = pd.Series([0.01, 0.02], index=pd.date_range("2020-01-01", periods=2))
    to = pd.Series([0.5, 0.0], index=gross.index)  # 50% do patrimônio negociado no dia 1
    net = apply_costs(gross, to, bps_per_side=10.0)  # 10 bps
    # dia 1: 0.01 − 0.5×0.001 = 0.0095 ; dia 2: sem trade -> 0.02
    assert net.iloc[0] == pytest.approx(0.0095, abs=1e-12)
    assert net.iloc[1] == pytest.approx(0.02, abs=1e-12)


def test_total_cost_drag_by_hand():
    to = pd.Series([0.1, 0.3])  # média 0.2
    # 0.2 × 5bps × 252 = 0.2 × 0.0005 × 252 = 0.0252 a.a.
    assert total_cost_drag(to, 5.0, 252) == pytest.approx(0.0252, abs=1e-12)


# ---------------------------------------------------------------------------
# Sharpe / t-NW / drawdown / turnover
# ---------------------------------------------------------------------------
def test_sharpe_by_hand():
    r = pd.Series([0.01, -0.01, 0.01, -0.01, 0.02])
    # mean=0.004; std(ddof=1)=sqrt(Σ(x-μ)²/4)... verificado numericamente:
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(r, 252) == pytest.approx(float(expected), abs=1e-12)


def test_sharpe_degenerate_is_nan():
    assert np.isnan(sharpe_ratio(pd.Series([0.01]), 252))
    assert np.isnan(sharpe_ratio(pd.Series([0.01] * 30), 252))  # std 0


def test_newey_west_matches_ols_when_no_autocorr():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 400))
    t_nw = newey_west_tstat(r, lags=0)
    t_plain = float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))
    assert t_nw == pytest.approx(t_plain, rel=0.02)  # lags=0 ~ OLS clássico


def test_max_drawdown_by_hand():
    # Curva: 1.0 -> 1.1 -> 0.99 -> 1.05 ; pico 1.1, vale 0.99 -> DD = −10%
    r = pd.Series([0.10, -0.10, 0.0606060606])
    assert max_drawdown(r) == pytest.approx(-0.10, abs=1e-9)


def test_annual_turnover():
    assert annual_turnover(pd.Series([0.02, 0.04]), 252) == pytest.approx(0.03 * 252)


# ---------------------------------------------------------------------------
# IC / IC-IR
# ---------------------------------------------------------------------------
def test_rank_ic_perfect_and_inverse():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert rank_ic(s, s * 10) == pytest.approx(1.0)
    assert rank_ic(s, -s) == pytest.approx(-1.0)


def test_rank_ic_nan_pairs_dropped():
    s = pd.Series([1.0, 2.0, 3.0, np.nan])
    r = pd.Series([1.0, 2.0, 3.0, 100.0])
    assert rank_ic(s, r) == pytest.approx(1.0)


def test_ic_by_period_and_ir():
    dates = pd.to_datetime(
        ["2020-01-10", "2020-02-10", "2020-03-10", "2020-04-10", "2020-05-10", "2020-06-10"]
    )
    # Q1: sinal e retorno perfeitamente alinhados; Q2: perfeitamente invertidos.
    ev = pd.DataFrame(
        {
            "decision_date": dates,
            "f": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "r": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
        }
    )
    ics = ic_by_period(ev, "f", "r", "decision_date", freq="QE")
    assert list(ics.round(6)) == [1.0, -1.0]
    assert ic_ir(ics) == pytest.approx(0.0, abs=1e-12)  # média 0


# ---------------------------------------------------------------------------
# Deflated Sharpe
# ---------------------------------------------------------------------------
def test_dsr_single_trial_reduces_to_psr():
    # n_trials=1 -> SR0=0: DSR = Φ(SR·sqrt(n−1)) p/ retornos ~normais (skew 0, kurt 3).
    from scipy import stats

    sr = 0.1
    n = 101
    expected = float(stats.norm.cdf(sr * np.sqrt(n - 1) / np.sqrt(1 - 0 + (3 - 1) / 4 * sr**2)))
    got = deflated_sharpe_ratio(sr, n, skew=0.0, kurt=3.0, n_trials=1, trials_sr_var=0.0)
    assert got == pytest.approx(expected, abs=1e-12)


def test_dsr_decreases_with_more_trials():
    kw = dict(n_obs=252, skew=0.0, kurt=3.0, trials_sr_var=0.01)
    few = deflated_sharpe_ratio(0.15, n_trials=2, **kw)
    many = deflated_sharpe_ratio(0.15, n_trials=100, **kw)
    assert many < few  # mais tentativas -> benchmark de sorte mais alto -> DSR menor


def test_dsr_invalid_inputs_nan():
    assert np.isnan(deflated_sharpe_ratio(float("nan"), 100, 0.0, 3.0, 5, 0.01))
    assert np.isnan(deflated_sharpe_ratio(0.1, 1, 0.0, 3.0, 5, 0.01))


def test_dsr_reference_value_independent_derivation():
    # Recalcula a fórmula de Bailey-LdP AQUI, passo a passo, com constantes
    # explícitas — mata mutantes de sinal do skew, dos pesos do SR0 e do
    # sqrt(V) que a comparação qualitativa não pega (auditoria da Fase 4).
    from scipy import stats

    sr, n, sk, ku, trials, var = 0.10, 252, -0.5, 4.0, 30, 0.0004
    g = float(np.euler_gamma)
    z1 = stats.norm.ppf(1 - 1 / trials)
    z2 = stats.norm.ppf(1 - 1 / (trials * np.e))
    sr0 = np.sqrt(var) * ((1 - g) * z1 + g * z2)
    denom = 1 - sk * sr + ((ku - 1) / 4) * sr**2
    expected = float(stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / np.sqrt(denom)))
    got = deflated_sharpe_ratio(sr, n, skew=sk, kurt=ku, n_trials=trials, trials_sr_var=var)
    assert got == pytest.approx(expected, abs=1e-12)
    assert 0.0 < sr0 < sr  # sanidade: benchmark de sorte positivo e abaixo do SR


def test_newey_west_widens_se_under_autocorrelation():
    # Série com autocorrelação POSITIVA (média móvel de ruído): o erro-padrão
    # HAC com lags>0 deve ser maior que o clássico -> t_NW < t_plain. Mata o
    # mutante que ignora o parâmetro lags (auditoria da Fase 4).
    rng = np.random.default_rng(5)
    base = rng.normal(0.002, 0.01, 600)
    r = pd.Series(base).rolling(5).mean().dropna()  # MA(5): autocorr forte
    t_plain = float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))
    t_nw = newey_west_tstat(r, lags=8)
    assert t_nw < 0.75 * t_plain
