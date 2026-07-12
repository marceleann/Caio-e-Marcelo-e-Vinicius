"""Testes da análise de decay (Fase 5): detecção de mudança de regime."""

import numpy as np
import pandas as pd
import pytest

from tonediv.backtest.decay import (
    _moving_block_sample,
    compare_car,
    compare_ic,
    compare_sharpe,
)
from tonediv.config import load_config

CFG = load_config()


def _ic_events(pre_signal: bool, post_signal: bool, seed: int = 0):
    """Eventos trimestrais 2010–2019; sinal preditivo liga/desliga por regime."""
    rng = np.random.default_rng(seed)
    rows = []
    for qdate in pd.date_range("2010-03-31", periods=40, freq="QE"):
        f = rng.normal(size=15)
        pred = pre_signal if qdate.year < 2015 else post_signal
        r = f if pred else rng.normal(size=15)  # ret=feature (IC 1) ou ruído (IC 0)
        for fi, ri in zip(f, r, strict=True):
            rows.append({"decision_date": qdate, "feat": fi, "ret": ri})
    return pd.DataFrame(rows)


def test_ic_decay_detected():
    # Sinal forte pré-2015, nulo pós-2015 -> IC despenca -> diferença significativa.
    ev = _ic_events(pre_signal=True, post_signal=False)
    rng = np.random.default_rng(CFG.seed)
    c = compare_ic(ev, "feat", "ret", "decision_date", 2015, 500, rng)
    assert c.value_pre > 0.9
    assert abs(c.value_post) < 0.2
    assert c.diff < 0
    assert c.p_value < 0.01


def test_ic_no_decay_high_p():
    # Sinal nulo nos DOIS regimes -> sem diferença -> p alto (margem folgada).
    ev = _ic_events(pre_signal=False, post_signal=False)
    rng = np.random.default_rng(CFG.seed)
    c = compare_ic(ev, "feat", "ret", "decision_date", 2015, 500, rng)
    assert c.p_value > CFG.validation.noise_pvalue_floor


def test_ic_stable_signal_no_decay():
    # Sinal forte e ESTÁVEL nos dois regimes -> IC alto nos dois, sem decay.
    ev = _ic_events(pre_signal=True, post_signal=True)
    rng = np.random.default_rng(CFG.seed)
    c = compare_ic(ev, "feat", "ret", "decision_date", 2015, 500, rng)
    assert c.value_pre > 0.9 and c.value_post > 0.9
    assert c.p_value > CFG.validation.noise_pvalue_floor


def _car_events(pre_arr, post_arr):
    dates = pd.to_datetime(
        [f"{y}-06-15" for y in range(2010, 2015) for _ in range(len(pre_arr) // 5)]
        + [f"{y}-06-15" for y in range(2015, 2020) for _ in range(len(post_arr) // 5)]
    )
    return pd.DataFrame({"decision_date": dates, "car": np.concatenate([pre_arr, post_arr])})


def test_car_decay_detected():
    rng = np.random.default_rng(1)
    ev = _car_events(rng.normal(0.03, 0.02, 300), rng.normal(0.0, 0.02, 300))
    c = compare_car(ev, "car", "decision_date", 2015, 500, np.random.default_rng(CFG.seed))
    assert c.value_pre == pytest.approx(0.03, abs=0.01)
    assert c.diff < 0
    assert c.p_value < 0.01


def test_car_no_decay_identical_regimes():
    # Regimes LITERALMENTE iguais: o teste NÃO pode fabricar diferença (p alto).
    # Construção determinística evita a flakiness do p uniforme sob H0 real.
    half = np.random.default_rng(1).normal(0.0, 0.02, 300)
    ev = _car_events(half, half.copy())
    c = compare_car(ev, "car", "decision_date", 2015, 500, np.random.default_rng(CFG.seed))
    assert c.diff == pytest.approx(0.0, abs=1e-12)
    assert c.p_value > 0.5


def _returns(pre_vals, post_vals):
    idx_pre = pd.bdate_range("2010-01-01", periods=len(pre_vals))
    idx_post = pd.bdate_range("2015-01-01", periods=len(post_vals))
    return pd.concat([pd.Series(pre_vals, index=idx_pre), pd.Series(post_vals, index=idx_post)])


def test_sharpe_decay_direction_and_significance():
    # Efeito FORTE (Sharpe pré ~3, pós ~0): robusto ao ruído da estimação.
    rng = np.random.default_rng(2)
    r = _returns(rng.normal(0.003, 0.01, 1300), rng.normal(0.0, 0.01, 1300))
    c = compare_sharpe(
        r, pd.Timestamp("2015-01-01"), 252, block=5, n_iter=500, rng=np.random.default_rng(CFG.seed)
    )
    assert c.value_pre > c.value_post  # Sharpe caiu
    assert c.diff < 0
    assert c.p_value < 0.05


def test_sharpe_no_decay_identical_regimes():
    # Regimes idênticos -> Sharpe igual -> diferença ~0 -> p alto.
    half = np.random.default_rng(2).normal(0.0005, 0.01, 1300)
    r = _returns(half, half.copy())
    c = compare_sharpe(
        r, pd.Timestamp("2015-01-01"), 252, block=5, n_iter=500, rng=np.random.default_rng(CFG.seed)
    )
    assert c.diff == pytest.approx(0.0, abs=1e-12)
    assert c.p_value > 0.5


def test_moving_block_sample_preserves_length():
    rng = np.random.default_rng(0)
    x = np.arange(100.0)
    out = _moving_block_sample(x, block=7, rng=rng)
    assert len(out) == len(x)
    assert set(np.unique(out)).issubset(set(x))


def _ar1(mean, phi, n, rng, sigma=0.01):
    """Série AR(1): autocorrelação lag-1 ≈ phi, marginal ~ sigma."""
    innov = rng.normal(0, sigma * np.sqrt(1 - phi**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + innov[i]
    return mean + x


def test_sharpe_block_bootstrap_respects_autocorrelation():
    """O CRÍTICO da auditoria da Fase 5: o parâmetro `block` PRECISA importar.

    Em dados autocorrelacionados (o caso que o método existe para tratar,
    ADR-019), o bootstrap i.i.d. (block=1) subestima a incerteza e declara
    significância FALSA; os blocos preservam a autocorrelação e dão o p correto.
    Sem este teste, trocar o block bootstrap por i.i.d. passaria despercebido.
    """
    rng = np.random.default_rng(11)
    pre = _ar1(0.0009, 0.6, 1300, rng)  # efeito MODERADO + autocorr forte
    post = _ar1(0.0, 0.6, 1300, rng)
    r = _returns(pre, post)
    split = pd.Timestamp("2015-01-01")
    p_block = compare_sharpe(r, split, 252, 5, 800, np.random.default_rng(0)).p_value
    p_iid = compare_sharpe(r, split, 252, 1, 800, np.random.default_rng(0)).p_value
    # Inversão material: blocos -> "não significativo" (correto); i.i.d. ->
    # "significativo" (falso). O block bootstrap dá p bem maior.
    assert p_iid < 0.05 < p_block
    assert p_block > 3.0 * p_iid
