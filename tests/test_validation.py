"""Testes da validação temporal (Fase 4): purga, embargo, placebo e PBO."""

import numpy as np
import pandas as pd
import pytest

from tonediv.backtest.validation import (
    pbo_cscv,
    placebo_within_window,
    purged_kfold_indices,
    temporal_holdout_split,
)


def _times(n=100, start="2020-01-01"):
    return pd.Series(pd.date_range(start, periods=n, freq="D"))


def test_purged_kfold_covers_all_and_disjoint():
    times = _times()
    seen = []
    for train, test in purged_kfold_indices(times, n_splits=5, embargo_days=3, horizon_days=5):
        assert set(train).isdisjoint(set(test))
        seen.extend(test)
    assert sorted(seen) == list(range(100))  # cada obs é teste exatamente 1x


def test_purged_kfold_purges_overlapping_horizon():
    times = _times()
    for train, test in purged_kfold_indices(times, n_splits=5, embargo_days=0, horizon_days=5):
        t_test_start = times.iloc[sorted(test)[0]]
        before = [i for i in train if times.iloc[i] < t_test_start]
        # Nenhum treino anterior pode ALCANÇAR o teste com horizonte de 5 dias.
        for i in before:
            assert times.iloc[i] + pd.Timedelta(days=5) < t_test_start


def test_purged_kfold_embargo_after_test():
    times = _times()
    folds = list(purged_kfold_indices(times, n_splits=5, embargo_days=10, horizon_days=0))
    train, test = folds[0]  # primeira dobra: teste no INÍCIO, embargo à direita
    t_test_end = times.iloc[sorted(test)[-1]]
    after = [i for i in train if times.iloc[i] > t_test_end]
    for i in after:
        assert times.iloc[i] > t_test_end + pd.Timedelta(days=10)


def test_placebo_detects_real_signal_and_ignores_noise():
    rng = np.random.default_rng(42)
    n = 400
    dates = pd.Series(pd.date_range("2015-01-01", periods=n, freq="5D"))
    signal = rng.normal(size=n)
    ret_with_alpha = 0.05 * signal + rng.normal(0, 0.02, n)
    ev = pd.DataFrame({"decision_date": dates, "f": signal, "r": ret_with_alpha})
    p_real, _ = placebo_within_window(ev, "f", "r", "decision_date", 200, rng)
    assert p_real < 0.05  # sinal real sobrevive ao placebo

    # Ruído puro com seed CONTROLADA: p deve ser alto (margem folgada — o p é
    # uniforme sob H0; a asserção usa o piso do config, armadilha nº 5).
    ret_noise = rng.normal(0, 0.02, n)
    ev_noise = pd.DataFrame({"decision_date": dates, "f": signal, "r": ret_noise})
    p_noise, _ = placebo_within_window(ev_noise, "f", "r", "decision_date", 200, rng)
    assert p_noise > 0.20


def test_placebo_permutes_within_window_only():
    # Sinal = f(janela): permutar DENTRO da janela não muda nada -> IC nulo
    # idêntico ao observado -> p ~ 1. Prova que a permutação respeita a janela.
    dates = pd.Series(pd.to_datetime(["2020-01-05", "2020-02-05", "2020-04-05", "2020-05-05"] * 5))
    quarter = dates.dt.quarter.astype(float)
    ev = pd.DataFrame({"decision_date": dates, "f": quarter, "r": quarter * 0.01})
    rng = np.random.default_rng(0)
    p, null = placebo_within_window(ev, "f", "r", "decision_date", 50, rng)
    assert p == pytest.approx(1.0)


def test_pbo_low_for_persistent_winner_high_for_noise():
    rng = np.random.default_rng(1)
    t = 320
    noise = rng.normal(0, 0.01, (t, 10))
    persistent = noise.copy()
    persistent[:, 0] += 0.004  # estratégia 0 é genuinamente melhor o tempo todo
    pbo_good = pbo_cscv(pd.DataFrame(persistent), n_blocks=8)
    pbo_noise = pbo_cscv(pd.DataFrame(noise), n_blocks=8)
    assert pbo_good < 0.15
    # PBO de ruído varia com a seed (~0.3–0.8 medido); o que importa e é
    # estável é o CONTRASTE: campeão genuíno persiste OOS, campeão de ruído não.
    assert pbo_noise > pbo_good + 0.2


def test_holdout_split_boundaries():
    ev = pd.DataFrame({"decision_date": pd.date_range("2020-01-01", periods=730, freq="D")})
    dev, hold = temporal_holdout_split(ev, "decision_date", holdout_months=6)
    assert len(dev) + len(hold) == 730
    assert dev["decision_date"].max() < hold["decision_date"].min()
    # ~6 meses de holdout
    assert 150 < len(hold) < 210
