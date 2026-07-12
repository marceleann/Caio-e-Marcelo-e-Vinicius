"""Testes do event study (Fase 4): recuperação de α/β e CAR sintéticos."""

import numpy as np
import pandas as pd
import pytest

from tonediv.backtest.event_study import (
    build_flat_panel,
    compute_cars,
    group_car_table,
    placebo_fake_dates,
)
from tonediv.config import load_config

CFG = load_config()
ES = CFG.event_study


def _panel_with_alpha(alpha=0.0, beta=1.5, jump=0.0, jump_at=None, days=260, seed=1):
    """Painel sintético: mercado ~N(0, 1%); ação = α + β·mercado (+ salto no evento)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=days)
    rm = rng.normal(0.0, 0.01, days)
    r = alpha + beta * rm
    if jump_at is not None:
        r[jump_at] += jump
    close_m = 100 * np.cumprod(1 + rm)
    close_s = 100 * np.cumprod(1 + r)
    stock = pd.DataFrame({"ticker": "AAA", "date": dates, "ret_cc": r, "close": close_s})
    mkt = pd.DataFrame({"ticker": "^GSPC", "date": dates, "ret_cc": rm, "close": close_m})
    return pd.concat([stock, mkt], ignore_index=True)


def _event_at(dates_panel, pos, ticker="AAA", feature=0.0):
    dates = pd.bdate_range("2020-01-02", periods=300)
    return pd.DataFrame(
        {
            "ticker": [ticker],
            "decision_date": [dates[pos]],
            "feat": [feature],
        }
    )


def test_car_zero_when_stock_follows_market_exactly():
    # Sem salto e sem alpha: AR ~ 0 em toda janela -> CAR ~ 0 (a menos de ruído zero).
    prices = _panel_with_alpha(alpha=0.0, beta=1.5)
    panel = build_flat_panel(prices, "^GSPC")
    ev = compute_cars(_event_at(prices, pos=150), panel, ES)
    assert ev.loc[0, "car"] == pytest.approx(0.0, abs=1e-10)


def test_car_recovers_planted_jump():
    # Salto de +5% no dia do evento (dentro da janela [−1,+10]) -> CAR ≈ 0.05.
    prices = _panel_with_alpha(jump=0.05, jump_at=150)
    panel = build_flat_panel(prices, "^GSPC")
    ev = compute_cars(_event_at(prices, pos=150), panel, ES)
    assert ev.loc[0, "car"] == pytest.approx(0.05, abs=1e-10)


def test_car_jump_outside_event_window_not_captured():
    # Salto 30 pregões APÓS o evento: fora de [−1,+10] -> CAR ~ 0.
    prices = _panel_with_alpha(jump=0.05, jump_at=180)
    panel = build_flat_panel(prices, "^GSPC")
    ev = compute_cars(_event_at(prices, pos=150), panel, ES)
    assert ev.loc[0, "car"] == pytest.approx(0.0, abs=1e-10)


def test_car_nan_without_estimation_history():
    # Evento no pregão 30: não há 120 pregões de estimação -> NaN, sem crash.
    prices = _panel_with_alpha()
    panel = build_flat_panel(prices, "^GSPC")
    ev = compute_cars(_event_at(prices, pos=30), panel, ES)
    assert np.isnan(ev.loc[0, "car"])


def test_group_car_table_hml_direction():
    rng = np.random.default_rng(7)
    n = 90
    feat = rng.normal(size=n)
    car = 0.02 * feat + rng.normal(0, 0.001, n)  # CAR cresce com a feature
    ev = pd.DataFrame({"f": feat, "car": car})
    table = group_car_table(ev, "f", "tercile").set_index("grupo")
    assert table.loc["G3", "car_medio"] > table.loc["G1", "car_medio"]
    assert table.loc["HML", "car_medio"] > 0
    assert table.attrs["hml_pvalue"] < 0.01


def test_car_zero_with_planted_alpha_in_estimation():
    # Ação com alpha diário POSITIVO constante (drift próprio): o market model
    # estima o alpha e o REMOVE -> CAR ~ 0. Mata mutantes que ignoram o termo
    # alpha do CAR (auditoria: todos os painéis anteriores usavam alpha=0).
    prices = _panel_with_alpha(alpha=0.002, beta=1.2)
    panel = build_flat_panel(prices, "^GSPC")
    ev = compute_cars(_event_at(prices, pos=150), panel, ES)
    assert ev.loc[0, "car"] == pytest.approx(0.0, abs=1e-10)


def test_car_recovers_jump_on_top_of_alpha_and_beta():
    prices = _panel_with_alpha(alpha=0.001, beta=0.8, jump=0.04, jump_at=150)
    panel = build_flat_panel(prices, "^GSPC")
    ev = compute_cars(_event_at(prices, pos=150), panel, ES)
    assert ev.loc[0, "car"] == pytest.approx(0.04, abs=1e-10)


def test_placebo_power_detects_planted_effect():
    # PODER do placebo: saltos de +5% plantados NOS eventos reais -> CAR médio
    # alto vs. datas falsas (~0) -> p BAIXO. Um placebo vácuo (que não desloca
    # nada) passaria despercebido sem este teste (auditoria da Fase 4).
    import dataclasses

    rng_prices = np.random.default_rng(3)
    days = 700
    dates = pd.bdate_range("2020-01-02", periods=days)
    rm = rng_prices.normal(0.0, 0.01, days)
    r = 1.1 * rm
    # Espaçamento de 170 pregões > janela de estimação (120): o salto de um
    # evento NÃO contamina a estimação do seguinte (com 80 de espaço, o alpha
    # estimado subia ~5bps/dia e o CAR médio caía p/ 0.045 — verificado).
    event_pos = [150, 320, 490, 660]
    for pos in event_pos:
        r[pos] += 0.05
    prices = pd.concat(
        [
            pd.DataFrame({"ticker": "AAA", "date": dates, "ret_cc": r}),
            pd.DataFrame({"ticker": "^GSPC", "date": dates, "ret_cc": rm}),
        ],
        ignore_index=True,
    )
    panel = build_flat_panel(prices, "^GSPC")
    events = pd.DataFrame({"ticker": "AAA", "decision_date": dates[event_pos], "feat": 0.0})
    ev = compute_cars(events, panel, ES)
    assert ev["car"].mean() == pytest.approx(0.05, abs=1e-6)
    fast = dataclasses.replace(ES, placebo_fake_date_iters=200)
    p, _ = placebo_fake_dates(ev, panel, fast, np.random.default_rng(CFG.seed))
    assert p < 0.05


def test_placebo_fake_dates_high_p_for_no_effect():
    # Ação = β·mercado sem alpha/salto: o CAR observado é ~0 e as datas falsas
    # também -> p alto (sem efeito espúrio). Seed fixa (armadilha nº 5).
    prices = _panel_with_alpha(days=600)
    panel = build_flat_panel(prices, "^GSPC")
    events = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "decision_date": pd.bdate_range("2020-01-02", periods=600)[[200, 280, 360, 440]],
            "feat": [0.0] * 4,
        }
    )
    ev = compute_cars(events, panel, ES)
    rng = np.random.default_rng(CFG.seed)
    import dataclasses

    fast = dataclasses.replace(ES, placebo_fake_date_iters=200)
    p, null = placebo_fake_dates(ev, panel, fast, rng)
    assert len(null) == 200
    assert p > CFG.validation.noise_pvalue_floor
