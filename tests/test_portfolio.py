"""Testes do portfólio calendar-time (Fase 4): caso mínimo verificável à mão."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from tonediv.backtest.calendar_portfolio import assign_sides, run_portfolio
from tonediv.config import load_config

CFG = load_config()


def _scfg(**overrides):
    base = dataclasses.replace(
        CFG.strategy,
        holding_days=2,
        top_fraction=0.5,
        min_rank_history=2,
        max_weight_per_name=1.0,
        signal_direction=1,  # mecânica testada na convenção clássica (alto→long);
        # a direção de produção (-1) tem teste dedicado e vive no config (ADR-022).
    )
    return dataclasses.replace(base, **overrides)


def _prices_flat(ticker="AAA", days=12, start="2020-01-06"):
    # open=100, close=110 no dia 0 do evento etc. — retornos fáceis de conferir:
    # todos os dias: open=100, close=102 -> day0 = +2%; c2c seguinte = 0%.
    dates = pd.bdate_range(start, periods=days)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "open": [100.0] * days,
            "close": [102.0] * days,
            "dollar_volume": [1e9] * days,
        }
    )


def test_assign_sides_pit_percentile():
    dates = pd.to_datetime(["2020-01-06", "2020-01-20", "2020-02-03", "2020-02-17"])
    ev = pd.DataFrame({"decision_date": dates, "ticker": list("ABCD"), "f": [1.0, 2.0, 3.0, 0.5]})
    out = assign_sides(ev, "f", _scfg()).set_index("ticker")
    # A e B: sem histórico mínimo (2) -> side 0.
    assert out.loc["A", "side"] == 0 and out.loc["B", "side"] == 0
    # C: janela {1,2}; 3.0 é o topo (pct=1.0 >= 0.5) -> long.
    assert out.loc["C", "side"] == 1
    # D: janela {1,2,3}; 0.5 é o fundo (pct<=0.5) -> short.
    assert out.loc["D", "side"] == -1


def test_assign_sides_strictly_before_same_day():
    # Dois eventos no MESMO dia não se veem: ambos rankeiam só contra o passado.
    dates = pd.to_datetime(["2020-01-06", "2020-01-07", "2020-02-03", "2020-02-03"])
    ev = pd.DataFrame({"decision_date": dates, "ticker": list("ABCD"), "f": [1.0, 2.0, 9.0, 0.1]})
    out = assign_sides(ev, "f", _scfg())
    same_day = out[out["decision_date"] == "2020-02-03"]
    # Janela de ambos é {1.0, 2.0} (não inclui o outro evento do dia).
    assert set(same_day["signal_pct"].round(3)) == {1.0, 0.0}


def test_assign_sides_direction_inverts_bet():
    # signal_direction=-1 (ADR-022): a aposta INVERTE — vender o sinal ALTO.
    # Mesma montagem de test_assign_sides_pit_percentile; C (topo) vira short
    # e D (fundo) vira long, o espelho exato do caso +1.
    dates = pd.to_datetime(["2020-01-06", "2020-01-20", "2020-02-03", "2020-02-17"])
    ev = pd.DataFrame({"decision_date": dates, "ticker": list("ABCD"), "f": [1.0, 2.0, 3.0, 0.5]})
    out = assign_sides(ev, "f", _scfg(signal_direction=-1)).set_index("ticker")
    assert out.loc["C", "side"] == -1  # topo do sinal -> VENDIDO (era +1 em long)
    assert out.loc["D", "side"] == 1  # fundo do sinal -> COMPRADO (era +1 em short)


def test_portfolio_single_event_hand_computed():
    prices = _prices_flat()
    ev = pd.DataFrame(
        {
            "decision_date": [pd.Timestamp("2020-01-13")],  # 6º pregão
            "ticker": ["AAA"],
            "side": [1],
        }
    )
    res = run_portfolio(ev, prices, _scfg(), bps_per_side=0.0, variant="long_short")
    # H=2, 1 evento -> peso 1/(2×1)=0.5. Dia 0: +2% × 0.5 = +1%. Dia 1: c2c 0%.
    assert res.gross_returns.loc["2020-01-13"] == pytest.approx(0.01, abs=1e-12)
    assert res.gross_returns.loc["2020-01-14"] == pytest.approx(
        0.5 * (102.0 / 102.0 - 1.0), abs=1e-12
    )
    # Turnover: entra 0.5 no dia 0; sai ao notional CORRENTE no dia 1:
    # 0.5 × (1.02 × 1.00) = 0.51 (day0 rendeu +2%; c2c do dia 1 = 0%).
    assert res.turnover.loc["2020-01-13"] == pytest.approx(0.5)
    assert res.turnover.loc["2020-01-14"] == pytest.approx(0.51, abs=1e-12)
    assert res.n_events_used == 1


def test_portfolio_costs_reduce_returns():
    prices = _prices_flat()
    ev = pd.DataFrame(
        {"decision_date": [pd.Timestamp("2020-01-13")], "ticker": ["AAA"], "side": [1]}
    )
    res = run_portfolio(ev, prices, _scfg(), bps_per_side=10.0, variant="long_short")
    # custo dia 0 = turnover 0.5 × 10bps = 0.0005
    assert res.net_returns.loc["2020-01-13"] == pytest.approx(0.01 - 0.0005, abs=1e-12)


def test_portfolio_short_side_sign():
    prices = _prices_flat()
    ev = pd.DataFrame(
        {"decision_date": [pd.Timestamp("2020-01-13")], "ticker": ["AAA"], "side": [-1]}
    )
    res = run_portfolio(ev, prices, _scfg(), bps_per_side=0.0, variant="long_short")
    assert res.gross_returns.loc["2020-01-13"] == pytest.approx(-0.01, abs=1e-12)


def test_long_only_variant_drops_shorts():
    prices = _prices_flat()
    ev = pd.DataFrame(
        {
            "decision_date": [pd.Timestamp("2020-01-13")] * 2,
            "ticker": ["AAA", "AAA"],
            "side": [1, -1],
        }
    )
    res = run_portfolio(ev, prices, _scfg(), bps_per_side=0.0, variant="long_only")
    assert res.n_events_used == 1  # só o long


def test_max_weight_cap_leaves_residual_in_cash():
    prices = _prices_flat()
    ev = pd.DataFrame(
        {"decision_date": [pd.Timestamp("2020-01-13")], "ticker": ["AAA"], "side": [1]}
    )
    res = run_portfolio(
        ev, prices, _scfg(max_weight_per_name=0.10), bps_per_side=0.0, variant="long_short"
    )
    # Peso bruto seria 0.5; teto 0.10 -> retorno dia 0 = 2% × 0.10.
    assert res.gross_returns.loc["2020-01-13"] == pytest.approx(0.002, abs=1e-12)


def test_overlapping_tranches_average():
    prices = _prices_flat()
    ev = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2020-01-13", "2020-01-14"]),
            "ticker": ["AAA", "AAA"],
            "side": [1, 1],
        }
    )
    res = run_portfolio(ev, prices, _scfg(), bps_per_side=0.0, variant="long_short")
    # Dia 14/01: tranche 1 em c2c (0%) + tranche 2 em day0 (+2%), pesos 0.5 cada
    # -> 0.5×0 + 0.5×0.02 = 1%.
    assert res.gross_returns.loc["2020-01-14"] == pytest.approx(0.01, abs=1e-12)
    assert res.n_positions.loc["2020-01-14"] == 2


# ---------------------------------------------------------------------------
# Mutantes da auditoria da Fase 4
# ---------------------------------------------------------------------------
def test_constant_signal_produces_no_positions():
    # Midrank: sinal constante fica no percentil 0.5 -> side 0 SEMPRE.
    # (Antes: empate resolvido p/ cima -> carteira 100% comprada sem informação.)
    dates = pd.to_datetime([f"2020-01-{d:02d}" for d in range(6, 21)]).sort_values()
    ev = pd.DataFrame(
        {"decision_date": dates, "ticker": [f"T{i}" for i in range(len(dates))], "f": 1.0}
    )
    # top_fraction 0.2 (o default do helper, 0.5, faria as zonas se tocarem
    # exatamente no midrank 0.5 — o caso que o guard do config proíbe).
    out = assign_sides(ev, "f", _scfg(min_rank_history=5, top_fraction=0.2))
    post_warmup = out[out["signal_pct"].notna()]
    assert len(post_warmup) > 0
    assert (post_warmup["side"] == 0).all()
    assert (post_warmup["signal_pct"] == 0.5).all()  # midrank do empate total


def test_rank_window_expires_old_events():
    # Todo o histórico é mais velho que a janela de 90d -> sem ranking -> side 0.
    # Mata o mutante que ignora o limite esquerdo (lo) da janela.
    dates = pd.to_datetime(["2019-01-10", "2019-01-20", "2019-02-01", "2020-06-01"])
    ev = pd.DataFrame({"decision_date": dates, "ticker": list("ABCD"), "f": [1.0, 2.0, 3.0, 9.0]})
    out = assign_sides(ev, "f", _scfg(min_rank_history=2)).set_index("ticker")
    assert out.loc["D", "side"] == 0  # janela vazia (tudo expirou)
    assert np.isnan(out.loc["D", "signal_pct"])


def test_net_exposure_reported_hand_computed():
    prices = _prices_flat()
    ev = pd.DataFrame(
        {"decision_date": [pd.Timestamp("2020-01-13")], "ticker": ["AAA"], "side": [1]}
    )
    res = run_portfolio(ev, prices, _scfg(), bps_per_side=0.0, variant="long_short")
    # 1 long, peso 0.5, H=2: exposição líquida = +0.5 nos dois dias.
    assert res.net_exposure.loc["2020-01-13"] == pytest.approx(0.5)
    assert res.net_exposure.loc["2020-01-14"] == pytest.approx(0.5)


def test_exit_turnover_at_current_notional():
    # Preço sobe 10% no dia: o desmonte negocia w×1.10, não w.
    dates = pd.bdate_range("2020-01-06", periods=6)
    prices = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": dates,
            "open": [100.0] * 6,
            "close": [110.0] * 6,
            "dollar_volume": 1e9,
        }
    )
    ev = pd.DataFrame(
        {"decision_date": [pd.Timestamp("2020-01-08")], "ticker": ["AAA"], "side": [1]}
    )
    scfg = _scfg(holding_days=1, max_weight_per_name=1.0)
    res = run_portfolio(ev, prices, scfg, bps_per_side=0.0, variant="long_short")
    # H=1: entra (1.0) e sai (1.0×1.10) no mesmo dia -> turnover 2.1.
    assert res.turnover.loc["2020-01-08"] == pytest.approx(1.0 + 1.10, abs=1e-12)


def test_trading_calendar_has_no_fabricated_days():
    # Painel sem o pregão de 2020-01-14 (feriado hipotético): a série do
    # portfólio NÃO pode conter esse dia como retorno 0 fabricado.
    dates = pd.to_datetime(["2020-01-13", "2020-01-15", "2020-01-16"])
    prices = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": dates,
            "open": [100.0] * 3,
            "close": [102.0] * 3,
            "dollar_volume": 1e9,
        }
    )
    ev = pd.DataFrame(
        {"decision_date": [pd.Timestamp("2020-01-13")], "ticker": ["AAA"], "side": [1]}
    )
    res = run_portfolio(ev, prices, _scfg(), bps_per_side=0.0, variant="long_short")
    assert pd.Timestamp("2020-01-14") not in res.gross_returns.index


def test_dropped_event_does_not_deflate_cohort_weights():
    # Dois eventos no mesmo cohort, mas um SEM série de preço: o peso do que
    # negocia deve ser 1/(H×1), não 1/(H×2).
    prices = _prices_flat()
    ev = pd.DataFrame(
        {
            "decision_date": [pd.Timestamp("2020-01-13")] * 2,
            "ticker": ["AAA", "SEM_PRECO"],
            "side": [1, 1],
        }
    )
    res = run_portfolio(ev, prices, _scfg(), bps_per_side=0.0, variant="long_short")
    assert res.n_events_used == 1
    assert res.n_events_dropped == 1
    # Peso 1/(2×1)=0.5 -> retorno dia 0 = 2% × 0.5 = 1%.
    assert res.gross_returns.loc["2020-01-13"] == pytest.approx(0.01, abs=1e-12)


def test_liquidity_filter_threshold_and_pit():
    import dataclasses

    from tonediv.backtest.calendar_portfolio import liquidity_filter

    scfg = dataclasses.replace(
        _scfg(), liquidity_adv_window_days=5, liquidity_min_adv_usd=1_000_000.0
    )
    dates = pd.bdate_range("2020-01-06", periods=20)

    def _panel(dv):
        return pd.DataFrame(
            {"ticker": "AAA", "date": dates, "open": 100.0, "close": 100.0, "dollar_volume": dv}
        )

    ev = pd.DataFrame({"decision_date": [dates[10]], "ticker": ["AAA"]})
    assert len(liquidity_filter(ev, _panel(2e6), scfg)) == 1  # líquido: passa
    assert len(liquidity_filter(ev, _panel(5e5), scfg)) == 0  # ilíquido: cai
    # PIT: volume gigante SÓ no dia da decisão não conta (junção estritamente
    # anterior); a média dos dias anteriores continua abaixo do piso.
    spiky = np.full(len(dates), 5e5)
    spiky[10] = 1e12
    assert len(liquidity_filter(ev, _panel(spiky), scfg)) == 0


def test_top_fraction_half_rejected_at_config_load():
    from tonediv.config import _build_strategy, load_config

    raw = dict(load_config().raw)
    bad = {**raw["strategy"], "top_fraction": 0.5}
    with pytest.raises(ValueError, match="top_fraction"):
        _build_strategy({**raw, "strategy": bad})
