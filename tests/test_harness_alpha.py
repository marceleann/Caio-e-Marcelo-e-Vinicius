"""Harness de alpha sintético (regra nº 12): detecta alpha plantado, não inventa em ruído.

O teste constrói um mundo artificial (preços + eventos) onde a verdade é
conhecida por construção, e roda a MESMA máquina usada nos dados reais:
assign_sides -> run_portfolio -> métricas -> placebo. Se a máquina não
recupera o alpha plantado, ela está quebrada; se acha alpha em ruído, ela
fabrica resultado — os dois lados são igualmente eliminatórios.
"""

import dataclasses

import numpy as np
import pandas as pd

from tonediv.backtest.calendar_portfolio import assign_sides, run_portfolio
from tonediv.backtest.metrics import rank_ic, sharpe_ratio
from tonediv.backtest.validation import placebo_within_window
from tonediv.config import load_config

CFG = load_config()


def _synthetic_world(alpha_per_event: float, seed: int, n_tickers: int = 20, days: int = 500):
    """Mundo sintético: random walk + eventos trimestrais com sinal.

    ``alpha_per_event`` é injetado nos H pregões seguintes à decisão dos
    eventos de sinal alto (e subtraído nos de sinal baixo): o sinal PREVÊ
    retorno por construção. Com alpha 0, sinal e retornos são independentes.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-06", periods=days)
    h = 5

    frames = []
    events = []
    for t in range(n_tickers):
        ticker = f"T{t:02d}"
        ret = rng.normal(0.0003, 0.015, days)
        # Eventos a cada ~63 pregões, defasados por ticker.
        ev_pos = np.arange(30 + (t % 21), days - h - 1, 63)
        for pos in ev_pos:
            signal = float(rng.normal())
            # Injeção do alpha: sinal alto -> retorno extra nos H dias pós-decisão.
            ret[pos : pos + h] += alpha_per_event * np.sign(signal) / h
            events.append({"ticker": ticker, "decision_date": dates[pos], "f": signal})
        close = 100 * np.cumprod(1 + ret)
        open_ = close / (1 + ret)  # abre no close anterior: day0 open->close = ret do dia
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "open": open_,
                    "close": close,
                    "dollar_volume": 1e9,
                }
            )
        )
    prices = pd.concat(frames, ignore_index=True)
    ev = pd.DataFrame(events)
    # Retorno futuro H pregões (open d0 -> close d4) p/ IC e placebo.
    fwd = []
    by_ticker = {t: g.reset_index(drop=True) for t, g in prices.groupby("ticker")}
    for _, e in ev.iterrows():
        g = by_ticker[e["ticker"]]
        pos = int(np.searchsorted(g["date"].to_numpy(), np.datetime64(e["decision_date"])))
        fwd.append(g["close"].iloc[pos + h - 1] / g["open"].iloc[pos] - 1.0)
    ev["ret_h"] = fwd
    return prices, ev


def _run_machine(prices, ev):
    from tonediv.backtest.calendar_portfolio import liquidity_filter

    # O alpha é plantado na direção clássica (sinal alto -> retorno alto, linha
    # ~44), então a máquina roda em signal_direction=+1 para recuperá-lo. A
    # direção de produção (-1, vender excesso de otimismo) é escolha empírica
    # separada, testada pelo backtest real + holdout, não por este harness.
    scfg = dataclasses.replace(
        CFG.strategy,
        holding_days=5,
        min_rank_history=10,
        max_weight_per_name=1.0,
        signal_direction=1,
    )
    # Exercita o caminho completo, incl. o filtro de liquidez (auditoria da
    # Fase 4: cobertura zero antes). dollar_volume=1e9 passa o piso assim que
    # a janela de ADV completa; eventos muito cedo caem — comportamento real.
    liquid = liquidity_filter(ev, prices, scfg)
    sided = assign_sides(liquid, "f", scfg)
    res = run_portfolio(sided, prices, scfg, bps_per_side=0.0, variant="long_short")
    return sided, res


def test_harness_recovers_planted_alpha():
    prices, ev = _synthetic_world(alpha_per_event=0.03, seed=CFG.seed)
    sided, res = _run_machine(prices, ev)
    # O sinal é preditivo por construção: IC e Sharpe fortemente positivos.
    assert rank_ic(ev["f"].abs() * np.sign(ev["f"]), ev["ret_h"]) > 0.3
    assert sharpe_ratio(res.gross_returns, CFG.metrics.trading_days_year) > 1.0
    # Placebo intra-janela reconhece o sinal como real.
    rng = np.random.default_rng(CFG.seed)
    p, _ = placebo_within_window(ev, "f", "ret_h", "decision_date", 200, rng)
    assert p < 0.05


def test_harness_finds_nothing_in_pure_noise():
    prices, ev = _synthetic_world(alpha_per_event=0.0, seed=CFG.seed)
    sided, res = _run_machine(prices, ev)
    # Sem alpha plantado: IC ~ 0 e Sharpe modesto (|SR| < 1 com seed fixa).
    assert abs(rank_ic(ev["f"], ev["ret_h"])) < 0.10
    assert abs(sharpe_ratio(res.gross_returns, CFG.metrics.trading_days_year)) < 1.0
    # Placebo NÃO rejeita a nula (margem folgada do config — armadilha nº 5:
    # p é uniforme sob H0; a seed fixa torna a asserção determinística).
    rng = np.random.default_rng(CFG.seed)
    p, _ = placebo_within_window(ev, "f", "ret_h", "decision_date", 200, rng)
    assert p > CFG.validation.noise_pvalue_floor
