"""08 — Análise de decay: IC, CAR e Sharpe pré vs. pós-2015, com teste de diferença.

Usa a amostra COMPLETA (o decay é resultado final descritivo, não seleção de
hiperparâmetro — o holdout não se aplica aqui). Produz
``data/outputs/decay_comparison.parquet``.

Uso:
    python scripts/08_decay_analysis.py --config config.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tonediv.backtest import decay
from tonediv.backtest import event_study as es
from tonediv.backtest.calendar_portfolio import build_strategy
from tonediv.config import configure_logging, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Análise de decay pré/pós-2015.")
    parser.add_argument("--config", default=None, help="Caminho do config.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)
    p = cfg.paths
    feature, h = cfg.strategy.signal_feature, cfg.strategy.holding_days
    split = cfg.decay.split_year
    n_iter = cfg.validation.placebo_perm_iters
    rng = np.random.default_rng(cfg.seed)

    events = pd.read_parquet(p.data_processed / "events.parquet")
    prices = pd.read_parquet(p.data_raw / "prices.parquet")

    panel = es.build_flat_panel(prices, cfg.data_sources.market_ticker)
    events_car = es.compute_cars(events, panel, cfg.event_study)
    res = build_strategy(
        events, prices, cfg.strategy, cfg.costs.bps_per_side, feature, "long_short"
    )
    split_date = pd.Timestamp(f"{split}-01-01")

    comparisons = [
        decay.compare_ic(events, feature, f"ret_{h}", "decision_date", split, n_iter, rng),
        decay.compare_car(events_car, "car", "decision_date", split, n_iter, rng),
        decay.compare_sharpe(
            res.net_returns, split_date, cfg.metrics.trading_days_year, h, n_iter, rng
        ),
    ]
    table = decay.comparisons_to_frame(comparisons)
    table.to_parquet(p.data_outputs / "decay_comparison.parquet", index=False)

    print(f"[08] decay pré vs pós-{split} | feature={feature}")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
