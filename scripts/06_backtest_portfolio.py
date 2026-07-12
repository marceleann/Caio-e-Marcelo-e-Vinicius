"""06 — Backtest calendar-time: variantes + placebo (N perms) + diagnóstico de exposição.

Roda no bloco de DESENVOLVIMENTO (holdout preservado para o script 09).
Produz ``portfolio_returns.parquet`` (séries diárias, incl. exposição líquida
e série hedgeada) e ``portfolio_summary.parquet``. O placebo de sinal
aleatório usa ``strategy.placebo_portfolio_perms`` permutações — um sorteio
único não teria valor inferencial (auditoria da Fase 4).

Uso:
    python scripts/06_backtest_portfolio.py --config config.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tonediv.backtest import calendar_portfolio as cp
from tonediv.backtest.metrics import performance_summary, sharpe_ratio
from tonediv.backtest.validation import temporal_holdout_split
from tonediv.config import configure_logging, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest calendar-time.")
    parser.add_argument("--config", default=None, help="Caminho do config.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)
    p, scfg = cfg.paths, cfg.strategy
    feature = scfg.signal_feature
    rng = np.random.default_rng(cfg.seed)

    all_events = pd.read_parquet(p.data_processed / "events.parquet")
    events, _holdout = temporal_holdout_split(
        all_events, "decision_date", cfg.validation.holdout_months
    )
    prices = pd.read_parquet(p.data_raw / "prices.parquet")
    liquid = cp.liquidity_filter(events.dropna(subset=[feature]), prices, scfg)
    mkt = (
        prices[prices["ticker"] == cfg.data_sources.market_ticker]
        .set_index("date")["ret_cc"]
        .sort_index()
    )

    runs: dict[str, pd.Series] = {}
    summaries = []
    for variant in scfg.variants:
        sided = cp.assign_sides(liquid, feature, scfg)
        res = cp.run_portfolio(sided, prices, scfg, cfg.costs.bps_per_side, variant)
        runs[f"{variant}_net"] = res.net_returns
        runs[f"{variant}_turnover"] = res.turnover
        runs[f"{variant}_net_exposure"] = res.net_exposure
        # Série hedgeada: remove o beta "acidental" dos cohorts unilaterais
        # subtraindo exposição líquida × retorno do mercado (diagnóstico).
        hedged = res.net_returns - res.net_exposure * mkt.reindex(res.net_returns.index).fillna(0.0)
        runs[f"{variant}_net_hedged"] = hedged
        summary = performance_summary(
            res.gross_returns,
            res.turnover,
            cfg.costs.bps_per_side,
            cfg.metrics.trading_days_year,
            cfg.metrics.newey_west_lags,
        )
        summary["sharpe_net_hedged"] = sharpe_ratio(hedged, cfg.metrics.trading_days_year)
        summary["net_exposure_mean_abs"] = float(res.net_exposure.abs().mean())
        summaries.append({"variant": variant, "n_events": res.n_events_used, **summary})

    # Placebo: N permutações do sinal (mesma máquina, variante long_short).
    placebo_sharpes = []
    for _ in range(scfg.placebo_portfolio_perms):
        shuffled = liquid.copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        sided = cp.assign_sides(shuffled, feature, scfg)
        res = cp.run_portfolio(sided, prices, scfg, cfg.costs.bps_per_side, "long_short")
        placebo_sharpes.append(sharpe_ratio(res.net_returns, cfg.metrics.trading_days_year))
    placebo_arr = np.array(placebo_sharpes, dtype=float)
    summaries.append(
        {
            "variant": f"placebo_random_x{scfg.placebo_portfolio_perms}",
            "n_events": len(liquid),
            "sharpe_net": float(np.nanmean(placebo_arr)),
            "sharpe_net_best": float(np.nanmax(placebo_arr)),
            "sharpe_net_std": float(np.nanstd(placebo_arr, ddof=1)),
        }
    )

    pd.DataFrame(runs).to_parquet(p.data_outputs / "portfolio_returns.parquet")
    summary_df = pd.DataFrame(summaries).set_index("variant")
    summary_df.to_parquet(p.data_outputs / "portfolio_summary.parquet")

    cols = [
        "n_events",
        "sharpe_net",
        "sharpe_net_hedged",
        "net_exposure_mean_abs",
        "sharpe_net_best",
    ]
    print(f"[06] feature={feature} | H={scfg.holding_days} | bloco DEV (holdout preservado)")
    print(summary_df.reindex(columns=cols).round(3).to_string())


if __name__ == "__main__":
    main()
