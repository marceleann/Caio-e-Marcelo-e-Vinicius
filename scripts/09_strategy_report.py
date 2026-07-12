"""09 — Tearsheet final: curva de capital vs. QQQ/SPY, tabela líquida, drawdown.

Roda a estratégia na amostra COMPLETA (relatório final; o holdout foi
preservado durante o desenvolvimento e é revelado aqui). Gera o PNG do
tearsheet, a tabela consolidada de métricas e o resumo por variante.

Uso:
    python scripts/09_strategy_report.py --config config.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from tonediv.backtest import report
from tonediv.backtest.calendar_portfolio import build_strategy
from tonediv.backtest.metrics import performance_summary
from tonediv.config import configure_logging, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Tearsheet e relatório final.")
    parser.add_argument("--config", default=None, help="Caminho do config.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)
    p, scfg = cfg.paths, cfg.strategy
    ppy = cfg.metrics.trading_days_year

    events = pd.read_parquet(p.data_processed / "events.parquet")
    prices = pd.read_parquet(p.data_raw / "prices.parquet")

    curves: dict[str, pd.Series] = {}
    rows = []
    for variant in scfg.variants:
        res = build_strategy(
            events, prices, scfg, cfg.costs.bps_per_side, scfg.signal_feature, variant
        )
        curves[variant] = report.equity_curve(res.net_returns)
        summ = performance_summary(
            res.gross_returns,
            res.turnover,
            cfg.costs.bps_per_side,
            ppy,
            cfg.metrics.newey_west_lags,
        )
        rows.append({"serie": variant, "n_events": res.n_events_used, **summ})

    main_variant = scfg.variants[0]
    index = curves[main_variant].index
    benches = report.benchmark_buy_hold(prices, cfg.data_sources.benchmarks, index)
    for name, ret in benches.items():
        curves[name] = report.equity_curve(ret)
        summ = performance_summary(
            ret, ret * 0.0, cfg.costs.bps_per_side, ppy, cfg.metrics.newey_west_lags
        )
        rows.append({"serie": name, "n_events": 0, **summ})

    dd = report.drawdown_series(
        build_strategy(
            events, prices, scfg, cfg.costs.bps_per_side, scfg.signal_feature, main_variant
        ).net_returns
    )
    report.save_tearsheet(
        curves, dd, "Tone Distance — capital vs. benchmarks", p.data_outputs / "tearsheet.png"
    )

    summary = pd.DataFrame(rows).set_index("serie")
    summary = summary.join(report.relative_capital_table(curves).set_index("serie"))
    summary.to_parquet(p.data_outputs / "strategy_report.parquet")

    cols = ["sharpe_net", "cagr", "max_drawdown_net", "annual_turnover", "retorno_total"]
    print(f"[09] tearsheet: data/outputs/tearsheet.png | feature={scfg.signal_feature}")
    print(summary.reindex(columns=cols).round(3).to_string())


if __name__ == "__main__":
    main()
