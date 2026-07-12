"""07 — Grade de robustez COMPLETA: horizonte × feature (+ DSR com n_trials real).

Varre TODAS as células {horizonte} × {feature} no bloco de DESENVOLVIMENTO
(holdout preservado), reporta a grade inteira (sem cherry-picking), o DSR do
campeão usando o número REAL de células, o PBO/CSCV sobre as séries da grade e
o placebo intra-janela da feature principal.

Uso:
    python scripts/07_robustness_grid.py --config config.yaml
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import pandas as pd
from scipy import stats

from tonediv.backtest import calendar_portfolio as cp
from tonediv.backtest.metrics import deflated_sharpe_ratio, ic_by_period, ic_ir, sharpe_ratio
from tonediv.backtest.validation import pbo_cscv, placebo_within_window, temporal_holdout_split
from tonediv.config import configure_logging, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade de robustez + DSR real.")
    parser.add_argument("--config", default=None, help="Caminho do config.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)
    p = cfg.paths
    rng = np.random.default_rng(cfg.seed)

    events = pd.read_parquet(p.data_processed / "events.parquet")
    prices = pd.read_parquet(p.data_raw / "prices.parquet")
    dev, _holdout = temporal_holdout_split(events, "decision_date", cfg.validation.holdout_months)

    rows = []
    net_series: dict[str, pd.Series] = {}
    for h in cfg.returns.horizons:
        scfg = dataclasses.replace(cfg.strategy, holding_days=h)
        for feat in cfg.validation.grid_features:
            if feat not in dev.columns:
                rows.append({"horizon": h, "feature": feat, "n_events": 0, "status": "sem_coluna"})
                continue
            liquid = cp.liquidity_filter(dev.dropna(subset=[feat]), prices, scfg)
            if len(liquid) < cfg.validation.min_cell_events:
                # A célula ENTRA na tabela (nada some da grade); só não roda.
                rows.append(
                    {
                        "horizon": h,
                        "feature": feat,
                        "n_events": len(liquid),
                        "status": "abaixo_min_cell_events",
                    }
                )
                continue
            sided = cp.assign_sides(liquid, feat, scfg)
            res = cp.run_portfolio(sided, prices, scfg, cfg.costs.bps_per_side, "long_short")
            ics = ic_by_period(liquid, feat, f"ret_{h}", "decision_date")
            key = f"H{h}|{feat}"
            net_series[key] = res.net_returns
            rows.append(
                {
                    "horizon": h,
                    "feature": feat,
                    "n_events": res.n_events_used,
                    "status": "ok",
                    "sharpe_net": sharpe_ratio(res.net_returns, cfg.metrics.trading_days_year),
                    "sharpe_daily": sharpe_ratio(res.net_returns, 1),
                    "ic_mean": float(ics.mean()) if len(ics) else float("nan"),
                    "ic_ir": ic_ir(ics),
                    "n_days": int(res.net_returns.notna().sum()),
                }
            )

    grid = pd.DataFrame(rows)
    ran = grid[grid["status"] == "ok"] if len(grid) else grid
    n_skipped = int((grid["status"] != "ok").sum()) if len(grid) else 0
    # n_trials do DSR = células que RODARAM (as puladas não geraram Sharpe).
    n_trials = len(ran)
    best = ran.loc[ran["sharpe_net"].idxmax()] if n_trials else None
    dsr = float("nan")
    if best is not None and n_trials >= 1:
        champ = net_series[f"H{int(best['horizon'])}|{best['feature']}"].dropna()
        dsr = deflated_sharpe_ratio(
            observed_sr=float(best["sharpe_daily"]),
            n_obs=len(champ),
            skew=float(stats.skew(champ)),
            kurt=float(stats.kurtosis(champ, fisher=False)),
            n_trials=n_trials,
            trials_sr_var=float(ran["sharpe_daily"].var(ddof=1)) if n_trials > 1 else 0.0,
        )
    returns_matrix = pd.DataFrame(net_series)
    pbo = pbo_cscv(returns_matrix, cfg.validation.pbo_cscv_blocks)
    h0 = cfg.strategy.holding_days
    p_placebo, _ = placebo_within_window(
        dev,
        cfg.strategy.signal_feature,
        f"ret_{h0}",
        "decision_date",
        cfg.validation.placebo_perm_iters,
        rng,
    )

    grid.to_parquet(p.data_outputs / "robustness_grid.parquet", index=False)
    returns_matrix.to_parquet(p.data_outputs / "robustness_returns.parquet")

    print(
        f"[07] grade: {n_trials} células rodadas + {n_skipped} reportadas sem rodar "
        f"(DEV; holdout intocado)"
    )
    if best is not None:
        print(
            f"     campeã: H{int(best['horizon'])}|{best['feature']} "
            f"sharpe_net={best['sharpe_net']:.2f} | DSR(n_trials={n_trials})={dsr:.3f}"
        )
    feat_main = cfg.strategy.signal_feature
    print(f"     PBO/CSCV={pbo:.3f} | placebo intra-janela ({feat_main}) p={p_placebo:.3f}")


if __name__ == "__main__":
    main()
