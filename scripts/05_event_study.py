"""05 — Event study: CAR por grupo de feature + placebo de datas falsas.

Lê ``events.parquet`` (script 04) e ``prices.parquet``; produz
``data/outputs/event_study_cars.parquet`` (CAR por evento) e
``data/outputs/event_study_groups.parquet`` (tabela por grupo, com p do
placebo e IC bootstrap nos atributos impressos no resumo).

Uso:
    python scripts/05_event_study.py --config config.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tonediv.backtest import event_study as es
from tonediv.config import configure_logging, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Event study (CAR + placebo).")
    parser.add_argument("--config", default=None, help="Caminho do config.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)
    p = cfg.paths

    from tonediv.backtest.validation import temporal_holdout_split

    all_events = pd.read_parquet(p.data_processed / "events.parquet")
    # Bloco de DESENVOLVIMENTO apenas: o holdout de 18 meses fica intocado até
    # o relatório final (script 09) — auditoria da Fase 4.
    events, _holdout = temporal_holdout_split(
        all_events, "decision_date", cfg.validation.holdout_months
    )
    prices = pd.read_parquet(p.data_raw / "prices.parquet")
    feature = cfg.strategy.signal_feature
    rng = np.random.default_rng(cfg.seed)

    panel = es.build_flat_panel(prices, cfg.data_sources.market_ticker)
    events_car = es.compute_cars(events, panel, cfg.event_study)
    table = es.group_car_table(events_car, feature, cfg.event_study.groups)
    p_placebo, _ = es.placebo_fake_dates(events_car, panel, cfg.event_study, rng)
    ci_lo, ci_hi = es.bootstrap_mean_ci(
        events_car["car"].to_numpy(), cfg.event_study.bootstrap_iters, rng
    )

    events_car.to_parquet(p.data_outputs / "event_study_cars.parquet", index=False)
    table.to_parquet(p.data_outputs / "event_study_groups.parquet", index=False)

    n_valid = int(events_car["car"].notna().sum())
    hml = table.set_index("grupo").loc["HML", "car_medio"] if len(table) else float("nan")
    print(
        f"[05] eventos c/ CAR={n_valid}/{len(events_car)} | feature={feature} "
        f"| CAR médio={events_car['car'].mean():+.4f} [IC95 {ci_lo:+.4f},{ci_hi:+.4f}] "
        f"| HML={hml:+.4f} (p Welch={table.attrs.get('hml_pvalue', float('nan')):.3f}) "
        f"| placebo datas falsas p={p_placebo:.3f}"
    )


if __name__ == "__main__":
    main()
