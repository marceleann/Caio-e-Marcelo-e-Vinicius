"""04 — Constrói as features por call e alinha a preços (point-in-time).

Lê ``utterance_scores`` (script 03), ``calls`` (script 01) e ``prices``
(script 01); produz ``data/processed/events.parquet`` — a tabela canônica de
eventos: uma linha por call com as 7 features (incl. ``_idio``), a decisão
T+1 e os retornos futuros por horizonte. A guarda ``assert_no_lookahead``
roda SEMPRE antes de salvar (ADR-002/003).

Uso:
    python scripts/04_build_features.py --config config.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from tonediv.align import pit
from tonediv.config import configure_logging, load_config
from tonediv.nlp import features as ft


def main() -> None:
    parser = argparse.ArgumentParser(description="Features por call + alinhamento PIT.")
    parser.add_argument("--config", default=None, help="Caminho do config.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)
    p = cfg.paths

    scores = pd.read_parquet(p.data_interim / "utterance_scores.parquet")
    roles = pd.read_parquet(p.data_interim / "utterances_roles.parquet")
    calls = pd.read_parquet(p.data_interim / "calls.parquet")
    prices = pd.read_parquet(p.data_raw / "prices.parquet")

    # Anexa o nome do orador às falas pontuadas — necessário para agrupar por
    # GESTOR no cálculo da distância de tom (tese Angelo com FinBERT, ADR-023).
    scores = scores.merge(
        roles[["call_id", "utterance_idx", "speaker"]],
        on=["call_id", "utterance_idx"],
        how="left",
    )

    feats = ft.aggregate_calls(scores)
    feats = feats.merge(ft.tone_distance_per_call(scores, cfg.features), on="call_id", how="left")
    feats = feats.merge(
        calls[["call_id", "ticker", "call_datetime", "has_time"]], on="call_id", how="left"
    )
    feats = ft.add_history_features(feats)
    feats = ft.add_idio_features(feats, cfg.features)

    events = pit.decision_intent(feats, cfg.pit)
    events = pit.align_decisions(events, prices)
    open_ts = pit.decision_open_timestamp(events, cfg.pit, cfg.sample.timezone)
    # A guarda compara o FIM estimado da call (início + duração assumida): as
    # features usam a transcrição completa, então é o fim que precisa anteceder
    # o open da decisão (auditoria da Fase 3).
    call_end = events["call_datetime"] + pd.Timedelta(minutes=cfg.pit.assumed_call_duration_minutes)
    pit.assert_no_lookahead(call_end, open_ts)
    events["decision_open_ts"] = open_ts
    events = pit.attach_forward_returns(events, prices, cfg.returns.horizons)

    events.to_parquet(p.data_processed / "events.parquet", index=False)

    n_nat = int(events["decision_date"].isna().sum())
    td_ok = events["tone_distance"].notna().mean()
    ret_col = f"ret_{cfg.returns.horizons[0]}"
    print(
        f"[04] eventos={len(events)} | sem pregão de decisão={n_nat} "
        f"| tone_distance disponível={td_ok:.1%} "
        f"| {ret_col} disponível={events[ret_col].notna().mean():.1%} "
        f"| guarda anti-look-ahead: OK"
    )


if __name__ == "__main__":
    main()
