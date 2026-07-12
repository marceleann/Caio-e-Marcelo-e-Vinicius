"""03 — Pontua o tom de cada fala com FinBERT (etapa mais cara do pipeline).

Lê ``data/interim/utterances_roles.parquet`` (script 02), carrega o FinBERT
(id2label validado em runtime, ADR-007) e salva
``data/interim/utterance_scores.parquet`` com [p_neg, p_neu, p_pos] por fala.

Tempo: horas em CPU / minutos em GPU (ver README). O scoring é
CHECKPOINTADO em fatias (``data/interim/score_shards/``): se interrompido
(máquina dormindo, reinício), basta re-executar — retoma de onde parou.

Uso:
    python scripts/03_score_tone.py --config config.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from tonediv.config import configure_logging, load_config
from tonediv.nlp import scorer as sc


def main() -> None:
    parser = argparse.ArgumentParser(description="Scoring de tom com FinBERT.")
    parser.add_argument("--config", default=None, help="Caminho do config.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)
    p = cfg.paths

    utterances = pd.read_parquet(p.data_interim / "utterances_roles.parquet")
    bundle = sc.load_finbert(cfg)
    scores = sc.score_utterances_checkpointed(
        utterances, cfg, bundle, p.data_interim / "score_shards"
    )
    scores.to_parquet(p.data_interim / "utterance_scores.parquet", index=False)

    by_role = scores.groupby("role")["net_tone"].mean().round(4).to_dict()
    print(
        f"[03] falas pontuadas={len(scores)} | device={bundle.device} "
        f"| chunks médios/fala={scores['n_chunks'].mean():.2f} "
        f"| net_tone médio por papel={by_role}"
    )


if __name__ == "__main__":
    main()
