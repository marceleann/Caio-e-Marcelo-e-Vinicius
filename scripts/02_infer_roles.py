"""02 — Infere papéis analista×gestão e gera o material de validação manual.

Lê ``data/interim/utterances.parquet`` (Fase 1), aplica a heurística ADR-006 e
salva:
    - data/interim/utterances_roles.parquet (falas anotadas: section/role);
    - data/outputs/roles_validation_sample.csv (conferência manual, seed fixa);
    - docs/roles_report.md (métricas de cobertura da heurística).

Uso:
    python scripts/02_infer_roles.py --config config.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import pandas as pd

from tonediv.config import configure_logging, load_config
from tonediv.nlp import roles as rl


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferência de papéis analista×gestão.")
    parser.add_argument("--config", default=None, help="Caminho do config.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)
    p = cfg.paths

    utterances = pd.read_parquet(p.data_interim / "utterances.parquet")
    annotated = rl.infer_roles(utterances, cfg.roles)
    coverage = rl.coverage_metrics(annotated)
    sample = rl.validation_sample(annotated, cfg.roles)

    annotated.to_parquet(p.data_interim / "utterances_roles.parquet", index=False)
    sample.to_csv(p.data_outputs / "roles_validation_sample.csv", index=False, encoding="utf-8")

    lines = ["# roles_report.md — Cobertura da heurística de papéis (script 02)", ""]
    lines += [f"- **{k}**: {v}" for k, v in asdict(coverage).items()]
    lines += ["", "Amostra de validação manual: `data/outputs/roles_validation_sample.csv`."]
    (p.docs / "roles_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        f"[02] calls={coverage.n_calls} | Q&A detectado={coverage.pct_calls_qa_detected:.1%} "
        f"| falas classificadas={coverage.pct_utterances_classified:.1%} "
        f"| gestão={coverage.n_management} analista={coverage.n_analyst} "
        f"| amostra manual: {len(sample)} linhas"
    )


if __name__ == "__main__":
    main()
