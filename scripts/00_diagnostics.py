"""00 — Diagnóstico da base: horários, cobertura, qualidade e survivorship.

Lê os parquets gravados pelo 01_download_data.py e gera:
    - docs/quality_report.md (relatório completo);
    - data/outputs/hour_histogram.png (distribuição de horários);
    - data/outputs/coverage_by_year.parquet e coverage_by_ticker.parquet.

Pré-requisito: rodar 01_download_data.py antes.

Uso:
    python scripts/00_diagnostics.py --config config.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from tonediv.config import configure_logging, load_config
from tonediv.data import diagnostics as dg


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico da base de transcrições/preços.")
    parser.add_argument("--config", default=None, help="Caminho do config.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)
    p = cfg.paths

    calls_all = pd.read_parquet(p.data_raw / "calls_all.parquet")
    universe_df = pd.read_parquet(p.data_interim / "universe.parquet")
    calls_uni = pd.read_parquet(p.data_interim / "calls.parquet")
    utts = pd.read_parquet(p.data_interim / "utterances.parquet")

    diag = dg.compute(cfg, calls_all, universe_df, calls_uni, utts)

    hist_path = p.data_outputs / "hour_histogram.png"
    dg.save_hour_histogram(diag.hour_dist, hist_path)
    diag.coverage_year.to_parquet(p.data_outputs / "coverage_by_year.parquet", index=False)
    diag.coverage_ticker.to_parquet(p.data_outputs / "coverage_by_ticker.parquet", index=False)

    report = dg.render_quality_report(diag, hist_rel_path=f"../data/outputs/{hist_path.name}")
    (p.docs / "quality_report.md").write_text(report, encoding="utf-8")

    n_absent = int(diag.survivorship["present"].eq(False).sum())
    n_empresas = diag.coverage_ticker["ticker"].nunique()
    print(
        f"[00] anos={len(diag.coverage_year)} | empresas universo={n_empresas} "
        f"| deslistadas ausentes={n_absent}/{len(diag.survivorship)} "
        f"| relatório: docs/quality_report.md"
    )


if __name__ == "__main__":
    main()
