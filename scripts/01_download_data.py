"""01 — Baixa transcrições (HuggingFace) e preços (yfinance), salva parquets.

Orquestração fina (regra nº 10): carrega config, chama a biblioteca e grava.
Pré-requisito das demais fases. Rode antes do 00_diagnostics.

Uso:
    python scripts/01_download_data.py --config config.yaml
"""

from __future__ import annotations

import argparse

from tonediv.config import configure_logging, load_config
from tonediv.data import prices as pr
from tonediv.data import transcripts as tr
from tonediv.data import universe as un


def main() -> None:
    parser = argparse.ArgumentParser(description="Download de transcrições e preços.")
    parser.add_argument("--config", default=None, help="Caminho do config.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)
    cfg.paths.ensure_dirs()
    p = cfg.paths

    # --- Transcrições: baixa, normaliza e aplica aliases (ticker canônico) ---
    calls = tr.prepare_calls(tr.load_hf_transcripts(cfg), cfg)
    calls.drop(columns="_content").to_parquet(p.data_raw / "calls_all.parquet", index=False)

    # --- Universo: rede programática (só o "resto desconhecido") ∪ curada ---
    present = {t for t in calls["ticker"].dropna().tolist()}
    u = cfg.universe
    remainder = sorted(present - u.curated_core() - u.sensitivity_tickers() - set(u.delisted_check))
    sector_df = un.fetch_sector_industry(
        remainder, cfg, cache_path=p.data_raw / "sector_cache.parquet"
    )
    universe_df = un.resolve_universe(present, sector_df, cfg)

    # Preços de TODOS os candidatos presentes (todos os blocos), p/ qualquer rodada.
    full_tickers = un.select_tickers(universe_df, active_blocks=u.sensitivity_blocks.keys())
    calls_kept, utts_kept, report = tr.build_universe_tables(calls, full_tickers, cfg)

    price = pr.download_prices(full_tickers, cfg)
    panel = pr.add_daily_returns(price.panel)
    universe_df["in_prices"] = universe_df["ticker"].isin(set(price.succeeded))

    # --- Persistência (schemas em docs/data_schemas.md) ---
    universe_df.to_parquet(p.data_interim / "universe.parquet", index=False)
    calls_kept.to_parquet(p.data_interim / "calls.parquet", index=False)
    utts_kept.to_parquet(p.data_interim / "utterances.parquet", index=False)
    panel.to_parquet(p.data_raw / "prices.parquet", index=False)

    n_in_dataset = int(universe_df["in_dataset"].sum())
    print(
        f"[01] calls_all={len(calls)} | universo(no dataset)={n_in_dataset} "
        f"| calls_universo(qualidade ok)={report.n_kept} | falas={len(utts_kept)} "
        f"| preços ok={len(price.succeeded)} falha={len(price.failed)}"
    )


if __name__ == "__main__":
    main()
