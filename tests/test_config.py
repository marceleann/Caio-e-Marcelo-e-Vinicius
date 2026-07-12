"""Testes do carregamento tipado de config (Fase 1)."""

from datetime import date

from tonediv.config import load_config


def test_load_config_types_and_dates():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.sample.start_date == date(2005, 1, 1)
    assert cfg.sample.price_start_date < cfg.sample.start_date  # cobre janela de estimação
    assert cfg.paths.root.is_absolute()


def test_curated_core_and_subtraction():
    cfg = load_config()
    core = cfg.universe.curated_core()
    sens = cfg.universe.sensitivity_tickers()
    # Núcleo contém IT clássico e internet/mídia; NÃO contém nomes de sensibilidade.
    assert {"AAPL", "GOOGL", "NFLX"} <= core
    assert {"V", "AMZN", "ENPH", "JBL"} <= sens
    assert core.isdisjoint(sens)  # regra de subtração é possível porque são disjuntos
    assert "AMZN" not in core  # caso Amazon: nunca no núcleo (ADR-001)
