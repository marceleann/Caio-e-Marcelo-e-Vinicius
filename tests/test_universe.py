"""Testes da seleção de universo (Fase 1): rede, subtração e seleção por bloco."""

import pandas as pd

from tonediv.config import load_config
from tonediv.data.universe import (
    apply_aliases,
    programmatic_net,
    resolve_universe,
    select_tickers,
)

CFG = load_config()


def test_apply_aliases_maps_to_canonical():
    out = apply_aliases(pd.Series(["FB", "AAPL", "PCLN"]), {"FB": "META", "PCLN": "BKNG"})
    assert list(out) == ["META", "AAPL", "BKNG"]


def _sector_df():
    # UBER: setor Technology (Yahoo classifica assim) -> entra na rede...
    # NFLX: extra ticker, vence o veto de "Entertainment".
    # T: Telecom Services -> vetado. GOOGL: internet -> entra. XYZ: banco -> fora.
    return pd.DataFrame(
        {
            "ticker": ["UBER", "NFLX", "T", "GOOGL", "XYZ"],
            "sector": [
                "Technology",
                "Communication Services",
                "Communication Services",
                "Communication Services",
                "Financial Services",
            ],
            "industry": [
                None,
                "Entertainment",
                "Telecom Services",
                "Internet Content & Information",
                "Banks",
            ],
        }
    )


def test_programmatic_net_rules():
    net = programmatic_net(_sector_df(), CFG.universe.discovery)
    assert net == {"UBER", "NFLX", "GOOGL"}


def test_resolve_applies_subtraction_rule():
    # UBER está no dataset e a rede o marcaria como Technology, mas ele pertence
    # ao bloco non_gics_tech -> deve SAIR do núcleo (in_core=False).
    present = {"AAPL", "UBER", "YHOO", "ZZZ"}
    uni = resolve_universe(present, _sector_df(), CFG)
    row = uni.set_index("ticker")
    assert row.loc["AAPL", "in_core"]
    assert not row.loc["UBER", "in_core"]
    assert row.loc["UBER", "sensitivity_block"] == "non_gics_tech"
    assert row.loc["YHOO", "is_delisted"]


def test_select_tickers_honors_active_blocks():
    present = {"AAPL", "UBER", "YHOO"}
    uni = resolve_universe(present, _sector_df(), CFG)
    core_only = select_tickers(uni, active_blocks=[])
    with_block = select_tickers(uni, active_blocks=["non_gics_tech"])
    assert "AAPL" in core_only and "YHOO" in core_only  # núcleo + deslistada
    assert "UBER" not in core_only  # bloco desligado
    assert "UBER" in with_block  # bloco ligado
