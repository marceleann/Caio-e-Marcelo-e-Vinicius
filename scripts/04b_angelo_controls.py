"""04b — Controles do Angelo + sinal LIMPO (residualização estritamente-passada).

Lê ``events.parquet`` (script 04), as falas pontuadas (com orador) e os preços;
anexa os controles que a distância de tom precisa para ser lida corretamente
(nível de tom, tom dos analistas, distância de tom dos analistas, tom do setor,
distância de tom defasada, comprimento, momento, reversão, tamanho) e grava a
coluna ``tone_distance_clean`` — o RESÍDUO estritamente-passado da distância de
tom sobre os confundidores contemporâneos (ADR-023). Sobrescreve events.parquet.

A distância de tom crua é confundida (ver Angelo, Tabela 2): o efeito só aparece
depois de remover esses fatores. O sinal operável é o resíduo, não o valor cru.

Uso:
    python scripts/04b_angelo_controls.py --config config.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tonediv.config import configure_logging, load_config
from tonediv.nlp import features as ft


def _weighted_net(sub: pd.DataFrame) -> float:
    w = sub["n_tokens"].to_numpy(dtype=np.float64)
    if w.sum() <= 0:
        return float("nan")
    return float((sub["net_tone"].to_numpy(dtype=np.float64) * w).sum() / w.sum())


def _market_controls(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Momento (12m-1m), reversão (1m) e tamanho (liquidez), estritamente antes da decisão."""
    by_ticker: dict[str, tuple] = {}
    for tkr, g in prices.sort_values("date").groupby("ticker", sort=False):
        by_ticker[tkr] = (
            g["date"].to_numpy(dtype="datetime64[ns]"),
            g["close"].to_numpy(dtype=np.float64),
            g["dollar_volume"].to_numpy(dtype=np.float64),
        )
    mom, rev, size = [], [], []
    for tkr, dec in zip(events["ticker"], events["decision_date"]):
        rec = by_ticker.get(tkr)
        if rec is None or pd.isna(dec):
            mom.append(np.nan); rev.append(np.nan); size.append(np.nan); continue
        dates, close, dv = rec
        i = int(np.searchsorted(dates, np.datetime64(dec), side="left"))  # pregões ESTRITAMENTE < decisão
        mom.append(close[i - 21] / close[i - 252] - 1.0 if i >= 252 else np.nan)
        rev.append(close[i - 1] / close[i - 21] - 1.0 if i >= 21 else np.nan)
        size.append(float(np.log(np.median(dv[i - 60 : i]))) if i >= 60 and np.median(dv[i - 60 : i]) > 0 else np.nan)
    return pd.DataFrame({"momentum": mom, "reversal": rev, "size_proxy": size}, index=events.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Controles do Angelo + sinal limpo.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.logging)
    p = cfg.paths

    events = pd.read_parquet(p.data_processed / "events.parquet")
    scores = pd.read_parquet(p.data_interim / "utterance_scores.parquet")
    roles = pd.read_parquet(p.data_interim / "utterances_roles.parquet")
    prices = pd.read_parquet(p.data_raw / "prices.parquet")
    scores = scores.merge(
        roles[["call_id", "utterance_idx", "speaker"]], on=["call_id", "utterance_idx"], how="left"
    )

    # Controles de TOM (das falas pontuadas)
    events["disclosure_tone"] = events["net_tone"]
    an = scores[scores["role"] == "analyst"]
    analyst_tone = an.groupby("call_id").apply(_weighted_net, include_groups=False).rename("analyst_tone")
    events = events.merge(analyst_tone, on="call_id", how="left")
    events = events.merge(
        ft.tone_distance_per_call(scores, cfg.features, role="analyst",
                                  out_col="analyst_tone_distance", count_col="n_analysts_td"),
        on="call_id", how="left",
    )
    length = scores.groupby("call_id")["n_tokens"].sum().apply(np.log1p).rename("length")
    events = events.merge(length, on="call_id", how="left")

    # year_quarter, tom do setor e distância de tom defasada (estritamente passada)
    ts_local = events["call_datetime"].dt.tz_convert(cfg.sample.timezone)
    events["year_quarter"] = ts_local.dt.year.astype(str) + "Q" + ts_local.dt.quarter.astype(str)
    # (industry_tone removido: no universo tech, é constante por trimestre e
    # perfeitamente absorvido pelo efeito fixo de trimestre — além de a média do
    # trimestre inteiro olhar o futuro. Ver auditoria/ADR-025.)
    events = events.sort_values(["ticker", "call_datetime"]).reset_index(drop=True)
    events["lagged_tone_distance"] = events.groupby("ticker")["tone_distance"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # Controles de MERCADO (point-in-time dos preços)
    events = events.join(_market_controls(events, prices))

    # SINAL LIMPO: resíduo estritamente-passado da distância de tom sobre os confundidores
    events["tone_distance_clean"] = ft.residualize_pit(
        events, "tone_distance",
        list(cfg.features.clean_signal_confounders), cfg.features.clean_signal_min_history,
    )

    events.to_parquet(p.data_processed / "events.parquet", index=False)
    clean_ok = events["tone_distance_clean"].notna().mean()
    print(
        f"[04b] controles anexados | tone_distance_clean disponível={clean_ok:.1%} "
        f"({int(events['tone_distance_clean'].notna().sum())} calls) | "
        f"confundidores={list(cfg.features.clean_signal_confounders)}"
    )


if __name__ == "__main__":
    main()
