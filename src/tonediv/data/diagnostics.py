"""Diagnósticos da base: horários, cobertura, qualidade e survivorship (script 00).

Por que existe:
    Antes de confiar no dataset, precisamos MEDIR sua realidade — distribuição
    de horários das calls (que fundamenta a regra T+1, ADR-002/008), cobertura
    temporal, quantas transcrições caem nos filtros de qualidade, e o TESTE DE
    SURVIVORSHIP (empresas tech deslistadas aparecem nos anos em que existiam?).
    As funções são puras e produzem DataFrames/relatório; o I/O (ler parquet,
    salvar png/markdown) fica no script 00, exceto o salvamento do histograma,
    que é uma função de I/O explícita e isolada aqui.

Referência: ADR-001 (ausências do universo quantificadas), ADR-002/008 (horários).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import pandas as pd

from tonediv.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Diagnostics:
    """Coletânea de tabelas de diagnóstico produzidas por :func:`compute`."""

    hour_dist: pd.DataFrame
    session_shares: pd.DataFrame
    coverage_year: pd.DataFrame
    coverage_ticker: pd.DataFrame
    survivorship: pd.DataFrame
    universe_layers: pd.DataFrame
    quality: pd.DataFrame


def _parse_hhmm(value: str) -> time:
    """Converte ``"HH:MM"`` do config em :class:`datetime.time`."""
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


def hour_distribution(calls_all: pd.DataFrame) -> pd.DataFrame:
    """Conta calls por hora do dia (apenas as com horário real presente).

    Returns:
        DataFrame ``[hour, n_calls]`` cobrindo 0–23. Calls sem horário
        (``has_time=False``) são excluídas aqui e contabilizadas à parte em
        :func:`session_shares`.
    """
    has_time = calls_all["has_time"].fillna(False)
    hours = calls_all.loc[has_time, "call_datetime"].dt.hour
    counts = hours.value_counts().reindex(range(24), fill_value=0).sort_index()
    return counts.rename_axis("hour").reset_index(name="n_calls")


def session_shares(calls_all: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Reparte as calls entre após-fechamento / pré-abertura / pregão / ausente.

    É a evidência empírica que justifica o fallback conservador T+1 quando o
    horário é ausente/suspeito (ADR-008): o relatório mostra QUANTAS calls caem
    nesse balde.

    Returns:
        DataFrame ``[bucket, n_calls, share]``.
    """
    pit = cfg.raw["pit"]
    close, open_ = _parse_hhmm(pit["market_close_et"]), _parse_hhmm(pit["market_open_et"])
    has = calls_all["has_time"].fillna(False) & calls_all["call_datetime"].notna()
    t = calls_all["call_datetime"].dt.time
    after = has & (t > close)
    before = has & (t < open_)
    during = has & ~after & ~before
    missing = ~has
    n = max(len(calls_all), 1)
    buckets = {
        "apos_fechamento (T+1 open)": int(after.sum()),
        "pre_abertura (mesmo dia open)": int(before.sum()),
        "durante_pregao (T+1 open)": int(during.sum()),
        "ausente_ou_suspeito (T+1 open)": int(missing.sum()),
    }
    df = pd.DataFrame({"bucket": list(buckets), "n_calls": list(buckets.values())})
    df["share"] = (df["n_calls"] / n).round(4)
    return df


def coverage_by_year(calls_all: pd.DataFrame) -> pd.DataFrame:
    """Cobertura temporal do dataset inteiro: calls e empresas distintas por ano."""
    g = calls_all.groupby("year")
    out = g.agg(n_calls=("call_id", "size"), n_tickers=("ticker", "nunique"))
    return out.reset_index().sort_values("year")


def coverage_by_ticker(calls_universe: pd.DataFrame) -> pd.DataFrame:
    """Cobertura por empresa no universo: nº de calls e primeiro/último ano."""
    g = calls_universe.groupby("ticker")
    out = g.agg(
        n_calls=("call_id", "size"),
        first_year=("year", "min"),
        last_year=("year", "max"),
    )
    return out.reset_index().sort_values("n_calls", ascending=False)


def survivorship_report(calls_all: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """TESTE DE SURVIVORSHIP: deslistadas tech aparecem quando existiam?

    Procura o ticker LITERAL (``ticker_raw``, pré-alias) de cada nome da lista
    ``delisted_check`` no dataset completo. A presença de Sun (SUNW/JAVA), Yahoo
    (YHOO), LinkedIn (LNKD) etc. nos anos corretos é o sinal de que a base NÃO
    sofre de viés de sobrevivência; a ausência é reportada honestamente.

    Returns:
        DataFrame ``[ticker, present, n_calls, first_year, last_year]``.
    """
    raw = calls_all["ticker_raw"] if "ticker_raw" in calls_all.columns else calls_all["ticker"]
    rows = []
    for ticker in cfg.universe.delisted_check:
        sub = calls_all[raw == ticker]
        present = len(sub) > 0
        rows.append(
            {
                "ticker": ticker,
                "present": present,
                "n_calls": int(len(sub)),
                "first_year": int(sub["year"].min()) if present else pd.NA,
                "last_year": int(sub["year"].max()) if present else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def universe_layer_summary(universe_df: pd.DataFrame) -> pd.DataFrame:
    """Resume o universo por camada: candidatos, presentes no dataset e com preço.

    Quantifica as AUSÊNCIAS do universo exigidas pelo ADR-001 (candidatos que a
    lista prevê mas que não existem no dataset e/ou no yfinance).
    """
    in_prices = universe_df["in_prices"] if "in_prices" in universe_df.columns else False
    tmp = universe_df.assign(in_prices=in_prices)
    g = tmp.groupby("layer")
    out = g.agg(
        n_candidates=("ticker", "size"),
        n_in_dataset=("in_dataset", "sum"),
        n_in_prices=("in_prices", "sum"),
    )
    return out.reset_index().sort_values("layer")


def quality_summary(
    calls_all: pd.DataFrame, calls_universe: pd.DataFrame, utterances: pd.DataFrame
) -> pd.DataFrame:
    """Resumo numérico de volume e qualidade (para o topo do relatório)."""
    metrics = {
        "calls_no_dataset": len(calls_all),
        "calls_no_universo_qualidade_ok": len(calls_universe),
        "falas_no_universo": len(utterances),
        "mediana_falas_por_call": (
            int(calls_universe["n_utterances"].median()) if len(calls_universe) else 0
        ),
        "empresas_no_universo": calls_universe["ticker"].nunique(),
    }
    return pd.DataFrame({"metrica": list(metrics), "valor": list(metrics.values())})


def compute(
    cfg: Config,
    calls_all: pd.DataFrame,
    universe_df: pd.DataFrame,
    calls_universe: pd.DataFrame,
    utterances: pd.DataFrame,
) -> Diagnostics:
    """Calcula todas as tabelas de diagnóstico (orquestrador puro para o script 00)."""
    return Diagnostics(
        hour_dist=hour_distribution(calls_all),
        session_shares=session_shares(calls_all, cfg),
        coverage_year=coverage_by_year(calls_all),
        coverage_ticker=coverage_by_ticker(calls_universe),
        survivorship=survivorship_report(calls_all, cfg),
        universe_layers=universe_layer_summary(universe_df),
        quality=quality_summary(calls_all, calls_universe, utterances),
    )


def _df_to_md(df: pd.DataFrame) -> str:
    """Renderiza um DataFrame como tabela Markdown (sem depender de ``tabulate``)."""
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    lines = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *lines])


def render_quality_report(diag: Diagnostics, hist_rel_path: str) -> str:
    """Monta o ``quality_report.md`` a partir das tabelas de diagnóstico.

    Args:
        diag: Saída de :func:`compute`.
        hist_rel_path: Caminho relativo do PNG do histograma de horários.

    Returns:
        Conteúdo Markdown completo do relatório (o script 00 grava em disco).
    """
    n_missing = int(diag.survivorship["present"].eq(False).sum())
    absent_universe = int(
        diag.universe_layers["n_candidates"].sum() - diag.universe_layers["n_in_dataset"].sum()
    )
    parts = [
        "# quality_report.md — Diagnóstico da base (gerado pelo script 00)",
        "",
        "> Gerado automaticamente. Reflete a base efetivamente baixada; reexecutar",
        "> `make download && make diagnostics` regenera este arquivo.",
        "",
        "## 1. Volume e qualidade",
        _df_to_md(diag.quality),
        "",
        "## 2. Cobertura temporal (dataset completo)",
        _df_to_md(diag.coverage_year),
        "",
        "## 3. Distribuição de horários das calls",
        f"![histograma de horários]({hist_rel_path})",
        "",
        "Repartição por janela de decisão (justifica o fallback T+1, ADR-002/008):",
        "",
        _df_to_md(diag.session_shares),
        "",
        "## 4. Universo por camada (ausências = candidatos ausentes do dataset/preço)",
        f"Candidatos ausentes do dataset: **{absent_universe}**.",
        "",
        _df_to_md(diag.universe_layers),
        "",
        "## 5. Teste de survivorship (deslistadas tech)",
        f"Nomes da lista ausentes do dataset: **{n_missing}** de {len(diag.survivorship)}.",
        "",
        _df_to_md(diag.survivorship),
        "",
        "## 6. Papéis analista×gestão",
        "Cobertura da heurística e amostra de validação manual: ver "
        "[roles_report.md](roles_report.md) (gerado pelo script 02).",
        "",
    ]
    return "\n".join(parts)


def save_hour_histogram(hour_dist: pd.DataFrame, out_path: Path) -> None:
    """Salva o histograma de horários como PNG (função de I/O explícita).

    Usa backend não-interativo (``Agg``) para rodar em ambiente headless (CI).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(hour_dist["hour"], hour_dist["n_calls"], color="#4C72B0")
    ax.set_xlabel("Hora do dia (US/Eastern)")
    ax.set_ylabel("Nº de calls")
    ax.set_title("Distribuição de horários das earnings calls (com horário presente)")
    ax.set_xticks(range(0, 24))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    logger.info("Histograma de horários salvo em %s", out_path)
