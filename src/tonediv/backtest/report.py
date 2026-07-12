"""Curvas de capital, benchmarks buy-and-hold e tearsheet (script 09).

Por que existe:
    O entregável que a banca vê é a "lâmina" da estratégia: curva de capital
    das duas variantes contra QQQ e SPY, tabela de métricas líquidas de custos,
    drawdown e turnover. Este módulo isola a lógica pura (curvas, retornos de
    benchmark, alinhamento) do I/O de plotagem, que fica numa única função
    explícita (padrão de ``diagnostics.save_hour_histogram``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def equity_curve(returns: pd.Series) -> pd.Series:
    """Curva de capital composta a partir de retornos diários, base 1.0.

    ``(1+r).cumprod()`` já É o capital acumulado a partir de 1.0 investido
    (inclui o retorno do dia 0): o último ponto é o capital final relativo à
    base 1.0. Dias sem retorno viram 0 (caixa) — a curva anda de lado, nunca
    quebra. Nota: o primeiro ponto plotado é ``1+r₀`` (o dia 0 já ocorreu), não
    exatamente 1.0 — convenção padrão de equity curve; o retorno total/CAGR
    ancoram na base 1.0, não no primeiro ponto (:func:`relative_capital_table`).
    """
    return (1.0 + returns.fillna(0.0)).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Série de drawdown (fração ≤ 0) da curva de capital."""
    curve = equity_curve(returns)
    return curve / curve.cummax() - 1.0


def benchmark_buy_hold(
    prices: pd.DataFrame, tickers: tuple[str, ...], index: pd.DatetimeIndex
) -> dict[str, pd.Series]:
    """Retornos diários buy-and-hold de cada benchmark, alinhados a ``index``.

    Usa ``ret_cc`` (retorno close-to-close ajustado) do painel, reindexado ao
    calendário da estratégia (dias sem cotação viram 0). A comparação é
    "comprar e segurar" — sem custos, o pior caso PARA a estratégia (que paga
    custos e ainda assim precisa vencer).

    Args:
        prices: Painel long com ``ticker, date, ret_cc``.
        tickers: Benchmarks (ex.: ``("QQQ", "SPY")``).
        index: Calendário-alvo (o da série da estratégia).

    Returns:
        Dicionário ``{ticker: série de retornos diários}``. Benchmarks AUSENTES
        do painel são OMITIDOS com um warning — jamais viram uma linha achatada
        de zeros silenciosa no tearsheet (um benchmark faltante deve ser
        visível, não disfarçado; risco real dado o histórico do yfinance).
    """
    out: dict[str, pd.Series] = {}
    for ticker in tickers:
        rows = prices.loc[prices["ticker"] == ticker, ["date", "ret_cc"]]
        if rows.empty:
            logger.warning(
                "Benchmark %s ausente do painel de preços — OMITIDO do relatório.", ticker
            )
            continue
        out[ticker] = rows.set_index("date")["ret_cc"].sort_index().reindex(index).fillna(0.0)
    return out


def relative_capital_table(curves: dict[str, pd.Series]) -> pd.DataFrame:
    """Retorno total e CAGR de cada curva (para a tabela do relatório)."""
    rows = []
    for name, curve in curves.items():
        if curve.empty:
            continue
        years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
        # Base 1.0 (o capital final relativo à base), NÃO curve.iloc[0]=1+r₀:
        # dividir por iloc[0] descartaria o retorno do dia 0 (auditoria Fase 5).
        final = float(curve.iloc[-1])
        rows.append(
            {"serie": name, "retorno_total": final - 1.0, "cagr": final ** (1.0 / years) - 1.0}
        )
    return pd.DataFrame(rows)


def save_tearsheet(
    curves: dict[str, pd.Series], drawdown: pd.Series, title: str, out_path: Path
) -> None:
    """Salva o tearsheet: curvas de capital (log) + drawdown da estratégia.

    Função de I/O explícita (backend Agg para ambiente headless/CI).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1], sharex=True)
    for name, curve in curves.items():
        if not curve.empty:
            ax1.plot(curve.index, curve.to_numpy(), label=name, linewidth=1.3)
    ax1.set_yscale("log")
    ax1.set_ylabel("Capital (base 1.0, log)")
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(drawdown.index, drawdown.to_numpy(), 0.0, color="#C44E52", alpha=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    logger.info("Tearsheet salvo em %s", out_path)


def excess_over_benchmark(
    strategy: pd.Series, benchmark: pd.Series, periods_per_year: int
) -> dict[str, float]:
    """Retorno excedente anualizado e information ratio vs. um benchmark.

    Alinha as duas séries pelo índice comum e mede o excesso diário; o IR usa
    o desvio do excesso (tracking error). Base da comparação da variante
    long-only "vs. benchmark" exigida pelo ADR-005.
    """
    df = pd.DataFrame({"s": strategy, "b": benchmark}).dropna()
    if len(df) < 2:
        return {"excess_annual": float("nan"), "information_ratio": float("nan")}
    excess = df["s"] - df["b"]
    te = float(excess.std(ddof=1))
    # nunique()==1 pega excesso constante cujo std sai ~1e-18 por erro de ponto
    # flutuante (não zero exato) — sem o guard, o IR explodiria (mesma classe
    # do guard de sharpe_ratio).
    ir = (
        float(excess.mean() / te * np.sqrt(periods_per_year))
        if te > 0 and excess.nunique() > 1
        else float("nan")
    )
    return {"excess_annual": float(excess.mean() * periods_per_year), "information_ratio": ir}
