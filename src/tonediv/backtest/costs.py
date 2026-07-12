"""Custos de transação: do retorno bruto ao líquido via turnover (ADR-011).

Por que existe:
    O entregável do desafio é uma estratégia LÍQUIDA de custos. O modelo aqui
    é deliberadamente simples e conservador: custo proporcional por perna
    (``bps_per_side`` do config, default 5 bps — mercado americano), aplicado
    sobre o turnover DIÁRIO da carteira. Sem custos fixos, sem impacto de
    mercado — limitações declaradas na lâmina da estratégia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_costs(gross_returns: pd.Series, turnover: pd.Series, bps_per_side: float) -> pd.Series:
    """Converte retornos brutos em líquidos descontando custo sobre o turnover.

    ``ret_net_t = ret_gross_t − turnover_t × (bps/10.000)``. O turnover já é a
    fração do patrimônio NEGOCIADA no dia (compras + vendas, cada perna paga) —
    quem o calcula é o portfólio; aqui só se aplica o preço por unidade.

    Args:
        gross_returns: Série diária de retornos brutos (índice = datas).
        turnover: Série diária de turnover (mesmo índice; fração negociada).
        bps_per_side: Custo por perna em basis points (config ``costs``).

    Returns:
        Série de retornos líquidos, alinhada ao índice de entrada.
    """
    aligned = turnover.reindex(gross_returns.index).fillna(0.0)
    cost = aligned * (bps_per_side / 10_000.0)
    return gross_returns - cost


def total_cost_drag(turnover: pd.Series, bps_per_side: float, periods_per_year: int) -> float:
    """Arrasto anualizado de custos (informativo para a lâmina/tearsheet)."""
    if len(turnover) == 0:
        return 0.0
    daily_drag = float(np.nanmean(turnover)) * (bps_per_side / 10_000.0)
    return daily_drag * periods_per_year
