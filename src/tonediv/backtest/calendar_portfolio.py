"""Portfólio calendar-time com tranches sobrepostas (Jegadeesh-Titman, ADR-005).

Por que existe:
    É O ENTREGÁVEL: uma estratégia 100% especificada por regras. Eventos
    (earnings calls) são esparsos — carteiras por quintil exigindo N empresas
    no mesmo dia ficam vazias (defeito documentado de implementações
    ingênuas). A solução calendar-time: a carteira de cada dia é a média das
    H sub-carteiras ("cohorts") abertas nos últimos H pregões; cohort vazio é
    caixa (rende zero — conservador, config ``cash_return``).

Regras (todas do config; nenhuma decisão discricionária):
    1. **Gatilho**: percentil do sinal do evento contra os eventos dos últimos
       ``signal_rank_window_days`` dias ESTRITAMENTE anteriores (PIT; mínimo
       ``min_rank_history`` eventos, senão o evento fica de fora). Percentil
       alto → long; baixo → short (na variante long-short).
    2. **Pesos**: iguais dentro do cohort — ``1/(H × n_cohort)`` por nome,
       teto ``max_weight_per_name`` (excesso vira caixa, sem redistribuição:
       redistribuir concentraria exatamente onde o teto quis limitar).
    3. **Liquidez**: ADV em dólar dos últimos ``adv_window_days`` pregões,
       anexado por junção ESTRITAMENTE anterior (o ADV do próprio dia só é
       conhecido no fechamento — ADR-003).
    4. **Saída**: mecânica após H pregões (open d0 → close d_{H−1}, ADR-017).
    5. **Custos**: sobre o turnover (cohort que entra + cohort que expira).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tonediv.backtest.costs import apply_costs
from tonediv.config import StrategyConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioResult:
    """Séries diárias da estratégia + contadores para o relatório.

    Attributes:
        gross_returns: Retorno diário bruto (índice = pregões com posição ou 0).
        net_returns: Retorno líquido de custos.
        turnover: Fração do patrimônio negociada no dia.
        n_positions: Nº de posições abertas no dia.
        net_exposure: Σ side×peso do dia — exposição líquida de mercado
            (diagnóstico da variante long-short; residual ≠ 0 com cohorts
            unilaterais).
        n_events_used: Eventos efetivamente NEGOCIADOS (com preço casado).
        n_events_dropped: Eventos de entrada que não viraram posição (side 0,
            sem preço, fora da variante).
    """

    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    n_positions: pd.Series
    net_exposure: pd.Series
    n_events_used: int
    n_events_dropped: int


def assign_sides(events: pd.DataFrame, feature: str, scfg: StrategyConfig) -> pd.DataFrame:
    """Atribui o lado (+1 long / −1 short / 0 fora) a cada evento — regra 1.

    O percentil é contra a janela PIT de eventos ANTERIORES (limite direito
    estrito: eventos do mesmo dia não se veem — mesma disciplina da janela
    ``_idio``). Sem histórico mínimo, o evento fica de fora (side 0) — os
    primeiros meses da amostra viram warm-up, nunca ranking contra o futuro.

    Args:
        events: Tabela de eventos com ``decision_date`` e a feature.
        feature: Nome da coluna-sinal (``strategy.signal_feature`` na rodada
            principal; a grade de robustez varre as demais).
        scfg: Configuração da estratégia.

    Returns:
        ``events`` ordenado por ``decision_date`` com colunas ``side`` e
        ``signal_pct`` (NaN quando sem histórico).
    """
    out = events.dropna(subset=["decision_date", feature]).sort_values("decision_date")
    out = out.reset_index(drop=True)
    dates = out["decision_date"].to_numpy(dtype="datetime64[ns]")
    # signal_direction=-1 inverte a aposta (percentil ALTO vira short): usado
    # para VENDER o excesso de otimismo da gestão (ADR-022). É decisão de config
    # documentada e fundamentada na tese, não um flip pós-hoc de número.
    values = float(scfg.signal_direction) * out[feature].to_numpy(dtype=np.float64)
    window = np.timedelta64(scfg.signal_rank_window_days, "D")

    lo = np.searchsorted(dates, dates - window, side="left")
    hi = np.searchsorted(dates, dates, side="left")  # ESTRITO: só eventos passados

    pct = np.full(len(out), np.nan)
    for i in range(len(out)):
        past = values[lo[i] : hi[i]]
        if len(past) >= scfg.min_rank_history:
            # MIDRANK nos empates: (menores + metade dos iguais) / n. Com a
            # regra ingênua (past <= v), qualquer empate com o máximo dava
            # percentil 1.0 e um sinal CONSTANTE virava carteira 100% comprada
            # (auditoria da Fase 4). Com midrank, sinal constante fica em 0.5
            # -> side 0 (nenhuma informação, nenhuma posição).
            less = float(np.sum(past < values[i]))
            equal = float(np.sum(past == values[i]))
            pct[i] = (less + 0.5 * equal) / len(past)

    side = np.zeros(len(out), dtype=np.int64)
    side[pct >= 1.0 - scfg.top_fraction] = 1
    side[pct <= scfg.top_fraction] = -1
    out["signal_pct"] = pct
    out["side"] = np.where(np.isnan(pct), 0, side)
    return out


def liquidity_filter(
    events: pd.DataFrame, prices: pd.DataFrame, scfg: StrategyConfig
) -> pd.DataFrame:
    """Remove eventos abaixo do piso de liquidez (regra 3) — point-in-time.

    O ADV (média rolante de ``dollar_volume``) é datado no próprio dia, mas a
    junção usa :func:`~tonediv.align.pit.merge_asof_backward` (estritamente
    anterior): a decisão no open de ``decision_date`` enxerga o ADV até o
    fechamento da VÉSPERA. Eventos sem ADV disponível são descartados e
    contados (ativo ilíquido ou recém-listado — capacidade importa).
    """
    from tonediv.align.pit import merge_asof_backward

    adv = prices[["ticker", "date", "dollar_volume"]].sort_values(["ticker", "date"]).copy()
    # min_periods = janela CHEIA: "ADV de 60 pregões" com 1 pregão de história
    # não é ADV — recém-listados esperam a janela completar (conservador; a
    # intenção do filtro é capacidade real, auditoria da Fase 4). Limitação
    # documentada: dollar_volume usa close AJUSTADO (proventos futuros alteram
    # levemente o nível passado); afeta só a vizinhança do piso do filtro.
    adv["adv_usd"] = adv.groupby("ticker", sort=False)["dollar_volume"].transform(
        lambda s: s.rolling(
            scfg.liquidity_adv_window_days, min_periods=scfg.liquidity_adv_window_days
        ).mean()
    )
    panel = adv.rename(columns={"date": "decision_date"})[["ticker", "decision_date", "adv_usd"]]
    merged = merge_asof_backward(events, panel, on="decision_date", by="ticker", cols=["adv_usd"])
    keep = merged["adv_usd"] >= scfg.liquidity_min_adv_usd
    logger.info(
        "Liquidez: %d de %d eventos acima do piso (US$ %.0f).",
        int(keep.sum()),
        len(merged),
        scfg.liquidity_min_adv_usd,
    )
    return merged[keep.fillna(False)].reset_index(drop=True)


def _tranche_frame(
    events: pd.DataFrame, prices: pd.DataFrame, scfg: StrategyConfig
) -> tuple[pd.DataFrame, int, int]:
    """Explode cada evento com side ≠ 0 em linhas (data, retorno, peso) — regras 2/4.

    Retorno do dia 0 é open→close (entrada no open); dias seguintes são
    close→close. O peso ``1/(H × n_cohort_do_lado)`` com teto é constante ao
    longo da vida da tranche (rebalanceio intra-tranche não existe — regra
    mecânica). Duas correções da auditoria da Fase 4:

    - o tamanho do cohort é contado APÓS casar evento↔preço (evento sem série
      não pode deflacionar o peso dos que negociam de fato);
    - ``exit_notional`` carrega o peso × crescimento acumulado da tranche —
      o desmonte é negociado ao notional CORRENTE, não ao de entrada.

    Returns:
        Tupla ``(frame, n_negociados, n_pulados_sem_preco)``.
    """
    sided = events[events["side"] != 0]
    by_ticker = {
        t: (g["date"].to_numpy(dtype="datetime64[ns]"), g["open"].to_numpy(), g["close"].to_numpy())
        for t, g in prices.sort_values("date").groupby("ticker", sort=False)
    }

    # Passo 1: resolver posição de cada evento no painel; descartar (contando)
    # os sem série/data — ANTES de calcular cohorts.
    resolved = []
    for _, ev in sided.iterrows():
        series = by_ticker.get(ev["ticker"])
        if series is None:
            resolved.append(-1)
            continue
        dates = series[0]
        pos = int(np.searchsorted(dates, np.datetime64(ev["decision_date"])))
        resolved.append(
            pos if pos < len(dates) and dates[pos] == np.datetime64(ev["decision_date"]) else -1
        )
    sided = sided.assign(_pos=resolved)
    n_skipped = int((sided["_pos"] < 0).sum())
    tradeable = sided[sided["_pos"] >= 0]

    cohort_n = tradeable.groupby(["decision_date", "side"])["ticker"].transform("size")
    weight = np.minimum(1.0 / (scfg.holding_days * cohort_n), scfg.max_weight_per_name)

    rows: list[dict[str, object]] = []
    for (_, ev), w in zip(tradeable.iterrows(), weight, strict=True):
        dates, opens, closes = by_ticker[ev["ticker"]]
        pos = int(ev["_pos"])
        last = min(pos + scfg.holding_days - 1, len(dates) - 1)
        growth = 1.0
        for k, day in enumerate(range(pos, last + 1)):
            if day == pos:
                ret = closes[day] / opens[day] - 1.0 if opens[day] > 0 else np.nan
            else:
                ret = closes[day] / closes[day - 1] - 1.0 if closes[day - 1] > 0 else np.nan
            if np.isfinite(ret):
                growth *= 1.0 + ret
            rows.append(
                {
                    "date": dates[day],
                    "side": int(ev["side"]),
                    "w": float(w),
                    "ret": float(ret) if np.isfinite(ret) else np.nan,
                    "is_entry": k == 0,
                    "is_exit": day == last,
                    "exit_notional": float(w * growth) if day == last else 0.0,
                }
            )
    return pd.DataFrame(rows), int(len(tradeable)), n_skipped


def run_portfolio(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    scfg: StrategyConfig,
    bps_per_side: float,
    variant: str,
) -> PortfolioResult:
    """Roda a estratégia calendar-time numa variante (regras 1–5 compostas).

    Args:
        events: Eventos JÁ com ``side`` (:func:`assign_sides`) e filtro de
            liquidez aplicado.
        prices: Painel long de preços (o calendário de pregões vem DELE — não
            de ``freq='B'``, que fabricaria feriados como dias de retorno 0 e
            diluiria Sharpe/t-NW; auditoria da Fase 4).
        scfg: Configuração da estratégia.
        bps_per_side: Custo por perna (config ``costs``).
        variant: ``"long_short"`` (com exposição líquida RESIDUAL — cohorts
            esparsos podem ser unilaterais; a série ``net_exposure`` reporta o
            diagnóstico) ou ``"long_only"``.

    Returns:
        :class:`PortfolioResult` com as séries diárias.
    """
    work = events.copy()
    if variant == "long_only":
        work = work[work["side"] >= 0]

    frame, n_traded, n_skipped = _tranche_frame(work, prices, scfg)
    n_dropped = int(len(events) - n_traded)  # side 0 + sem preço + fora da variante
    if frame.empty:
        empty = pd.Series(dtype=np.float64)
        return PortfolioResult(empty, empty, empty, empty, empty, 0, n_dropped)

    frame["signed_wret"] = frame["side"] * frame["w"] * frame["ret"]
    frame["signed_w"] = frame["side"] * frame["w"]
    daily_gross = frame.groupby("date")["signed_wret"].sum()
    daily_n = frame.groupby("date")["w"].size()
    # Exposição líquida (Σ side×w): diagnóstico central da variante long-short
    # — cohorts unilaterais tornam o livro direcional; o valor é reportado e a
    # série hedgeada é construída no script 06 (auditoria da Fase 4).
    daily_net_exp = frame.groupby("date")["signed_w"].sum()
    # Turnover: entradas ao notional de ENTRADA; saídas ao notional CORRENTE
    # (w × crescimento acumulado). Ambas as pernas pagam custo.
    entered = frame.loc[frame["is_entry"], :].groupby("date")["w"].sum()
    exited = frame.loc[frame["is_exit"], :].groupby("date")["exit_notional"].sum()
    daily_turnover = entered.add(exited, fill_value=0.0)

    # Calendário REAL de pregões dentro do intervalo da estratégia.
    all_days = np.unique(prices["date"].to_numpy(dtype="datetime64[ns]"))
    lo = np.searchsorted(all_days, daily_gross.index.min().to_datetime64())
    hi = np.searchsorted(all_days, daily_gross.index.max().to_datetime64(), side="right")
    full_index = pd.DatetimeIndex(all_days[lo:hi])

    gross = daily_gross.reindex(full_index).fillna(0.0)  # dias sem posição = caixa (0)
    turnover = daily_turnover.reindex(full_index).fillna(0.0)
    n_pos = daily_n.reindex(full_index).fillna(0).astype(int)
    net_exposure = daily_net_exp.reindex(full_index).fillna(0.0)
    net = apply_costs(gross, turnover, bps_per_side)

    logger.info(
        "Portfólio %s: %d eventos negociados (%d sem preço), %d pregões, turnover médio %.4f.",
        variant,
        n_traded,
        n_skipped,
        len(gross),
        float(turnover.mean()),
    )
    return PortfolioResult(gross, net, turnover, n_pos, net_exposure, n_traded, n_dropped)


def build_strategy(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    scfg: StrategyConfig,
    bps_per_side: float,
    feature: str,
    variant: str,
) -> PortfolioResult:
    """Compõe liquidez → lados → portfólio (orquestrador único para os scripts).

    Evita que 06/08/09 dupliquem a mesma sequência (e divirjam). Orquestrador
    PURO — nenhum I/O; o chamador carrega/salva.
    """
    liquid = liquidity_filter(events.dropna(subset=[feature]), prices, scfg)
    sided = assign_sides(liquid, feature, scfg)
    return run_portfolio(sided, prices, scfg, bps_per_side, variant)
