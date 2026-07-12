"""Alinhamento point-in-time texto→preço: decisão T+1, retornos e guarda forte.

Por que existe:
    É a fronteira entre "sinal com informação" e "sinal com vazamento". Três
    responsabilidades, todas inegociáveis (ADR-002/003):

    1. **Timestamp de decisão**: converte o instante da call na PRIMEIRA
       abertura de pregão em que a informação era negociável — call após as
       16:00 ET decide no open do pregão seguinte; antes das 09:30 ET, no open
       do mesmo dia; durante o pregão ou horário ausente/suspeito, fallback
       CONSERVADOR no open seguinte.
    2. **Retornos futuros**: entrada no OPEN do pregão de decisão, saída no
       CLOSE após H pregões DE POSIÇÃO (open de d0 → close de d_{H−1}; a
       posição atravessa exatamente H sessões). Nunca o close do dia da call.
    3. **Guarda `assert_no_lookahead`**: compara TIMESTAMPS COMPLETOS — para
       call às 19h, uma decisão no open do MESMO dia (09:30 < 19:00) DISPARA
       erro. Guardas que normalizam para data deixam esse caso passar
       (armadilha nº 2 do protótipo anterior; há teste que prova a captura).

    A junção de informação de mercado à call usa SEMPRE
    :func:`merge_asof_backward` — nunca ``direction='forward'``, que casaria
    a call com dados posteriores à decisão.
"""

from __future__ import annotations

import logging
from datetime import time, timedelta

import numpy as np
import pandas as pd

from tonediv.config import PitConfig

logger = logging.getLogger(__name__)


class LookaheadError(AssertionError):
    """Violação de point-in-time: decisão não é estritamente posterior à call."""


def _parse_hhmm(value: str) -> time:
    """Converte ``"HH:MM"`` do config em :class:`datetime.time`."""
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


def decision_intent(calls: pd.DataFrame, pcfg: PitConfig) -> pd.DataFrame:
    """Determina a DATA-ALVO da decisão e a regra aplicada, por call.

    Regras (ADR-002, revisadas na auditoria da Fase 3):
        - ``same_open``: call com horário confiável iniciada ANTES de
          ``open − assumed_call_duration``: as features usam a transcrição
          COMPLETA e o Q&A (fonte da tese) acontece no FIM da call — uma call
          iniciada às 09:00 ainda está acontecendo às 09:30, então "antes da
          abertura" precisa significar "TERMINA antes da abertura" (12,4% das
          calls reais começam 08:30–09:29 e migram para T+1 com o buffer).
        - ``next_open``: todo o resto — pós-fechamento, durante o pregão,
          horário ausente/suspeito (conservador).
        - ``no_timestamp``: ``call_datetime`` nulo — sem instante não há
          decisão; ``intended_date`` fica NaT e o evento é contabilizado (e
          excluído) a jusante, nunca descartado em silêncio.

    A data-alvo é CALENDÁRIO; o casamento com pregões reais (fins de semana,
    feriados, halts) acontece em :func:`align_decisions`.

    Args:
        calls: Tabela com ``call_datetime`` (tz-aware ET) e ``has_time``.
        pcfg: Configuração PIT.

    Returns:
        ``calls`` com colunas novas ``intended_date`` e ``decision_rule``.
    """
    open_t = _parse_hhmm(pcfg.market_open_et)
    anchor = pd.Timestamp("2000-01-01").replace(hour=open_t.hour, minute=open_t.minute)
    cutoff_t = (anchor - timedelta(minutes=pcfg.assumed_call_duration_minutes)).time()
    suspicious = set(pcfg.suspicious_times)

    dt = calls["call_datetime"]
    notna = dt.notna()
    has_time = calls["has_time"].fillna(False).astype(bool) & notna
    time_str = dt.dt.strftime("%H:%M:%S")
    trustworthy = has_time & ~time_str.isin(suspicious)

    # Comparação de hora só nas linhas com timestamp (NaT quebraria o `<`).
    before_cutoff = pd.Series(False, index=calls.index)
    before_cutoff[notna] = pd.Series(dt[notna].dt.time, index=dt[notna].index) < cutoff_t
    same_day = (trustworthy & before_cutoff).to_numpy()

    out = calls.copy()
    base_date = dt.dt.tz_localize(None).dt.normalize()
    out["intended_date"] = pd.to_datetime(
        np.where(same_day, base_date, base_date + timedelta(days=1))
    )
    out["decision_rule"] = np.select(
        [~notna.to_numpy(), same_day], ["no_timestamp", "same_open"], default="next_open"
    )
    return out


def align_decisions(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Casa a data-alvo com o PRIMEIRO pregão negociável ``>=`` alvo, por ticker.

    Usa as datas do PRÓPRIO ticker no painel de preços (não um calendário
    genérico): se o papel não negociou no dia-alvo (feriado, halt, listagem
    tardia), a decisão desliza para o primeiro pregão em que ELE negociou.
    Calls sem pregão disponível (fim da amostra / sem preço) ficam ``NaT`` e
    são contabilizadas pelo chamador — nunca descartadas em silêncio.

    Args:
        events: Saída de :func:`decision_intent` com ``ticker``.
        prices: Painel long de preços (``ticker``, ``date``).

    Returns:
        ``events`` com coluna nova ``decision_date`` (NaT se inalcançável).
    """
    out = events.copy()
    out["decision_date"] = pd.NaT
    sorted_prices = prices.sort_values("date")
    dates_by_ticker = {
        t: g["date"].to_numpy() for t, g in sorted_prices.groupby("ticker", sort=False)
    }
    for ticker, idx in out.groupby("ticker", sort=False).groups.items():
        dates = dates_by_ticker.get(ticker)
        if dates is None:
            continue  # sem preço para o ticker: decision_date fica NaT
        targets = out.loc[idx, "intended_date"].to_numpy(dtype="datetime64[ns]")
        pos = np.searchsorted(dates, targets, side="left")
        found = pos < len(dates)
        vals = np.full(len(idx), np.datetime64("NaT"), dtype="datetime64[ns]")
        vals[found] = dates[pos[found]]
        out.loc[idx, "decision_date"] = vals
    return out


def decision_open_timestamp(events: pd.DataFrame, pcfg: PitConfig, tz: str) -> pd.Series:
    """Timestamp COMPLETO (tz-aware) do open do pregão de decisão.

    É contra ESTE instante que a guarda compara a call — normalizar para data
    esconderia o caso "call 19h → decisão no mesmo dia" (armadilha nº 2).
    """
    open_t = _parse_hhmm(pcfg.market_open_et)
    naive = events["decision_date"] + pd.Timedelta(hours=open_t.hour, minutes=open_t.minute)
    return naive.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")


def assert_no_lookahead(call_ts: pd.Series, decision_open_ts: pd.Series) -> None:
    """Guarda FORTE: toda decisão deve ser estritamente posterior à call.

    Compara TIMESTAMPS COMPLETOS (não datas): uma call às 19:00 com decisão no
    open (09:30) do MESMO dia viola a condição e DISPARA — é exatamente o caso
    que guardas baseadas em data normalizada deixam passar (armadilha nº 2;
    coberto por teste em tests/test_pit_leakage.py).

    Args:
        call_ts: Instantes das calls (tz-aware).
        decision_open_ts: Instantes de abertura da decisão (tz-aware; NaT é
            ignorado — significa "sem pregão disponível", tratado à parte).

    Raises:
        LookaheadError: Com amostra das linhas ofensoras.
    """
    both = call_ts.notna() & decision_open_ts.notna()
    bad = both & (decision_open_ts <= call_ts)
    if bad.any():
        sample = pd.DataFrame(
            {"call_ts": call_ts[bad], "decision_open_ts": decision_open_ts[bad]}
        ).head(5)
        raise LookaheadError(
            f"{int(bad.sum())} evento(s) com decisão NÃO-posterior à call "
            f"(vazamento de look-ahead). Amostra:\n{sample}"
        )
    logger.info("Guarda anti-look-ahead: %d eventos verificados, 0 violações.", int(both.sum()))


def attach_forward_returns(
    events: pd.DataFrame, prices: pd.DataFrame, horizons: tuple[int, ...]
) -> pd.DataFrame:
    """Anexa retornos futuros: entrada no OPEN de d0, saída no CLOSE de d_{H−1}.

    Convenção (documentada no módulo): a posição atravessa exatamente H
    sessões — ``ret_H = close[d0+H−1] / open[d0] − 1``. Horizontes sem pregões
    suficientes até o fim da amostra ficam NaN (nunca são extrapolados).

    Args:
        events: Eventos com ``ticker`` e ``decision_date`` (NaT ignorado).
        prices: Painel long com ``ticker, date, open, close``.
        horizons: Horizontes H em pregões (ex.: ``(3, 5, 10)``).

    Returns:
        ``events`` com colunas ``entry_open`` e ``ret_{H}`` por horizonte.
    """
    out = events.copy()
    out["entry_open"] = np.nan
    for h in horizons:
        out[f"ret_{h}"] = np.nan

    by_ticker = {
        t: (g["date"].to_numpy(), g["open"].to_numpy(), g["close"].to_numpy())
        for t, g in prices.sort_values("date").groupby("ticker", sort=False)
    }
    for ticker, idx in out.groupby("ticker", sort=False).groups.items():
        series = by_ticker.get(ticker)
        if series is None:
            continue
        dates, opens, closes = series
        dec = out.loc[idx, "decision_date"].to_numpy(dtype="datetime64[ns]")
        pos = np.searchsorted(dates, dec, side="left")
        valid = (
            (~pd.isna(dec)) & (pos < len(dates)) & (dates[np.minimum(pos, len(dates) - 1)] == dec)
        )
        entry = np.where(valid, opens[np.minimum(pos, len(dates) - 1)], np.nan)
        out.loc[idx, "entry_open"] = entry
        for h in horizons:
            exit_pos = pos + h - 1
            ok = valid & (exit_pos < len(dates)) & (entry > 0)
            ret = np.full(len(idx), np.nan)
            ret[ok] = closes[np.minimum(exit_pos, len(dates) - 1)][ok] / entry[ok] - 1.0
            out.loc[idx, f"ret_{h}"] = ret
    return out


def merge_asof_backward(
    events: pd.DataFrame, panel: pd.DataFrame, on: str, by: str, cols: list[str]
) -> pd.DataFrame:
    """Junção asof backward ESTRITA: anexa a última informação JÁ conhecida.

    Duas garantias NÃO-configuráveis (ADR-003; endurecidas na auditoria da
    Fase 3, achado 3/3):

    1. ``direction='backward'`` — ``forward`` casaria o evento com dados
       posteriores (vazamento óbvio).
    2. ``allow_exact_matches=False`` — a decisão do projeto acontece no OPEN
       (09:30), mas painéis de mercado são carimbados no dia e medidos no
       FECHAMENTO (ADV, volume, close): aceitar o match exato entregaria
       ~6,5h de futuro. A última observação ESTRITAMENTE anterior é a única
       que existia no open.

    Linhas com ``on`` nulo (eventos sem pregão de decisão) são preservadas com
    as colunas do painel em NaN — o merge_asof cru levantaria ValueError.

    Args:
        events: Lado esquerdo (eventos), qualquer ordem.
        panel: Lado direito (painel de mercado) com colunas ``on``/``by``.
        on: Coluna temporal da junção (mesmo nome nos dois lados).
        by: Coluna de agrupamento (``ticker``).
        cols: Colunas do painel a anexar.

    Returns:
        ``events`` com as colunas anexadas (última observação < ``on``), na
        ordem e com o índice originais.
    """
    pos_key = "__tonediv_pos__"
    left = events.copy()
    left[pos_key] = np.arange(len(left))
    # Alinha o dtype da chave ``by``: eventos lidos de parquet podem vir como
    # StringDtype e o painel de preços como object. ``pd.merge_asof`` exige que
    # a chave ``by`` tenha EXATAMENTE o mesmo dtype nos dois lados, senão levanta
    # MergeError (achado na primeira execução real — scripts 06-09).
    left[by] = left[by].astype(object)
    valid = left[on].notna()

    right = panel[[by, on, *cols]].dropna(subset=[on]).sort_values(on).copy()
    right[by] = right[by].astype(object)
    merged = pd.merge_asof(
        left[valid].sort_values(on),
        right,
        on=on,
        by=by,
        direction="backward",
        allow_exact_matches=False,
    )
    out = pd.concat([merged, left[~valid]], ignore_index=True).sort_values(pos_key)
    out.index = events.index[out[pos_key].to_numpy()]
    return out.drop(columns=pos_key).rename_axis(events.index.name)
