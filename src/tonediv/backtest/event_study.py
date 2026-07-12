"""Event study: CAR via market model + placebo de datas falsas (evidência da tese).

Por que existe:
    Antes de empacotar o sinal como estratégia, provamos que ele contém
    informação: retornos anormais acumulados (CAR) ao redor do evento,
    agrupados por magnitude da feature, contra uma distribuição nula honesta.

Arquitetura numérica (decisão de engenharia relevante):
    O market model (α, β por evento) e o CAR são calculados por **somas
    acumuladas + indexação vetorial** sobre um painel FLAT (todas as séries de
    retorno concatenadas com offsets por ticker). OLS de janela vira aritmética
    O(1) por evento — o que torna o placebo de ``fake_date_iters × n_eventos``
    reestimações (milhões de regressões) uma operação de segundos, sem
    aproximação: cada data falsa reestima α/β na própria janela deslocada.

Placebo de DATAS FALSAS (ADR-009): mesma empresa, data deslocada
aleatoriamente em ±[21, max_shift] pregões (excluindo a vizinhança do evento
real para não contaminar a nula). A distribuição do CAR médio nas datas falsas
é a régua contra a qual o CAR observado é julgado.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tonediv.config import EventStudyConfig

logger = logging.getLogger(__name__)

# Folga adicional do deslocamento mínimo do placebo além da própria janela de
# evento. Não é parâmetro de config porque não há decisão a tomar; o mínimo
# efetivo é DERIVADO das janelas do config em runtime (ver
# :func:`_placebo_min_shift`) — uma constante fixa ficaria silenciosamente
# inválida se alguém alargasse a janela de evento (auditoria da Fase 4).
_PLACEBO_SHIFT_MARGIN = 10


def _placebo_min_shift(cfg: EventStudyConfig) -> int:
    """Deslocamento mínimo das datas falsas: largura da janela de evento + folga."""
    return (cfg.event_end - cfg.event_start + 1) + _PLACEBO_SHIFT_MARGIN


@dataclass(frozen=True)
class FlatPanel:
    """Painel flat de retornos alinhados ao mercado, com somas-prefixo.

    Attributes:
        dates: Datas (datetime64) concatenadas de todos os tickers.
        offsets: Início da fatia de cada ticker no array flat.
        lengths: Comprimento da fatia de cada ticker.
        ticker_index: Mapeamento ticker -> posição em offsets/lengths.
        p_r, p_rm, p_r_rm, p_rm2: Somas-prefixo (len+1) de r, rm, r·rm, rm².
    """

    dates: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    ticker_index: dict[str, int]
    p_r: np.ndarray
    p_rm: np.ndarray
    p_r_rm: np.ndarray
    p_rm2: np.ndarray


def build_flat_panel(prices: pd.DataFrame, market_ticker: str) -> FlatPanel:
    """Alinha cada ticker ao mercado e monta o painel flat com somas-prefixo.

    Mantém apenas datas em que AMBOS (ação e mercado) têm retorno finito — o
    market model exige pares completos; buracos de negociação saem da conta em
    vez de virarem zeros artificiais.

    Args:
        prices: Painel long com ``ticker, date, ret_cc`` (script 01).
        market_ticker: Ticker do proxy de mercado (``^GSPC`` no config).

    Returns:
        :class:`FlatPanel` pronto para :func:`compute_cars` e o placebo.
    """
    mkt = (
        prices.loc[prices["ticker"] == market_ticker, ["date", "ret_cc"]]
        .dropna()
        .rename(columns={"ret_cc": "rm"})
    )
    stocks = prices.loc[prices["ticker"] != market_ticker, ["ticker", "date", "ret_cc"]].dropna()
    merged = stocks.merge(mkt, on="date", how="inner").sort_values(["ticker", "date"])

    dates_parts: list[np.ndarray] = []
    r_parts: list[np.ndarray] = []
    rm_parts: list[np.ndarray] = []
    offsets: list[int] = []
    lengths: list[int] = []
    ticker_index: dict[str, int] = {}
    cursor = 0
    for ticker, g in merged.groupby("ticker", sort=False):
        ticker_index[ticker] = len(offsets)
        offsets.append(cursor)
        lengths.append(len(g))
        cursor += len(g)
        dates_parts.append(g["date"].to_numpy(dtype="datetime64[ns]"))
        r_parts.append(g["ret_cc"].to_numpy(dtype=np.float64))
        rm_parts.append(g["rm"].to_numpy(dtype=np.float64))

    r = np.concatenate(r_parts) if r_parts else np.empty(0)
    rm = np.concatenate(rm_parts) if rm_parts else np.empty(0)

    def _prefix(x: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(x)])

    return FlatPanel(
        dates=np.concatenate(dates_parts) if dates_parts else np.empty(0, dtype="datetime64[ns]"),
        offsets=np.asarray(offsets, dtype=np.int64),
        lengths=np.asarray(lengths, dtype=np.int64),
        ticker_index=ticker_index,
        p_r=_prefix(r),
        p_rm=_prefix(rm),
        p_r_rm=_prefix(r * rm),
        p_rm2=_prefix(rm * rm),
    )


def _window_sums(panel: FlatPanel, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, ...]:
    """Somas de r, rm, r·rm, rm² nas janelas ``[lo, hi]`` (índices FLAT, inclusivos)."""
    return (
        panel.p_r[hi + 1] - panel.p_r[lo],
        panel.p_rm[hi + 1] - panel.p_rm[lo],
        panel.p_r_rm[hi + 1] - panel.p_r_rm[lo],
        panel.p_rm2[hi + 1] - panel.p_rm2[lo],
    )


def _car_at_positions(
    panel: FlatPanel, base: np.ndarray, length: np.ndarray, pos: np.ndarray, cfg: EventStudyConfig
) -> np.ndarray:
    """CAR de cada evento com t0 em ``pos`` (posição RELATIVA à fatia do ticker).

    α/β estimados na janela ``[pos+est_start, pos+est_end]`` e aplicados na
    janela ``[pos+event_start, pos+event_end]``:
    ``CAR = Σr_ev − (α·n_ev + β·Σrm_ev)``. Eventos cuja janela extrapola a
    série do ticker (ou com estimação < ``min_estimation_obs``) saem NaN.

    Nota (auditoria da Fase 4): como o painel flat só contém datas com AMBOS
    os retornos, janelas dentro da série são sempre CHEIAS — o check de
    ``min_estimation_obs`` só decide se as janelas do config forem alteradas
    (ex.: estimação mais curta). Não é parâmetro morto: é invariante de
    segurança para mudanças de configuração.
    """
    est_lo_rel, est_hi_rel = pos + cfg.estimation_start, pos + cfg.estimation_end
    ev_lo_rel, ev_hi_rel = pos + cfg.event_start, pos + cfg.event_end
    n_est = est_hi_rel - est_lo_rel + 1
    n_ev = ev_hi_rel - ev_lo_rel + 1

    valid = (
        (est_lo_rel >= 0)
        & (ev_hi_rel < length)
        & (est_lo_rel < ev_lo_rel)
        & (n_est >= cfg.min_estimation_obs)
    )
    safe = np.where(valid)[0]
    car = np.full(len(pos), np.nan)
    if len(safe) == 0:
        return car

    est_lo = base[safe] + est_lo_rel[safe]
    est_hi = base[safe] + est_hi_rel[safe]
    ev_lo = base[safe] + ev_lo_rel[safe]
    ev_hi = base[safe] + ev_hi_rel[safe]

    s_r, s_rm, s_r_rm, s_rm2 = _window_sums(panel, est_lo, est_hi)
    n = n_est[safe].astype(np.float64)
    denom = n * s_rm2 - s_rm**2
    ok = np.abs(denom) > 1e-18
    beta = np.full(len(safe), np.nan)
    beta[ok] = (n[ok] * s_r_rm[ok] - s_r[ok] * s_rm[ok]) / denom[ok]
    alpha = (s_r - beta * s_rm) / n

    e_r, e_rm, _, _ = _window_sums(panel, ev_lo, ev_hi)
    car[safe] = e_r - (alpha * n_ev[safe] + beta * e_rm)
    return car


def _locate_events(events: pd.DataFrame, panel: FlatPanel) -> tuple[np.ndarray, ...]:
    """Resolve (base, length, pos) de cada evento no painel flat.

    ``pos`` é a posição do ``decision_date`` DENTRO da fatia do ticker; exige
    match exato de data (a decisão veio das mesmas séries de preço). Eventos
    de ticker sem série ou data não encontrada ficam pos=-1 (inválidos).
    """
    n = len(events)
    base = np.zeros(n, dtype=np.int64)
    length = np.zeros(n, dtype=np.int64)
    pos = np.full(n, -1, dtype=np.int64)
    dec = events["decision_date"].to_numpy(dtype="datetime64[ns]")
    tickers = events["ticker"].to_numpy()
    for i in range(n):
        idx = panel.ticker_index.get(tickers[i])
        if idx is None or np.isnat(dec[i]):
            continue
        b, ln = panel.offsets[idx], panel.lengths[idx]
        sl = panel.dates[b : b + ln]
        p = int(np.searchsorted(sl, dec[i]))
        if p < ln and sl[p] == dec[i]:
            base[i], length[i], pos[i] = b, ln, p
    return base, length, pos


def compute_cars(events: pd.DataFrame, panel: FlatPanel, cfg: EventStudyConfig) -> pd.DataFrame:
    """Anexa a coluna ``car`` (e validade) a cada evento.

    Returns:
        Cópia de ``events`` com ``car`` (NaN para eventos sem janela válida).
        O nº de descartes é logado — nunca silencioso.
    """
    base, length, pos = _locate_events(events, panel)
    out = events.copy()
    car = np.full(len(events), np.nan)
    located = pos >= 0
    if located.any():
        car[located] = _car_at_positions(panel, base[located], length[located], pos[located], cfg)
    out["car"] = car
    logger.info(
        "Event study: %d eventos, %d com CAR válido (%d sem janela/posic.).",
        len(out),
        int(np.isfinite(car).sum()),
        int((~np.isfinite(car)).sum()),
    )
    return out


def group_car_table(events_car: pd.DataFrame, feature: str, groups: str) -> pd.DataFrame:
    """Tabela de CAR médio por grupo de magnitude da feature (tercil/quintil).

    Inclui a linha ``HML`` (grupo alto − grupo baixo, teste t de Welch) — o
    contraste que a tese prevê. Erros-padrão simples: a correção temporal
    (Newey-West) pertence às SÉRIES do portfólio; aqui a unidade é o evento.
    """
    from scipy import stats as sps

    q = 3 if groups == "tercile" else 5
    df = events_car[[feature, "car"]].dropna()
    if len(df) < q * 2:
        return pd.DataFrame()
    df = df.assign(bucket=pd.qcut(df[feature], q, labels=[f"G{i + 1}" for i in range(q)]))
    rows = []
    for name, g in df.groupby("bucket", observed=True):
        rows.append(
            {
                "grupo": str(name),
                "n": len(g),
                "car_medio": g["car"].mean(),
                "t_simples": g["car"].mean() / (g["car"].std(ddof=1) / np.sqrt(len(g))),
            }
        )
    hi = df[df["bucket"] == f"G{q}"]["car"]
    lo = df[df["bucket"] == "G1"]["car"]
    t, p = sps.ttest_ind(hi, lo, equal_var=False)
    rows.append(
        {"grupo": "HML", "n": len(hi) + len(lo), "car_medio": hi.mean() - lo.mean(), "t_simples": t}
    )
    table = pd.DataFrame(rows)
    table.attrs["hml_pvalue"] = float(p)
    return table


def placebo_fake_dates(
    events: pd.DataFrame, panel: FlatPanel, cfg: EventStudyConfig, rng: np.random.Generator
) -> tuple[float, np.ndarray]:
    """Distribuição nula do CAR médio via datas falsas (mesma empresa).

    Para cada iteração, cada evento é deslocado ±[min, max_shift] pregões
    (sinal e magnitude sorteados; a vizinhança do evento real fica de fora — o
    mínimo é derivado da janela de evento do config) e o market model é
    REESTIMADO na janela deslocada. Datas falsas PODEM cair perto de OUTRAS
    calls da mesma empresa (por design: earnings acontecem a cada ~63 pregões;
    excluí-las esvaziaria o espaço de deslocamentos) — o efeito medido na
    auditoria foi desprezível e a direção é CONSERVADORA (nula levemente mais
    gorda). O p-valor bicaudal usa a correção +1 (nunca reporta zero exato).

    Returns:
        Tupla ``(p_valor, distribuição_nula_do_CAR_médio)``.
    """
    base, length, pos = _locate_events(events, panel)
    located = pos >= 0
    base, length, pos = base[located], length[located], pos[located]
    observed = float(np.nanmean(events.loc[located, "car"].to_numpy()))

    min_shift = _placebo_min_shift(cfg)
    null = np.full(cfg.placebo_fake_date_iters, np.nan)
    for it in range(cfg.placebo_fake_date_iters):
        mag = rng.integers(min_shift, cfg.placebo_max_shift_days + 1, size=len(pos))
        sign = rng.choice([-1, 1], size=len(pos))
        fake_pos = pos + sign * mag
        cars = _car_at_positions(panel, base, length, fake_pos, cfg)
        null[it] = np.nanmean(cars) if np.isfinite(cars).any() else np.nan
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(observed):
        return float("nan"), null
    p = (1.0 + float(np.sum(np.abs(null) >= abs(observed)))) / (len(null) + 1.0)
    return p, null


def bootstrap_mean_ci(
    values: np.ndarray, iters: int, rng: np.random.Generator, level: float = 0.95
) -> tuple[float, float]:
    """IC percentílico do CAR médio por reamostragem de eventos (bootstrap)."""
    v = values[np.isfinite(values)]
    if len(v) < 3:
        return float("nan"), float("nan")
    means = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(iters)])
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))
