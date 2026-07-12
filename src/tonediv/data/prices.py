"""Download de preços (yfinance), painel long e retornos diários.

Por que existe:
    Fornece o painel de preços canônico contra o qual o sinal textual é
    alinhado (Fase 3) e os retornos futuros são medidos (Fase 4). O download é
    feito TICKER A TICKER com retry e captura graciosa de falhas: nomes
    deslistados falham de propósito e essa falha alimenta o teste de
    survivorship do script 00, em vez de derrubar o pipeline inteiro.

    Preços são baixados desde ``price_start_date`` (meados de 2004), ANTES do
    início amostral, para cobrir a janela de estimação do market model (−120
    pregões) dos primeiros eventos de 2005 (ADR-012).

Schema de saída documentado em docs/data_schemas.md.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd

from tonediv.config import Config

logger = logging.getLogger(__name__)

_OHLCV_MAP: dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


@dataclass(frozen=True)
class PriceDownloadResult:
    """Resultado do download de preços.

    Attributes:
        panel: Painel long de preços (ver :func:`download_prices`).
        succeeded: Tickers baixados com sucesso (>=1 pregão).
        failed: Tickers sem dados (deslistados, sem cobertura ou erro de rede).
    """

    panel: pd.DataFrame
    succeeded: tuple[str, ...]
    failed: tuple[str, ...]


def _download_one(
    ticker: str, start: str, end: str, auto_adjust: bool, retries: int, pause: float
) -> pd.DataFrame | None:
    """Baixa OHLCV de um ticker com até ``retries`` tentativas.

    Retorna ``None`` se todas as tentativas falharem ou vierem vazias — condição
    ESPERADA para nomes deslistados e diagnosticada, não um erro fatal.
    """
    import yfinance as yf  # import tardio: dependência pesada, só no I/O real

    for attempt in range(1, retries + 1):
        wait = pause
        try:
            df = yf.download(
                ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False, threads=False
            )
            if df is not None and not df.empty:
                return df
        except Exception as exc:  # noqa: BLE001 - rede é imprevisível; logamos e tentamos de novo
            # Backoff exponencial em rate-limit (429): insistir no ritmo normal
            # só prolonga o bloqueio do Yahoo (visto na primeira execução real).
            wait = min(pause * (2**attempt), 60.0) if "429" in str(exc) else pause
            logger.warning("yfinance %s tentativa %d/%d falhou: %s", ticker, attempt, retries, exc)
        time.sleep(wait)
    return None


def _tidy_one(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Converte o OHLCV bruto do yfinance em formato long tidy de um ticker.

    Achata eventuais colunas MultiIndex (yfinance às vezes as retorna mesmo para
    um ticker) e padroniza nomes para minúsculas.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=_OHLCV_MAP).reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "date"})
    keep = ["date", *[c for c in _OHLCV_MAP.values() if c in df.columns]]
    out = df[keep].copy()
    out.insert(1, "ticker", ticker)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out


def download_prices(tickers: list[str], cfg: Config) -> PriceDownloadResult:
    """Baixa preços do universo + mercado + benchmarks e monta o painel long.

    Args:
        tickers: Tickers do universo (canônicos). Mercado (``^GSPC``) e
            benchmarks (``SPY``/``QQQ``) do config são acrescentados e
            deduplicados automaticamente.
        cfg: Configuração (datas, auto_adjust, política de retry).

    Returns:
        :class:`PriceDownloadResult` com o painel e as listas de sucesso/falha.
    """
    ds = cfg.data_sources
    start = cfg.sample.price_start_date.isoformat()
    end = cfg.sample.end_date.isoformat()
    wanted = list(
        dict.fromkeys([*tickers, ds.market_ticker, *ds.benchmarks])
    )  # ordem estável, sem dup
    overrides = dict(cfg.universe.price_symbol_overrides)
    logger.info("Baixando preços de %d tickers (%s a %s)...", len(wanted), start, end)

    frames: list[pd.DataFrame] = []
    succeeded: list[str] = []
    failed: list[str] = []
    for ticker in wanted:
        # Baixa pelo símbolo DO YAHOO (que pode divergir do canônico: Fiserv é
        # FI para nós, FISV no Yahoo) mas grava sempre o ticker CANÔNICO — o
        # resto do pipeline nunca vê o símbolo de fonte.
        one = _download_one(
            overrides.get(ticker, ticker),
            start,
            end,
            ds.auto_adjust,
            ds.max_retries,
            ds.retry_pause_seconds,
        )
        if one is None:
            failed.append(ticker)
            continue
        frames.append(_tidy_one(one, ticker))
        succeeded.append(ticker)

    panel = pd.concat(frames, ignore_index=True) if frames else _empty_panel()
    logger.info("Preços: %d ok, %d sem dados.", len(succeeded), len(failed))
    return PriceDownloadResult(panel=panel, succeeded=tuple(succeeded), failed=tuple(failed))


def _empty_panel() -> pd.DataFrame:
    """Retorna um painel vazio com o schema correto (evita erros a jusante)."""
    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    return pd.DataFrame(columns=cols)


def add_daily_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta retorno close-to-close diário e volume financeiro ao painel.

    Args:
        panel: Painel long de :func:`download_prices`.

    Returns:
        O painel ordenado por ``(ticker, date)`` com colunas extras:
        ``ret_cc`` (retorno diário simples do fechamento) e ``dollar_volume``
        (``close * volume``, base para o filtro de liquidez point-in-time da
        estratégia na Fase 4). O retorno futuro do evento é calculado depois,
        no alinhamento PIT (entrada no open, saída no close H pregões à frente).
    """
    if panel.empty:
        return panel.assign(
            ret_cc=pd.Series(dtype="float64"), dollar_volume=pd.Series(dtype="float64")
        )
    out = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    out["ret_cc"] = out.groupby("ticker", sort=False)["close"].pct_change()
    out["dollar_volume"] = out["close"] * out["volume"]
    return out
