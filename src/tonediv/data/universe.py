"""Seleção do universo tech e mapeamento de tickers (ADR-001).

Método híbrido decidido em conjunto com o time:
    1. REDE PROGRAMÁTICA — inclui tickers cujo setor/indústria atuais (yfinance)
       batem com as regras de ``discovery`` (pega tech que a lista curada
       esqueceu).
    2. UNIÃO com a lista curada (núcleo IT + internet/mídia interativa +
       deslistadas para survivorship).
    3. SUBTRAÇÃO — tickers de ``sensitivity_blocks`` saem do núcleo automático,
       permanecendo removíveis no teste com/sem cada bloco. Sem isso, a Yahoo
       classifica UBER/ENPH/FSLR/JBL/FLEX como "Technology" e eles entrariam no
       núcleo, quebrando o teste de sensibilidade.

Aliases (FB→META, PCLN→BKNG, ...) são aplicados ANTES de qualquer agrupamento
por empresa, senão a série de ΔTone (que compara com a call anterior da MESMA
empresa) quebra silenciosamente na troca de ticker.

Schema de saída (``universe_df``) documentado em docs/data_schemas.md.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from tonediv.config import Config, DiscoveryConfig, UniverseConfig

logger = logging.getLogger(__name__)


def canonical_ticker(ticker: str, aliases: dict[str, str]) -> str:
    """Mapeia um ticker para sua forma canônica via ``aliases`` (idempotente)."""
    return aliases.get(ticker, ticker)


def apply_aliases(tickers: pd.Series, aliases: dict[str, str]) -> pd.Series:
    """Aplica ``ticker_aliases`` a uma série de tickers, retornando os canônicos.

    Deve ser chamada ANTES de qualquer agrupamento por empresa (ver módulo).
    """
    return tickers.map(lambda t: aliases.get(t, t)).astype("string")


def _fetch_one_info(ticker: str, retries: int, pause: float) -> tuple[str | None, str | None]:
    """Busca ``(sector, industry)`` de um ticker no yfinance, tolerante a falha.

    Backoff EXPONENCIAL em rate-limit (HTTP 429): o endpoint quoteSummary do
    Yahoo é agressivamente limitado; insistir no ritmo normal só prolonga o
    bloqueio (cascata de 429 observada na primeira execução real). Falha
    definitiva retorna ``(None, None)`` — comum para deslistados; a ausência
    apenas mantém o ticker fora da rede programática, sem quebrar nada.
    """
    import yfinance as yf  # import tardio: dependência pesada, só no I/O real

    for attempt in range(1, max(1, retries) + 1):
        try:
            info = yf.Ticker(ticker).get_info()
            return info.get("sector"), info.get("industry")
        except Exception as exc:  # noqa: BLE001 - rede/dados imprevisíveis; degradar graciosamente
            wait = pause * (2**attempt) if "429" in str(exc) else pause
            logger.debug(
                "info %s tentativa %d falhou (%s); aguardando %.0fs", ticker, attempt, exc, wait
            )
            time.sleep(min(wait, 60.0))
    return None, None


def fetch_sector_industry(
    tickers: Iterable[str], cfg: Config, cache_path: Path | None = None
) -> pd.DataFrame:
    """Consulta setor/indústria atuais dos ``tickers`` (rede programática).

    Chamado apenas para o "resto desconhecido" (tickers presentes no dataset que
    não estão em nenhuma lista curada), minimizando consultas ``.info`` — que são
    lentas e sujeitas a rate-limit — ao estritamente necessário. Duas defesas
    adicionais (pós-primeira execução real, que tomou 429 em cascata):

    - **Pausa ENTRE requisições** (``retry_pause_seconds``), não só em falha —
      sem ela o Yahoo bloqueia a sequência inteira;
    - **Cache persistente opcional** (``cache_path``, parquet): consultas bem-
      sucedidas nunca são repetidas entre execuções; só os ausentes são
      buscados. I/O explícito por parâmetro (regra nº 10).

    Returns:
        DataFrame ``[ticker, sector, industry]`` (valores podem ser ``None``).
    """
    ds = cfg.data_sources
    wanted = list(dict.fromkeys(tickers))

    cached = pd.DataFrame(columns=["ticker", "sector", "industry"])
    if cache_path is not None and cache_path.is_file():
        cached = pd.read_parquet(cache_path)
        # Só reaproveita sucessos: falhas antigas merecem nova tentativa.
        cached = cached[cached["sector"].notna() & cached["ticker"].isin(wanted)]
        logger.info("Rede programática: %d setores do cache.", len(cached))

    missing = [t for t in wanted if t not in set(cached["ticker"])]
    logger.info("Rede programática: consultando setor de %d tickers...", len(missing))
    rows = []
    for ticker in missing:
        sector, industry = _fetch_one_info(ticker, ds.max_retries, ds.retry_pause_seconds)
        rows.append({"ticker": ticker, "sector": sector, "industry": industry})
        time.sleep(ds.retry_pause_seconds)  # ritmo educado ENTRE consultas

    fresh = pd.DataFrame(rows, columns=["ticker", "sector", "industry"])
    out = pd.concat([cached, fresh], ignore_index=True)
    if cache_path is not None and not fresh.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
    return out


def programmatic_net(sector_df: pd.DataFrame, disc: DiscoveryConfig) -> set[str]:
    """Aplica as regras de ``discovery`` para selecionar tickers tech por setor.

    Ordem das regras (o allow explícito vence o veto, ver ADR-001):
        1. ``core_comm_extra_tickers`` (ex.: NFLX) — incluído sempre.
        2. Indústrias vetadas (telecom/mídia tradicional) — excluídas.
        3. ``core_sectors`` (ex.: "Technology") — incluído.
        4. ``core_comm_industries`` (internet/gaming) — incluído.

    Args:
        sector_df: Saída de :func:`fetch_sector_industry`.
        disc: Bloco :class:`~tonediv.config.DiscoveryConfig`.

    Returns:
        Conjunto de tickers admitidos pela rede programática.
    """
    if not disc.use_yfinance_sector or sector_df.empty:
        return set()
    extra = set(disc.core_comm_extra_tickers)
    net: set[str] = set()
    for row in sector_df.itertuples(index=False):
        if row.ticker in extra:
            net.add(row.ticker)
        elif row.industry in disc.exclude_comm_industries:
            continue
        elif row.sector in disc.core_sectors or row.industry in disc.core_comm_industries:
            net.add(row.ticker)
    return net


def _block_of(ticker: str, universe: UniverseConfig) -> str | None:
    """Retorna o nome do bloco de sensibilidade do ticker, ou ``None``."""
    for block, tickers in universe.sensitivity_blocks.items():
        if ticker in tickers:
            return block
    return None


def _layer_of(ticker: str, universe: UniverseConfig, net: set[str], block: str | None) -> str:
    """Rotula a camada de origem primária do ticker (para o relatório/auditoria)."""
    if block is not None:
        return f"sensitivity:{block}"
    if ticker in universe.core_it:
        return "core_it"
    if ticker in universe.core_internet_media:
        return "core_internet_media"
    if ticker in net:
        return "net_discovery"
    return "delisted"


def resolve_universe(
    present_tickers: Iterable[str], sector_df: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Resolve o universo candidato completo e sua tabela de pertencimento.

    Combina lista curada + rede programática + deslistadas e aplica a regra de
    subtração dos blocos de sensibilidade (ver módulo/ADR-001).

    Args:
        present_tickers: Tickers CANÔNICOS presentes no dataset de transcrições
            (aliases já aplicados pelo chamador).
        sector_df: Saída de :func:`fetch_sector_industry` (pode ser vazia).
        cfg: Configuração do projeto.

    Returns:
        ``universe_df`` com uma linha por ticker candidato e colunas:
        ``ticker, in_core, sensitivity_block, is_delisted, layer, sector,
        industry, in_dataset``. ``in_core`` = pertence ao núcleo sempre-dentro
        (curado ∪ rede, menos sensibilidade). A seleção efetiva por rodada (com/
        sem cada bloco) é feita por :func:`select_tickers`.
    """
    u = cfg.universe
    present = set(present_tickers)
    net = programmatic_net(sector_df, u.discovery)
    curated_core = u.curated_core()
    sensitivity = u.sensitivity_tickers()
    delisted = set(u.delisted_check)

    candidates = sorted(curated_core | net | sensitivity | delisted)
    sectors = sector_df.set_index("ticker") if not sector_df.empty else pd.DataFrame()

    records = []
    for ticker in candidates:
        block = _block_of(ticker, u)
        in_core = ticker in (curated_core | net) and ticker not in sensitivity
        sec = sectors.loc[ticker, "sector"] if ticker in sectors.index else None
        ind = sectors.loc[ticker, "industry"] if ticker in sectors.index else None
        records.append(
            {
                "ticker": ticker,
                "in_core": in_core,
                "sensitivity_block": block,
                "is_delisted": ticker in delisted,
                "layer": _layer_of(ticker, u, net, block),
                "sector": sec,
                "industry": ind,
                "in_dataset": ticker in present,
            }
        )
    universe_df = pd.DataFrame.from_records(records)
    logger.info(
        "Universo resolvido: %d candidatos, %d no dataset (núcleo=%d, deslistados=%d).",
        len(universe_df),
        int(universe_df["in_dataset"].sum()),
        int(universe_df["in_core"].sum()),
        int(universe_df["is_delisted"].sum()),
    )
    return universe_df


def select_tickers(universe_df: pd.DataFrame, active_blocks: Iterable[str]) -> list[str]:
    """Seleciona os tickers efetivos de uma rodada, dado os blocos ativos.

    Regra: entram o núcleo sempre-dentro, as deslistadas, e os tickers dos
    blocos de sensibilidade ATIVOS — tudo restrito ao que existe no dataset.
    Este é o ponto onde a grade de robustez liga/desliga cada bloco.

    Args:
        universe_df: Saída de :func:`resolve_universe`.
        active_blocks: Blocos de sensibilidade a incluir nesta rodada (vazio =
            só núcleo + deslistadas).

    Returns:
        Lista ordenada de tickers canônicos elegíveis na rodada.
    """
    active = set(active_blocks)
    in_block = universe_df["sensitivity_block"].isin(active)
    keep = universe_df["in_dataset"] & (
        universe_df["in_core"] | universe_df["is_delisted"] | in_block
    )
    return sorted(universe_df.loc[keep, "ticker"].tolist())
