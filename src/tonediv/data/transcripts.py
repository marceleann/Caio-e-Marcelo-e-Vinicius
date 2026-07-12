"""Download, parse e filtro de qualidade das transcrições de earnings calls.

Fonte: ``kurry/sp500_earnings_transcripts`` (HuggingFace, MIT, ~33k calls,
2005–2025). Campo ``structured_content`` traz a call segmentada por orador.

Por que existe:
    Isola TODO o conhecimento sobre o formato bruto do dataset num único lugar,
    convertendo-o em duas tabelas canônicas e documentadas (``calls`` e
    ``utterances``) que o resto do pipeline consome. O schema real do dataset é
    tratado de forma DEFENSIVA (nomes de coluna resolvidos por candidatos,
    case-insensitive) porque não foi possível inspecioná-lo em tempo de
    desenvolvimento; o script 00 revela e valida a estrutura efetiva.

Schemas de saída documentados em docs/data_schemas.md.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tonediv.config import Config

logger = logging.getLogger(__name__)

# Nomes de coluna CANDIDATOS no dataset bruto. Resolvidos case-insensitive; o
# primeiro que casar é usado. Comentário do PORQUÊ: o schema exato do HF não
# pôde ser verificado no desenvolvimento, então aceitamos sinônimos comuns em
# vez de cravar um nome e quebrar silenciosamente. O script 00 loga o que achou.
_TICKER_COLS: tuple[str, ...] = ("ticker", "symbol", "tic", "company_ticker")
_COMPANY_COLS: tuple[str, ...] = ("company", "company_name", "name", "conm")
_DATETIME_COLS: tuple[str, ...] = ("datetime", "date_time", "call_datetime", "timestamp")
_DATE_COLS: tuple[str, ...] = ("date", "call_date", "event_date", "report_date")
_TIME_COLS: tuple[str, ...] = ("time", "call_time", "event_time")
_YEAR_COLS: tuple[str, ...] = ("year", "fiscal_year", "fy")
_QUARTER_COLS: tuple[str, ...] = ("quarter", "fiscal_quarter", "fq", "fqtr")

# Chaves candidatas DENTRO de cada item do structured_content.
_SPEAKER_KEYS: tuple[str, ...] = ("speaker", "name", "presenter", "participant", "role")
_TEXT_KEYS: tuple[str, ...] = ("text", "content", "speech", "value", "utterance", "body")


@dataclass(frozen=True)
class QualityDropReport:
    """Contabiliza calls descartadas pelos filtros de qualidade (armadilha nº 6).

    Attributes:
        n_input: Calls recebidas (já filtradas ao universo).
        n_dropped_short: Calls descartadas por texto total curto.
        n_dropped_few_utts: Calls descartadas por poucas falas.
        n_kept: Calls sobreviventes.
    """

    n_input: int
    n_dropped_short: int
    n_dropped_few_utts: int
    n_kept: int


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Retorna o 1º nome de coluna de ``df`` que casa com ``candidates``.

    Comparação case-insensitive. Retorna ``None`` se nenhum casar — o chamador
    decide se isso é fatal ou se há fallback (ex.: derivar ano do datetime).
    """
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_hf_transcripts(cfg: Config) -> pd.DataFrame:
    """Baixa o dataset de transcrições do HuggingFace como DataFrame bruto.

    Função FINA de I/O (a lógica pura vive nas funções de normalização, para
    permitir teste offline). O download é cacheado localmente pelo ``datasets``.

    Args:
        cfg: Configuração do projeto (dataset id e split vêm do config).

    Returns:
        DataFrame bruto, uma linha por call, com o schema original do dataset.
    """
    from datasets import load_dataset  # import tardio: dependência pesada, só no I/O real

    ds = cfg.data_sources
    logger.info(
        "Baixando transcrições de %s [%s]...", ds.transcripts_hf_dataset, ds.transcripts_split
    )
    dataset = load_dataset(ds.transcripts_hf_dataset, split=ds.transcripts_split)
    df = dataset.to_pandas()
    logger.info("Transcrições brutas: %d linhas, colunas=%s", len(df), list(df.columns))
    return df


def _combine_datetime(df: pd.DataFrame, tz: str) -> tuple[pd.Series, pd.Series]:
    """Constrói o timestamp da call e a flag de horário real presente.

    Regras (base para o alinhamento PIT da Fase 3, ver ADR-002/ADR-008):
        - Combina coluna de datetime única OU (date + time) separadas.
        - Timestamps naive são assumidos JÁ em ``tz`` (fuso da bolsa) e
          localizados; DST ambíguo/inexistente é tratado sem quebrar.
        - ``has_time`` é ``False`` quando não há hora ou a hora é meia-noite
          exata (00:00:00) — tratada como suspeita/ausente, não como intraday.

    Returns:
        Tupla ``(call_datetime, has_time)`` alinhada ao índice de ``df``.
    """
    dt_col = _resolve_column(df, _DATETIME_COLS)
    if dt_col is not None:
        raw = pd.to_datetime(df[dt_col], errors="coerce")
    else:
        date_col = _resolve_column(df, _DATE_COLS)
        if date_col is None:
            raise KeyError("Nenhuma coluna de data/datetime encontrada no dataset.")
        time_col = _resolve_column(df, _TIME_COLS)
        combined = df[date_col].astype(str)
        if time_col is not None:
            combined = combined + " " + df[time_col].astype(str)
        raw = pd.to_datetime(combined, errors="coerce")

    if raw.dt.tz is None:
        raw = raw.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    else:
        raw = raw.dt.tz_convert(tz)
    # Meia-noite exata é indistinguível de "hora ausente" -> não confiar como intraday.
    has_time = ~((raw.dt.hour == 0) & (raw.dt.minute == 0) & (raw.dt.second == 0))
    has_time = has_time & raw.notna()
    return raw, has_time


def _dedupe_ids(base: pd.Series) -> pd.Series:
    """Torna únicos os ``call_id`` repetidos anexando um sufixo sequencial.

    Duas calls podem gerar a mesma base (mesmo ticker/tri). Em vez de perder
    uma, sufixamos determinísticamente (_2, _3, ...) preservando a ordem do
    dataset, garantindo chave primária única sem descartar dados.
    """
    counter = base.groupby(base).cumcount()
    suffix = counter.map(lambda n: "" if n == 0 else f"_{n + 1}")
    return base.str.cat(suffix)


def normalize_calls(raw_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Normaliza o DataFrame bruto para metadados canônicos por call.

    Args:
        raw_df: Saída de :func:`load_hf_transcripts` (ou equivalente em teste).
        cfg: Configuração (fuso e nome do campo de conteúdo).

    Returns:
        DataFrame com colunas ``[call_id, ticker, company, call_datetime,
        has_time, year, quarter, _content]``. ``ticker`` em maiúsculas e SEM
        alias aplicado ainda (aliases entram em :mod:`tonediv.data.universe`).
        ``_content`` (privado) carrega o ``structured_content`` bruto para o
        explode subsequente; é descartado antes de salvar a tabela de calls.
    """
    ticker_col = _resolve_column(raw_df, _TICKER_COLS)
    if ticker_col is None:
        raise KeyError("Nenhuma coluna de ticker encontrada no dataset.")
    content_col = _resolve_column(
        raw_df, (cfg.data_sources.transcripts_content_field,)
    ) or _resolve_column(raw_df, ("structured_content", "transcript", "content"))
    if content_col is None:
        raise KeyError("Nenhuma coluna de conteúdo estruturado encontrada no dataset.")

    call_datetime, has_time = _combine_datetime(raw_df, cfg.sample.timezone)
    year_col, quarter_col = _resolve_column(raw_df, _YEAR_COLS), _resolve_column(
        raw_df, _QUARTER_COLS
    )
    company_col = _resolve_column(raw_df, _COMPANY_COLS)

    out = pd.DataFrame(index=raw_df.index)
    out["ticker"] = raw_df[ticker_col].astype("string").str.strip().str.upper()
    out["company"] = raw_df[company_col].astype("string") if company_col else pd.NA
    out["call_datetime"] = call_datetime
    out["has_time"] = has_time
    # Ano/tri explícitos quando existem; senão derivados do datetime (fallback).
    out["year"] = raw_df[year_col] if year_col else call_datetime.dt.year
    out["year"] = out["year"].astype("Int64")
    out["quarter"] = raw_df[quarter_col] if quarter_col else call_datetime.dt.quarter
    out["quarter"] = out["quarter"].astype("Int64")
    out["_content"] = raw_df[content_col].to_numpy()

    base = (
        out["ticker"].fillna("NA")
        + "_"
        + out["year"].astype("string")
        + "Q"
        + out["quarter"].astype("string")
    )
    out.insert(0, "call_id", _dedupe_ids(base))
    return out.reset_index(drop=True)


def _extract_utterance(
    item: Any,  # noqa: ANN401 - item bruto do dataset, tipo desconhecido por design
) -> tuple[str | None, str] | None:
    """Extrai ``(speaker, text)`` de um item do structured_content.

    Aceita dicionários com chaves sinônimas (ver ``_SPEAKER_KEYS``/``_TEXT_KEYS``).
    Retorna ``None`` se não houver texto aproveitável (item ignorado).
    """
    if not isinstance(item, dict):
        return None
    speaker = next((str(item[k]) for k in _SPEAKER_KEYS if k in item and item[k] is not None), None)
    text = next((str(item[k]) for k in _TEXT_KEYS if k in item and item[k] is not None), "")
    text = text.strip()
    if not text:
        return None
    return speaker, text


def _parse_structured_content(
    value: Any,  # noqa: ANN401 - valor bruto do dataset, tipo desconhecido por design
) -> list[tuple[str | None, str]]:
    """Converte o ``structured_content`` bruto numa lista ``[(speaker, text)]``.

    Robusto a três formatos plausíveis: lista/array de dicts, string JSON que
    decodifica para lista, ou valor não segmentável (retorna lista vazia e loga,
    fazendo a call cair no filtro de qualidade em vez de contaminar o sinal).
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            logger.debug("structured_content em texto puro não segmentável; call será filtrada.")
            return []
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        return []
    parsed = [_extract_utterance(item) for item in value]
    return [u for u in parsed if u is not None]


def build_utterances(calls_df: pd.DataFrame) -> pd.DataFrame:
    """Explode ``calls_df['_content']`` numa tabela longa de falas.

    Args:
        calls_df: Saída de :func:`normalize_calls` (precisa da coluna ``_content``).

    Returns:
        DataFrame ``[call_id, utterance_idx, speaker, text, n_chars]``, uma linha
        por fala, ordenada como no original (``utterance_idx`` preserva a ordem
        — essencial para a heurística de papéis, que depende da sequência).
    """
    rows: list[dict[str, Any]] = []
    for call_id, content in zip(calls_df["call_id"], calls_df["_content"], strict=True):
        for idx, (speaker, text) in enumerate(_parse_structured_content(content)):
            rows.append(
                {
                    "call_id": call_id,
                    "utterance_idx": idx,
                    "speaker": speaker,
                    "text": text,
                    "n_chars": len(text),
                }
            )
    return pd.DataFrame(rows, columns=["call_id", "utterance_idx", "speaker", "text", "n_chars"])


def calls_metadata(calls_df: pd.DataFrame, utterances_df: pd.DataFrame) -> pd.DataFrame:
    """Anexa contagens agregadas por call e remove a coluna interna ``_content``.

    Returns:
        Metadados por call, prontos para salvar, com ``n_utterances`` e
        ``n_chars_total`` derivados da tabela de falas.
    """
    agg = utterances_df.groupby("call_id").agg(
        n_utterances=("utterance_idx", "size"),
        n_chars_total=("n_chars", "sum"),
    )
    meta = calls_df.drop(columns="_content").merge(agg, on="call_id", how="left")
    meta["n_utterances"] = meta["n_utterances"].fillna(0).astype("Int64")
    meta["n_chars_total"] = meta["n_chars_total"].fillna(0).astype("Int64")
    return meta


def filter_quality(
    calls_meta: pd.DataFrame, utterances_df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, QualityDropReport]:
    """Descarta calls curtas/vazias e reporta quantas caíram (armadilha nº 6).

    Aplica filtros de NÍVEL DE CALL (texto total e nº de falas). O filtro de
    fala curta (``min_chars_utterance``) é aplicado só no scoring (Fase 3), pois
    falas curtas do Operator ("Thank you") ainda ajudam a detectar a estrutura
    do Q&A na inferência de papéis.

    Args:
        calls_meta: Saída de :func:`calls_metadata`.
        utterances_df: Tabela de falas correspondente.
        cfg: Configuração (limiares de qualidade).

    Returns:
        Tupla ``(calls_kept, utterances_kept, report)``.
    """
    q = cfg.quality
    short = calls_meta["n_chars_total"] < q.min_chars_call
    few = calls_meta["n_utterances"] < q.min_utterances_call
    keep = ~(short | few)
    calls_kept = calls_meta[keep].reset_index(drop=True)
    utts_kept = utterances_df[utterances_df["call_id"].isin(calls_kept["call_id"])].reset_index(
        drop=True
    )
    report = QualityDropReport(
        n_input=len(calls_meta),
        n_dropped_short=int(short.sum()),
        n_dropped_few_utts=int((few & ~short).sum()),
        n_kept=int(keep.sum()),
    )
    logger.info(
        "Qualidade: %d calls -> %d mantidas (%d curtas, %d poucas falas).",
        report.n_input,
        report.n_kept,
        report.n_dropped_short,
        report.n_dropped_few_utts,
    )
    return calls_kept, utts_kept, report


def prepare_calls(raw_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compõe normalização + aplicação de aliases (ticker canônico).

    Orquestrador PURO (sem I/O) que o script 01 chama logo após o download,
    mantendo o script fino. Retorna as calls com ``_content`` ainda anexado.
    """
    from tonediv.data.universe import apply_aliases  # local: evita qualquer ciclo futuro

    calls = normalize_calls(raw_df, cfg)
    # Preserva o ticker ORIGINAL: o teste de survivorship (script 00) precisa
    # procurar o nome literal (ex.: JAVA, SYMC), que o alias substituiria.
    calls["ticker_raw"] = calls["ticker"]
    calls["ticker"] = apply_aliases(calls["ticker"], dict(cfg.universe.ticker_aliases))
    return calls


def build_universe_tables(
    calls: pd.DataFrame, tickers: list[str], cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, QualityDropReport]:
    """Filtra ao universo, explode falas e aplica qualidade (orquestrador puro).

    Args:
        calls: Saída de :func:`prepare_calls` (com ``_content``).
        tickers: Tickers canônicos elegíveis (universo ∩ dataset).
        cfg: Configuração.

    Returns:
        Tupla ``(calls_kept, utterances_kept, report)`` pronta para salvar.
    """
    subset = calls[calls["ticker"].isin(set(tickers))].reset_index(drop=True)
    utterances = build_utterances(subset)
    meta = calls_metadata(subset, utterances)
    return filter_quality(meta, utterances, cfg)
