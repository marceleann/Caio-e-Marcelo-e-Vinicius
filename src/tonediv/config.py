"""Carregamento TIPADO do ``config.yaml`` (fonte única de verdade do projeto).

Por que existe:
    A regra de engenharia nº 5 exige zero números mágicos no código — todo
    parâmetro vive no ``config.yaml``. Este módulo transforma esse YAML em um
    objeto :class:`Config` com dataclasses congeladas, para que o resto da
    biblioteca acesse parâmetros por atributo tipado (com autocompletar e
    checagem estática) em vez de indexar dicionários soltos e propensos a erro.

    Os dataclasses são construídos por FASE: cada fase do projeto acrescenta as
    seções que passa a usar. Seções ainda não modeladas continuam acessíveis via
    :attr:`Config.raw` (escape hatch), evitando código morto para etapas futuras.

Referência: docs/DECISIONS.md (parâmetros justificados nos ADRs correspondentes).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Dataclasses por seção (congeladas = imutáveis; config é read-only em runtime)
# =============================================================================
@dataclass(frozen=True)
class PathsConfig:
    """Caminhos do projeto, já resolvidos para absolutos a partir da raiz."""

    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    data_outputs: Path
    docs: Path

    def ensure_dirs(self) -> None:
        """Cria os diretórios de dados/saída se ainda não existirem.

        Chamado pelos scripts antes de gravar parquet. Fica aqui (e não em cada
        script) para centralizar a política de layout de diretórios.
        """
        for path in (self.data_raw, self.data_interim, self.data_processed, self.data_outputs):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SampleConfig:
    """Janela amostral e fuso. 2015 NÃO corta a amostra (ver ADR-012)."""

    start_date: date
    end_date: date
    price_start_date: date  # anterior a start_date p/ cobrir estimação do market model
    timezone: str


@dataclass(frozen=True)
class DataSourcesConfig:
    """Identificadores das fontes open-source (HuggingFace + yfinance)."""

    transcripts_hf_dataset: str
    transcripts_split: str
    transcripts_content_field: str
    prices_provider: str
    market_ticker: str
    benchmarks: tuple[str, ...]
    auto_adjust: bool
    max_retries: int
    retry_pause_seconds: float


@dataclass(frozen=True)
class QualityConfig:
    """Filtros de qualidade das transcrições (armadilha nº 6)."""

    min_chars_call: int
    min_chars_utterance: int
    min_utterances_call: int
    report_dropped: bool


@dataclass(frozen=True)
class RolesConfig:
    """Heurística de papéis analista×gestão e sua validação manual (ADR-006).

    Marcadores de Q&A em 3 classes (design endurecido em revisão multi-agente):
    ``qa_start_markers`` (FORTES: abrem em qualquer posição),
    ``qa_start_markers_weak`` (FRACOS: exigem fala nomeada anterior e ausência
    de aviso) e ``qa_aviso_markers`` (anti-padrões: fala é aviso, nunca abre).
    """

    operator_label: str
    qa_start_markers: tuple[str, ...]
    qa_start_markers_weak: tuple[str, ...]
    qa_aviso_markers: tuple[str, ...]
    analyst_intro_regex: str
    validation_sample_size: int
    validation_seed: int
    validation_export_chars: int


@dataclass(frozen=True)
class FinbertConfig:
    """FinBERT: modelo, chunking e frases-sentinela do teste de labels (ADR-007).

    Nota: NÃO há parâmetro de sobreposição de chunks — sem overlap é decisão
    fixa de design (ADR-015); um parâmetro ignorado seria um no-op silencioso,
    exatamente o tipo de falha que este projeto proíbe.
    """

    model_name: str
    max_tokens: int
    batch_size: int
    device: str
    checkpoint_shard_utterances: int
    sentinel_positive: str
    sentinel_negative: str


@dataclass(frozen=True)
class FeaturesConfig:
    """Parâmetros das features de tom e do ajuste idiossincrático por pares PIT.

    Cobre o sinal principal do projeto — a Distância de Tom entre gestores da
    call (tese Angelo 2025 com FinBERT), com seus limiares de gestores/tokens e
    a residualização estritamente-passada sobre os confundidores (ADR-023) — e a
    janela de ajuste por pares (ADR-004) usada pelas versões ``_idio``.
    """

    peer_window_days: int
    peer_min_calls: int
    idio_features: tuple[str, ...]
    # Tone Distance (tese Angelo 2025 com FinBERT): divergência de tom ENTRE os
    # gestores da call. Defaults p/ construção direta em teste; o config manda.
    tone_distance_min_managers: int = 2
    tone_distance_min_manager_tokens: int = 25
    # Sinal LIMPO: resíduo estritamente-passado da distância de tom sobre os
    # confundidores (ADR-023). O sinal operável é esse resíduo, não o valor cru.
    clean_signal_confounders: tuple[str, ...] = (
        "disclosure_tone",
        "analyst_tone",
        "analyst_tone_distance",
        "size_proxy",
        "length",
    )
    clean_signal_min_history: int = 250


@dataclass(frozen=True)
class PitConfig:
    """Regras de timestamp de decisão e junção anti-look-ahead (ADR-002/003).

    Sem parâmetros para "fallback" ou direção do merge: T+1 conservador e
    ``backward`` são fixos por design — configurá-los permitiria reintroduzir
    vazamento com o pipeline "rodando verde".
    """

    market_close_et: str
    market_open_et: str
    assumed_call_duration_minutes: int
    suspicious_times: tuple[str, ...]


@dataclass(frozen=True)
class ReturnsConfig:
    """Horizontes e convenção de entrada/saída dos retornos futuros."""

    horizons: tuple[int, ...]
    entry: str
    exit: str


@dataclass(frozen=True)
class EventStudyConfig:
    """Janelas e testes do event study (ADR-018).

    Sem parâmetro de Newey-West aqui: a tabela de grupos usa teste t simples/
    Welch por decisão documentada — NW pertence às séries do portfólio.
    """

    estimation_start: int
    estimation_end: int
    event_start: int
    event_end: int
    min_estimation_obs: int
    groups: str
    bootstrap_iters: int
    placebo_fake_date_iters: int
    placebo_max_shift_days: int


@dataclass(frozen=True)
class StrategyConfig:
    """Portfólio calendar-time com tranches sobrepostas (ADR-005/018).

    Caixa a 0%, pesos iguais por tranche e gatilho top/bottom são FIXOS por
    design — não há parâmetros para eles (no-op silencioso é proibido).
    """

    holding_days: int
    variants: tuple[str, ...]
    signal_feature: str
    signal_direction: int  # +1 = comprar sinal alto; -1 = vender sinal alto (ADR-022)
    top_fraction: float
    signal_rank_window_days: int
    min_rank_history: int
    max_weight_per_name: float
    placebo_portfolio_perms: int
    liquidity_adv_window_days: int
    liquidity_min_adv_usd: float


@dataclass(frozen=True)
class CostsConfig:
    """Custos de transação (ADR-011)."""

    bps_per_side: float


@dataclass(frozen=True)
class MetricsConfig:
    """Parâmetros das métricas de desempenho."""

    trading_days_year: int
    newey_west_lags: int


@dataclass(frozen=True)
class ValidationConfig:
    """Validação temporal: PurgedKFold, placebo intra-data, PBO/CSCV e grade."""

    kfold_splits: int
    embargo_days: int
    holdout_months: int
    placebo_perm_iters: int
    pbo_cscv_blocks: int
    noise_pvalue_floor: float
    min_cell_events: int
    grid_features: tuple[str, ...]


@dataclass(frozen=True)
class DecayConfig:
    """Análise de decay pré/pós-publicação (ADR-012)."""

    split_year: int
    compare_metrics: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryConfig:
    """Regras da rede programática de universo via setor yfinance (ADR-001)."""

    use_yfinance_sector: bool
    core_sectors: tuple[str, ...]
    core_comm_industries: tuple[str, ...]
    core_comm_extra_tickers: tuple[str, ...]
    exclude_comm_industries: tuple[str, ...]


@dataclass(frozen=True)
class UniverseConfig:
    """Universo tech por lista explícita em camadas + blocos de sensibilidade.

    Ver ADR-001: núcleo (IT clássico + internet/mídia interativa) sempre dentro;
    blocos de sensibilidade rodam com/sem; deslistadas entram até a saída.
    """

    discovery: DiscoveryConfig
    core_it: tuple[str, ...]
    core_internet_media: tuple[str, ...]
    delisted_check: tuple[str, ...]
    ticker_aliases: Mapping[str, str]
    price_symbol_overrides: Mapping[str, str]
    sensitivity_blocks: Mapping[str, tuple[str, ...]]

    def curated_core(self) -> frozenset[str]:
        """Retorna o núcleo curado (IT clássico ∪ internet/mídia interativa).

        Estes tickers estão SEMPRE dentro do universo (independem de bloco de
        sensibilidade). Usado por :mod:`tonediv.data.universe` no passo de união.
        """
        return frozenset(self.core_it) | frozenset(self.core_internet_media)

    def sensitivity_tickers(self) -> frozenset[str]:
        """Retorna a união de todos os tickers de blocos de sensibilidade.

        São REMOVIDOS do núcleo automático (regra de subtração, ADR-001) para
        permanecerem removíveis no teste com/sem cada bloco.
        """
        out: set[str] = set()
        for tickers in self.sensitivity_blocks.values():
            out.update(tickers)
        return frozenset(out)


@dataclass(frozen=True)
class LoggingConfig:
    """Configuração de logging (regra nº 8: logging em biblioteca, não print)."""

    level: str
    format: str


@dataclass(frozen=True)
class Config:
    """Configuração tipada do projeto, carregada de ``config.yaml``.

    Attributes:
        root: Raiz do projeto (diretório onde vive o ``config.yaml``).
        seed: Semente global de determinismo (regra nº 6).
        paths: Caminhos resolvidos.
        sample: Janela amostral e fuso.
        data_sources: Fontes de dados.
        quality: Filtros de qualidade.
        universe: Definição do universo tech.
        logging: Configuração de logging.
        raw: Dicionário bruto completo do YAML (escape hatch para seções ainda
            não modeladas por dataclass — preenchidas nas fases seguintes).
    """

    root: Path
    seed: int
    paths: PathsConfig
    sample: SampleConfig
    data_sources: DataSourcesConfig
    quality: QualityConfig
    universe: UniverseConfig
    roles: RolesConfig
    finbert: FinbertConfig
    features: FeaturesConfig
    pit: PitConfig
    returns: ReturnsConfig
    event_study: EventStudyConfig
    strategy: StrategyConfig
    costs: CostsConfig
    metrics: MetricsConfig
    validation: ValidationConfig
    decay: DecayConfig
    logging: LoggingConfig
    raw: Mapping[str, Any]


# =============================================================================
# Construção a partir do dicionário YAML
# =============================================================================
def default_config_path() -> Path:
    """Retorna o caminho padrão do ``config.yaml`` (raiz do projeto).

    A raiz é inferida da posição deste arquivo (``src/tonediv/config.py``),
    subindo dois níveis, para não depender do diretório de trabalho atual.
    """
    return Path(__file__).resolve().parents[2] / "config.yaml"


def _build_paths(root: Path, raw: Mapping[str, Any]) -> PathsConfig:
    """Resolve os caminhos relativos do config para absolutos sob ``root``."""
    p = raw["paths"]
    return PathsConfig(
        root=root,
        data_raw=root / p["data_raw"],
        data_interim=root / p["data_interim"],
        data_processed=root / p["data_processed"],
        data_outputs=root / p["data_outputs"],
        docs=root / p["docs"],
    )


def _ticker_tuple(tickers: list[Any], field: str) -> tuple[str, ...]:
    """Normaliza uma lista de tickers para tupla de str, falhando em tokens não-str.

    Guarda contra a armadilha do YAML 1.1: barewords como ``ON``/``OFF``/``YES``/
    ``NO`` viram booleanos ao carregar. Se um ticker não citado for parseado como
    bool/None, falhamos ALTO aqui em vez de corromper silenciosamente a lista.
    """
    bad = [t for t in tickers if not isinstance(t, str)]
    if bad:
        raise ValueError(
            f"Universo '{field}' contém token não-textual {bad} — provável bareword "
            f"YAML (ON/OFF/YES/NO). Cite o ticker entre aspas no config.yaml."
        )
    return tuple(tickers)


def _build_universe(raw: Mapping[str, Any]) -> UniverseConfig:
    """Constrói :class:`UniverseConfig` normalizando listas para tuplas imutáveis."""
    u = raw["universe"]
    d = u["discovery"]
    discovery = DiscoveryConfig(
        use_yfinance_sector=bool(d["use_yfinance_sector"]),
        core_sectors=tuple(d["core_sectors"]),
        core_comm_industries=tuple(d["core_comm_industries"]),
        core_comm_extra_tickers=_ticker_tuple(
            d["core_comm_extra_tickers"], "core_comm_extra_tickers"
        ),
        exclude_comm_industries=tuple(d["exclude_comm_industries"]),
    )
    return UniverseConfig(
        discovery=discovery,
        core_it=_ticker_tuple(u["core_it"], "core_it"),
        core_internet_media=_ticker_tuple(u["core_internet_media"], "core_internet_media"),
        delisted_check=_ticker_tuple(u["delisted_check"], "delisted_check"),
        ticker_aliases=dict(u["ticker_aliases"]),
        price_symbol_overrides=dict(u.get("price_symbol_overrides", {})),
        sensitivity_blocks={k: _ticker_tuple(v, k) for k, v in u["sensitivity_blocks"].items()},
    )


def _build_config(root: Path, raw: Mapping[str, Any]) -> Config:
    """Monta o :class:`Config` completo a partir do dicionário YAML e da raiz."""
    s, ds, q, lg = raw["sample"], raw["data_sources"], raw["quality"], raw["logging"]
    return Config(
        root=root,
        seed=int(raw["seed"]),
        paths=_build_paths(root, raw),
        sample=SampleConfig(
            start_date=date.fromisoformat(s["start_date"]),
            end_date=date.fromisoformat(s["end_date"]),
            price_start_date=date.fromisoformat(s["price_start_date"]),
            timezone=s["timezone"],
        ),
        data_sources=DataSourcesConfig(
            transcripts_hf_dataset=ds["transcripts"]["hf_dataset"],
            transcripts_split=ds["transcripts"]["split"],
            transcripts_content_field=ds["transcripts"]["content_field"],
            prices_provider=ds["prices"]["provider"],
            market_ticker=ds["prices"]["market_ticker"],
            benchmarks=tuple(ds["prices"]["benchmarks"]),
            auto_adjust=bool(ds["prices"]["auto_adjust"]),
            max_retries=int(ds["prices"]["max_retries"]),
            retry_pause_seconds=float(ds["prices"]["retry_pause_seconds"]),
        ),
        quality=QualityConfig(
            min_chars_call=int(q["min_chars_call"]),
            min_chars_utterance=int(q["min_chars_utterance"]),
            min_utterances_call=int(q["min_utterances_call"]),
            report_dropped=bool(q["report_dropped"]),
        ),
        universe=_build_universe(raw),
        roles=_build_roles(raw),
        finbert=_build_finbert(raw),
        features=FeaturesConfig(
            peer_window_days=int(raw["features"]["peer_window_days"]),
            peer_min_calls=int(raw["features"]["peer_min_calls"]),
            idio_features=tuple(raw["features"]["idio_features"]),
            tone_distance_min_managers=int(raw["features"]["tone_distance"]["min_managers"]),
            tone_distance_min_manager_tokens=int(
                raw["features"]["tone_distance"]["min_manager_tokens"]
            ),
            clean_signal_confounders=tuple(raw["features"]["clean_signal"]["confounders"]),
            clean_signal_min_history=int(raw["features"]["clean_signal"]["min_history"]),
        ),
        pit=PitConfig(
            market_close_et=str(raw["pit"]["market_close_et"]),
            market_open_et=str(raw["pit"]["market_open_et"]),
            assumed_call_duration_minutes=int(raw["pit"]["assumed_call_duration_minutes"]),
            suspicious_times=tuple(raw["pit"]["suspicious_times"]),
        ),
        returns=ReturnsConfig(
            horizons=tuple(int(h) for h in raw["returns"]["horizons"]),
            entry=str(raw["returns"]["entry"]),
            exit=str(raw["returns"]["exit"]),
        ),
        event_study=EventStudyConfig(
            estimation_start=int(raw["event_study"]["estimation_start"]),
            estimation_end=int(raw["event_study"]["estimation_end"]),
            event_start=int(raw["event_study"]["event_start"]),
            event_end=int(raw["event_study"]["event_end"]),
            min_estimation_obs=int(raw["event_study"]["min_estimation_obs"]),
            groups=str(raw["event_study"]["groups"]),
            bootstrap_iters=int(raw["event_study"]["bootstrap_iters"]),
            placebo_fake_date_iters=int(raw["event_study"]["placebo"]["fake_date_iters"]),
            placebo_max_shift_days=int(raw["event_study"]["placebo"]["max_shift_days"]),
        ),
        strategy=_build_strategy(raw),
        costs=CostsConfig(bps_per_side=float(raw["costs"]["bps_per_side"])),
        metrics=MetricsConfig(
            trading_days_year=int(raw["metrics"]["trading_days_year"]),
            newey_west_lags=int(raw["metrics"]["newey_west_lags"]),
        ),
        validation=ValidationConfig(
            kfold_splits=int(raw["validation"]["purged_kfold"]["n_splits"]),
            embargo_days=int(raw["validation"]["purged_kfold"]["embargo_days"]),
            holdout_months=int(raw["validation"]["holdout_months"]),
            placebo_perm_iters=int(raw["validation"]["placebo_perm_iters"]),
            pbo_cscv_blocks=int(raw["validation"]["pbo_cscv_blocks"]),
            noise_pvalue_floor=float(raw["validation"]["noise_pvalue_floor"]),
            min_cell_events=int(raw["validation"]["min_cell_events"]),
            grid_features=tuple(raw["validation"]["grid_features"]),
        ),
        decay=DecayConfig(
            split_year=int(raw["decay"]["split_year"]),
            compare_metrics=tuple(raw["decay"]["compare_metrics"]),
        ),
        logging=LoggingConfig(level=lg["level"], format=lg["format"]),
        raw=raw,
    )


def _marker_tuple(
    value: Any,  # noqa: ANN401 - valor bruto do YAML; o tipo é justamente o que validamos
    field: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    """Valida uma lista de marcadores textuais do config (falha alta).

    Guardas (todas de achados da revisão multi-agente):
    - string ESCALAR em vez de lista (typo de esquecer o ``-`` no YAML) seria
      iterada caractere a caractere e viraria alternância ``q|u|e|...`` que
      casa com quase tudo — o bug crítico de Q&A prematuro reaberto por
      confusão de tipo;
    - item não-string (bareword YAML: ``ON``→bool, item nulo) explodiria só
      muito depois, com erro obscuro;
    - whitespace nas bordas faria o marcador nunca casar.
    """
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(
            f"roles.{field} deve ser uma LISTA de strings no config.yaml "
            f"(recebido: {type(value).__name__})."
        )
    bad = [m for m in value if not isinstance(m, str)]
    if bad:
        raise ValueError(f"roles.{field} contém itens não-textuais: {bad}.")
    markers = tuple(m.strip() for m in value if m.strip())
    if not markers and not allow_empty:
        raise ValueError(
            f"roles.{field} vazio no config.yaml — a detecção de Q&A ficaria "
            f"sem marcadores; defina ao menos um."
        )
    return markers


def _build_roles(raw: Mapping[str, Any]) -> RolesConfig:
    """Constrói :class:`RolesConfig` a partir da seção ``roles`` do YAML.

    Raises:
        ValueError: Se qualquer lista de marcadores for malformada (escalar,
            itens não-string) ou se os marcadores FORTES estiverem vazios —
            ver :func:`_marker_tuple`.
    """
    r = raw["roles"]
    v = r["validation"]
    return RolesConfig(
        operator_label=str(r["operator_label"]),
        qa_start_markers=_marker_tuple(
            r["qa_start_markers"], "qa_start_markers", allow_empty=False
        ),
        qa_start_markers_weak=_marker_tuple(
            r.get("qa_start_markers_weak", []), "qa_start_markers_weak", allow_empty=True
        ),
        qa_aviso_markers=_marker_tuple(
            r.get("qa_aviso_markers", []), "qa_aviso_markers", allow_empty=True
        ),
        analyst_intro_regex=str(r["analyst_intro_regex"]),
        validation_sample_size=int(v["sample_size"]),
        validation_seed=int(v["seed"]),
        validation_export_chars=int(v["export_utterance_chars"]),
    )


def _build_strategy(raw: Mapping[str, Any]) -> StrategyConfig:
    """Constrói :class:`StrategyConfig` com validação de fronteiras.

    Raises:
        ValueError: Se ``top_fraction`` não estiver em (0, 0.5) — com >= 0.5
            as zonas long e short do gatilho percentil se sobreporiam e o
            short silenciosamente sobrescreveria o long (achado da auditoria
            da Fase 4).
    """
    s = raw["strategy"]
    top = float(s["top_fraction"])
    if not 0.0 < top < 0.5:
        raise ValueError(
            f"strategy.top_fraction={top} inválido: deve estar em (0, 0.5) — "
            "com >= 0.5 as zonas long e short se sobrepõem."
        )
    direction = int(s["signal_direction"])
    if direction not in (-1, 1):
        raise ValueError(
            f"strategy.signal_direction={direction} inválido: deve ser +1 (comprar "
            "sinal alto) ou -1 (vender sinal alto)."
        )
    return StrategyConfig(
        holding_days=int(s["holding_days"]),
        variants=tuple(s["variants"]),
        signal_feature=str(s["signal_feature"]),
        signal_direction=direction,
        top_fraction=top,
        signal_rank_window_days=int(s["signal_rank_window_days"]),
        min_rank_history=int(s["min_rank_history"]),
        max_weight_per_name=float(s["max_weight_per_name"]),
        placebo_portfolio_perms=int(s["placebo_portfolio_perms"]),
        liquidity_adv_window_days=int(s["liquidity"]["adv_window_days"]),
        liquidity_min_adv_usd=float(s["liquidity"]["min_adv_usd"]),
    )


def _build_finbert(raw: Mapping[str, Any]) -> FinbertConfig:
    """Constrói :class:`FinbertConfig` a partir da seção ``finbert`` do YAML."""
    f = raw["finbert"]
    return FinbertConfig(
        model_name=str(f["model_name"]),
        max_tokens=int(f["max_tokens"]),
        batch_size=int(f["batch_size"]),
        device=str(f["device"]),
        checkpoint_shard_utterances=int(f["checkpoint_shard_utterances"]),
        sentinel_positive=str(f["sentinel_positive"]),
        sentinel_negative=str(f["sentinel_negative"]),
    )


def load_config(path: str | Path | None = None) -> Config:
    """Carrega e valida o ``config.yaml``, retornando um :class:`Config` tipado.

    Args:
        path: Caminho do YAML. Se ``None``, usa :func:`default_config_path`
            (o ``config.yaml`` na raiz do projeto).

    Returns:
        O :class:`Config` com todas as seções da Fase 1 modeladas e o restante
        acessível via ``.raw``.

    Raises:
        FileNotFoundError: Se o arquivo de config não existir.
    """
    cfg_path = Path(path) if path is not None else default_config_path()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.yaml não encontrado em: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    logger.debug("Config carregado de %s", cfg_path)
    return _build_config(cfg_path.parent, raw)


def configure_logging(cfg: LoggingConfig) -> None:
    """Aplica a configuração global de logging (nível e formato do YAML).

    Chamado uma vez no início de cada script (regra nº 8). Idempotente o
    suficiente para uso em scripts; usa ``force=True`` para reconfigurar handlers
    caso alguma dependência já tenha chamado ``basicConfig``.
    """
    logging.basicConfig(level=cfg.level, format=cfg.format, force=True)
