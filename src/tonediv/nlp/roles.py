"""Inferência de papéis (gestão × analista) e seções (remarks × Q&A) por call.

Por que existe:
    O campo ``speaker`` do dataset traz o NOME do orador, não o papel — e o
    sinal principal (a Distância de Tom entre GESTORES no Q&A, tese Angelo 2025)
    exige isolar as falas de GESTÃO das de ANALISTA no Q&A: sem esse rótulo, os
    analistas contaminam o conjunto de gestores e distorcem a distância de tom.
    A separação também alimenta a antiga feature de comparação analistas×gestão
    (``mgmt_analyst_divergence``, legado da tese Brockman). Este módulo aplica a
    heurística em cascata do ADR-006:

    1. Detecta o início do Q&A por marcadores na fala do Operator
       (``qa_start_markers`` do config).
    2. Quem fala ANTES do Q&A (exceto Operator) = GESTÃO — em prepared remarks
       só a empresa fala; analistas não têm microfone antes do Q&A.
    3. No Q&A: Operator é neutro (descartado do sinal); quem já era gestão
       continua gestão; orador NOVO que estreia no Q&A = ANALISTA.
    4. Refinamento: nomes anunciados pelo Operator ("...comes from {nome} with
       {corretora}") são extraídos por regex e usados como CONFIRMAÇÃO.

    A validação é OBRIGATÓRIA (amostra manual com seed fixa + métricas de
    cobertura) porque a heurística é falível — calls sem marcador de Q&A ou com
    grafias divergentes do mesmo nome são degradadas de forma diagnosticável,
    nunca silenciosa.

Saída canônica: colunas ``section`` ∈ {remarks, qa} e ``role`` ∈
{management, analyst, operator, unknown} anexadas à tabela de falas.
Schema documentado em docs/data_schemas.md.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tonediv.config import RolesConfig

logger = logging.getLogger(__name__)

# Ordem canônica das colunas de saída (falas anotadas).
_OUT_COLS = ["section", "role", "operator_confirmed"]


@dataclass(frozen=True)
class RolesCoverage:
    """Métricas de cobertura da heurística (entram no relatório de qualidade).

    Attributes:
        n_calls: Nº de calls processadas.
        n_calls_qa_detected: Calls com início de Q&A detectado por marcador.
        pct_calls_qa_detected: Fração correspondente.
        n_utterances: Nº total de falas (inclui Operator).
        pct_utterances_classified: Fração de falas NÃO-Operator com papel
            atribuído (management/analyst) — o resto é ``unknown``.
        n_management: Falas classificadas como gestão.
        n_analyst: Falas classificadas como analista.
        n_operator: Falas do Operator (neutras, fora do sinal).
        n_unknown: Falas não-Operator sem papel (ex.: speaker nulo).
    """

    n_calls: int
    n_calls_qa_detected: int
    pct_calls_qa_detected: float
    n_utterances: int
    pct_utterances_classified: float
    n_management: int
    n_analyst: int
    n_operator: int
    n_unknown: int


# Prefixos de artefato de transcrição ("Q - ", "A - ") e sufixo de cargo/empresa
# (" - Chief Executive Officer") observados no dataset real.
_QA_PREFIX_RE = re.compile(r"^[QA]\s*[-–:]\s*")  # \s* (não \s+): pega prefixo colado "Q –Clarke"
_TITLE_SUFFIX_RE = re.compile(r"\s+[-–—]\s+.*$")
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
# Prefixo explícito de papel no formato antigo (~2005-2010): "Q - " = pergunta de
# analista; "A - " = resposta da gestão. Sinal direto de papel e de início do Q&A.
_Q_PREFIX_RE = re.compile(r"^Q\s*[-–:]")
_A_PREFIX_RE = re.compile(r"^A\s*[-–:]")


def _clean_name(name: object) -> str:
    """Limpa um nome de orador para forma canônica legível ("" = sem nome).

    Passos, cada um atacando uma classe de sujeira MEDIDA no dataset real
    (3ª rodada da revisão multi-agente — ~130 calls com gestão virando
    analista por grafia divergente do MESMO executivo entre seções):

    1. nulos pandas (``None``/``NaN``) viram "" — ``str(nan)`` seria nome válido;
    2. prefixo de artefato "Q - "/"A - " removido;
    3. sufixo de cargo/empresa após " - " removido ("Tim Cook - Apple CEO");
    4. nomes COLADOS separados por camel-case ("MarcBenioff" -> "Marc Benioff",
       artefato sistemático de um formato-fonte ~2020-21);
    5. acentos/mojibake normalizados (NFKD sem combinantes);
    6. casefold + colapso de espaços.
    """
    if name is None or not isinstance(name, str):
        if name is None or pd.isna(name):
            return ""
        name = str(name)
    name = _QA_PREFIX_RE.sub("", name.strip())
    name = _TITLE_SUFFIX_RE.sub("", name)
    name = _CAMEL_RE.sub(" ", name)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", name).strip().casefold()


def _normalize_name(name: object) -> str:
    """Chave de comparação EXATA de um orador (limpeza completa, sem fusão)."""
    return _clean_name(name)


def _match_key(name: object) -> str:
    """Chave de PAREAMENTO intra-call: ``sobrenome|inicial-do-primeiro-nome``.

    Por que existe (3ª rodada da revisão): a chave exata não iguala
    "Michael Splinter" ↔ "Mike Splinter" nem "Luca A. Maestri" ↔ "Luca
    Maestri" — variantes reais do MESMO executivo entre remarks e Q&A que
    convertiam gestão em analista. A chave sobrenome+inicial iguala essas
    variantes; iniciais do meio (tokens de 1 letra) são descartadas.

    O escopo é UMA call (mgmt_keys/announced são por call), então o risco de
    colisão — duas pessoas DISTINTAS com mesmo sobrenome e mesma inicial na
    MESMA call — é raro e aceito como trade-off documentado: o dano medido da
    chave estrita (~950 falas de gestão contaminando o sinal) é ordens de
    grandeza maior que o das colisões.
    """
    cleaned = _clean_name(name)
    if not cleaned:
        return ""
    tokens = [t for t in cleaned.replace(".", " ").split() if len(t) > 1]
    if not tokens:
        return cleaned
    return f"{tokens[-1]}|{tokens[0][0]}"


def _is_operator(speaker: str | None, operator_key: str) -> bool:
    """Testa se o orador é o Operator (comparação pela chave normalizada)."""
    return _normalize_name(speaker) == operator_key


def _find_qa_start(
    speakers: pd.Series,
    texts: pd.Series,
    operator_key: str,
    strong_re: re.Pattern[str],
    weak_re: re.Pattern[str] | None,
    aviso_re: re.Pattern[str] | None,
) -> int | None:
    """Localiza o índice posicional da fala do Operator que abre o Q&A.

    Design em 3 classes de marcadores, endurecido em DUAS rodadas de revisão
    multi-agente (cada regra abaixo fecha um buraco REPRODUZIDO):

    - Só falas do OPERATOR contam: os marcadores são jargão de moderação;
      procurá-los em falas da gestão geraria falsos positivos.
    - ``aviso_re`` (anti-padrão) veta a fala inteira: "...Later, we will
      conduct a question-and-answer session and instructions will follow..."
      é AVISO de abertura, não abertura — sem o veto, bastava um host de RI
      (ou uma linha com speaker nulo) falar antes do boilerplate para o bug
      crítico original ressurgir silenciosamente.
    - Marcadores FORTES ("first question comes from"...) são transição
      inequívoca e valem em QUALQUER posição — cobre transcrições só-Q&A, em
      que ninguém fala antes do Q&A real (falso negativo da 1ª correção).
    - Marcadores FRACOS ("question-and-answer session") são ambíguos: só
      valem após uma fala NOMEADA de não-Operator (chave normalizada não
      vazia — speaker NaN não conta como "gestão já falou").

    Returns:
        Índice posicional (0-based) da fala que abre o Q&A, ou ``None`` se
        nenhum marcador válido for encontrado (call sem Q&A detectável).
    """
    marker_pos: int | None = None
    seen_named_non_operator = False
    for pos, (spk, txt) in enumerate(zip(speakers, texts, strict=True)):
        key = _normalize_name(spk)
        if key != operator_key:
            if key:
                seen_named_non_operator = True
            continue
        text = str(txt) if txt is not None else ""
        if not text:
            continue
        if aviso_re is not None and aviso_re.search(text):
            continue  # fala é aviso: nunca abre o Q&A
        if strong_re.search(text):
            marker_pos = pos
            break
        if seen_named_non_operator and weak_re is not None and weak_re.search(text):
            marker_pos = pos
            break
    # Formato antigo: oradores prefixados "Q -"/"A -" — a 1ª ocorrência abre o Q&A
    # mesmo sem marcador do Operator (recupera calls ~2005-2010, ADR-025).
    prefix_pos: int | None = None
    for pos, spk in enumerate(speakers):
        if _QA_PREFIX_RE.match(str(spk) if spk is not None else ""):
            prefix_pos = pos
            break
    candidates = [c for c in (marker_pos, prefix_pos) if c is not None]
    return min(candidates) if candidates else None


def _extract_announced_analysts(
    texts_qa_operator: list[str], intro_re: re.Pattern[str]
) -> set[str]:
    """Extrai nomes de analistas anunciados pelo Operator no Q&A (normalizados).

    Serve de CONFIRMAÇÃO (refinamento nº 4 do ADR-006), não de fonte primária:
    a regra posicional (estreia no Q&A) decide; a announced-list marca a coluna
    ``operator_confirmed`` para auditoria e para a amostra de validação manual.
    """
    announced: set[str] = set()
    for text in texts_qa_operator:
        for match in intro_re.finditer(text):
            announced.add(_normalize_name(match.group(1)))
    return announced


def _classify_call(
    group: pd.DataFrame,
    rcfg: RolesConfig,
    strong_re: re.Pattern[str],
    weak_re: re.Pattern[str] | None,
    aviso_re: re.Pattern[str] | None,
    intro_re: re.Pattern[str],
) -> pd.DataFrame:
    """Aplica a heurística completa a UMA call e devolve as colunas de saída.

    Args:
        group: Falas de uma call, já ordenadas por ``utterance_idx``.
        rcfg: Configuração de papéis.
        strong_re: Regex dos marcadores FORTES de início de Q&A.
        weak_re: Regex dos marcadores FRACOS (ou ``None`` se não houver).
        aviso_re: Regex dos anti-padrões de aviso (ou ``None``).
        intro_re: Regex compilada das apresentações de analistas.

    Returns:
        DataFrame com colunas ``section``/``role``/``operator_confirmed``
        alinhado ao índice de ``group``.
    """
    operator_key = _normalize_name(rcfg.operator_label)
    speakers, texts = group["speaker"], group["text"]
    qa_start = _find_qa_start(speakers, texts, operator_key, strong_re, weak_re, aviso_re)

    n = len(group)
    positions = np.arange(n)
    in_qa = positions >= qa_start if qa_start is not None else np.zeros(n, dtype=bool)

    keys = speakers.map(_normalize_name).to_numpy()
    is_op = np.array([k == operator_key for k in keys])
    # Chave de PAREAMENTO (sobrenome|inicial) para continuidade de identidade
    # dentro da call — tolera "Mike/Michael", inicial do meio, nome colado etc.
    # (3ª rodada da revisão: ~130 calls tinham gestão virando analista por
    # grafia divergente do MESMO executivo entre remarks e Q&A).
    mkeys = speakers.map(_match_key).to_numpy()

    # Regra 2: gestão = quem fala antes do Q&A (exceto Operator e sem nome).
    mgmt_mkeys = {
        mk for mk, iq, op in zip(mkeys, in_qa, is_op, strict=True) if mk and not iq and not op
    }
    # Sem Q&A detectado, a call inteira é remarks -> ninguém vira analista (conservador).
    announced = (
        _extract_announced_analysts([str(t) for t in texts.to_numpy()[in_qa & is_op]], intro_re)
        if qa_start is not None
        else set()
    )
    announced_mkeys = {_match_key(a) for a in announced}

    roles: list[str] = []
    for key, mkey, op in zip(keys, mkeys, is_op, strict=True):
        if op:
            roles.append("operator")
        elif not key:
            roles.append("unknown")  # speaker nulo: não adivinhar
        elif mkey in mgmt_mkeys:
            roles.append("management")
        else:
            roles.append("analyst")  # regra 3: estreia no Q&A

    # Override por prefixo explícito "Q -"/"A -" (formato antigo): papel direto no
    # dado — Q = pergunta de analista, A = resposta da gestão (ADR-025). Sobrepõe
    # a heurística posicional, que erraria gestor que só responde no Q&A ("A - X").
    spk_arr = speakers.to_numpy()
    for i in range(len(roles)):
        if is_op[i]:
            continue
        s = str(spk_arr[i]) if spk_arr[i] is not None else ""
        if _Q_PREFIX_RE.match(s):
            roles[i] = "analyst"
        elif _A_PREFIX_RE.match(s):
            roles[i] = "management"

    out = pd.DataFrame(index=group.index)
    out["section"] = np.where(in_qa, "qa", "remarks")
    out["role"] = roles
    out["operator_confirmed"] = [bool(mk) and mk in announced_mkeys for mk in mkeys]
    return out


def infer_roles(utterances: pd.DataFrame, rcfg: RolesConfig) -> pd.DataFrame:
    """Anota a tabela de falas com seção e papel inferido (heurística ADR-006).

    Args:
        utterances: Tabela ``[call_id, utterance_idx, speaker, text, ...]``
            (saída da Fase 1). A ordem por ``utterance_idx`` é essencial.
        rcfg: Configuração de papéis (marcadores, regex, operator label).

    Returns:
        A tabela de entrada com três colunas novas: ``section`` (remarks|qa),
        ``role`` (management|analyst|operator|unknown) e ``operator_confirmed``
        (bool — analista também anunciado pelo Operator).

    Raises:
        ValueError: Se a entrada estiver vazia, tiver índice duplicado ou
            ``call_id`` nulo — mensagens EXPLÍCITAS em vez dos erros obscuros
            de pandas que essas condições produziriam a jusante. ``call_id``
            nulo é REJEITADO (não agrupado): linhas de calls distintas se
            fundiriam num pseudo-call com papéis confiantes e errados, e o
            ``validation_sample`` quebraria ao ordenar — melhor apontar o
            problema upstream (2ª rodada da revisão multi-agente).
    """
    if utterances.empty:
        raise ValueError(
            "Tabela de falas vazia — verifique os filtros de qualidade e a "
            "interseção universo ∩ dataset (script 01) antes de inferir papéis."
        )
    if not utterances.index.is_unique:
        raise ValueError("Índice duplicado na tabela de falas; use reset_index antes.")
    n_null_ids = int(utterances["call_id"].isna().sum())
    if n_null_ids:
        raise ValueError(
            f"{n_null_ids} fala(s) com call_id nulo — corrija a origem (script 01); "
            "agrupá-las fundiria calls distintas num pseudo-call com papéis errados."
        )

    def _compile(markers: tuple[str, ...]) -> re.Pattern[str] | None:
        if not markers:
            return None
        return re.compile("|".join(re.escape(m) for m in markers), re.IGNORECASE)

    strong_re = _compile(rcfg.qa_start_markers)
    if strong_re is None:  # inalcançável: config falha alto em lista vazia
        raise ValueError("qa_start_markers vazio — config deveria ter rejeitado.")
    weak_re = _compile(rcfg.qa_start_markers_weak)
    aviso_re = _compile(rcfg.qa_aviso_markers)
    intro_re = re.compile(rcfg.analyst_intro_regex)

    ordered = utterances.sort_values(["call_id", "utterance_idx"])
    parts = [
        _classify_call(group, rcfg, strong_re, weak_re, aviso_re, intro_re)
        for _, group in ordered.groupby("call_id", sort=False)
    ]
    annotated = pd.concat(parts).reindex(utterances.index)
    result = pd.concat([utterances, annotated[_OUT_COLS]], axis=1)
    logger.info(
        "Papéis inferidos: %d falas (%s).",
        len(result),
        result["role"].value_counts().to_dict(),
    )
    return result


def coverage_metrics(annotated: pd.DataFrame) -> RolesCoverage:
    """Calcula as métricas de cobertura exigidas pela validação (ADR-006).

    ``pct_calls_qa_detected`` usa a presença de QUALQUER fala com
    ``section == 'qa'`` como evidência de que o marcador foi encontrado.
    """
    n_calls = annotated["call_id"].nunique()
    qa_calls = annotated.loc[annotated["section"] == "qa", "call_id"].nunique()
    non_op = annotated[annotated["role"] != "operator"]
    classified = non_op["role"].isin(["management", "analyst"])
    counts = annotated["role"].value_counts()
    return RolesCoverage(
        n_calls=int(n_calls),
        n_calls_qa_detected=int(qa_calls),
        pct_calls_qa_detected=float(qa_calls / n_calls) if n_calls else 0.0,
        n_utterances=int(len(annotated)),
        pct_utterances_classified=float(classified.mean()) if len(non_op) else 0.0,
        n_management=int(counts.get("management", 0)),
        n_analyst=int(counts.get("analyst", 0)),
        n_operator=int(counts.get("operator", 0)),
        n_unknown=int(counts.get("unknown", 0)),
    )


def validation_sample(annotated: pd.DataFrame, rcfg: RolesConfig) -> pd.DataFrame:
    """Monta o CSV de conferência manual: 1ª fala de cada orador em N calls.

    Sorteia ``validation_sample_size`` calls com seed FIXA (determinismo,
    regra nº 6) e exporta uma linha por (call, orador) com o papel inferido e a
    primeira fala truncada — suficiente para um humano julgar o papel sem ler a
    transcrição inteira.

    Returns:
        DataFrame ``[call_id, speaker, papel_inferido, secao_primeira_fala,
        operator_confirmed, primeira_fala_truncada]`` ordenado por call/orador.
    """
    rng = np.random.default_rng(rcfg.validation_seed)
    call_ids = np.sort(annotated["call_id"].unique())
    size = min(rcfg.validation_sample_size, len(call_ids))
    chosen = rng.choice(call_ids, size=size, replace=False)

    sample = annotated[annotated["call_id"].isin(chosen)].sort_values(["call_id", "utterance_idx"])
    firsts = sample.groupby(["call_id", "speaker"], dropna=False, sort=True).first().reset_index()
    out = pd.DataFrame(
        {
            "call_id": firsts["call_id"],
            "speaker": firsts["speaker"],
            "papel_inferido": firsts["role"],
            "secao_primeira_fala": firsts["section"],
            "operator_confirmed": firsts["operator_confirmed"],
            "primeira_fala_truncada": firsts["text"].str.slice(0, rcfg.validation_export_chars),
        }
    )
    return out.sort_values(["call_id", "papel_inferido", "speaker"]).reset_index(drop=True)
