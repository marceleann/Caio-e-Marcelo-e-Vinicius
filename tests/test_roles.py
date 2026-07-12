"""Testes da heurística de papéis (Fase 2) com uma mini-call construída à mão."""

import pandas as pd
import pytest

from tonediv.config import load_config
from tonediv.nlp.roles import coverage_metrics, infer_roles, validation_sample

CFG = load_config()

# Fala de ABERTURA padrão de operadores reais: contém o marcador de Q&A como
# AVISO ("Later, we will conduct..."). É o cenário do bug crítico pego na
# revisão multi-agente — o Q&A não pode ser detectado aqui.
_OPENING_BOILERPLATE = (
    "Good day, and welcome to the Apple earnings call. All participants will "
    "be in a listen-only mode. Later, we will conduct a question-and-answer "
    "session and instructions will follow at that time."
)


def _mini_call(call_id="AAPL_2020Q1"):
    # Estrutura canônica (com o boilerplate realista): Operator abre citando o
    # Q&A como aviso; CEO/CFO falam nos remarks; Operator anuncia o Q&A de
    # verdade; analista NOVO pergunta; CFO responde; Operator encerra.
    rows = [
        ("Operator", _OPENING_BOILERPLATE),
        ("Tim Cook", "Thank you. Revenue grew strongly this quarter across all segments."),
        (
            "Luca Maestri",
            "Gross margin expanded and we returned capital to shareholders. "
            "Operating cash flow reached an all-time record for the December "
            "quarter, driven by strength across our services portfolio and "
            "continued discipline in operating expenses throughout the period.",
        ),
        (
            "Operator",
            "We will now begin the question-and-answer session. "
            "The first question comes from Katy Huberty with Morgan Stanley.",
        ),
        ("Katy Huberty", "Can you talk about iPhone demand trends going forward?"),
        ("Luca Maestri", "Demand remains healthy despite the macro environment."),
        ("Operator", "Thank you. This concludes today's conference."),
    ]
    return pd.DataFrame(
        {
            "call_id": [call_id] * len(rows),
            "utterance_idx": range(len(rows)),
            "speaker": [r[0] for r in rows],
            "text": [r[1] for r in rows],
            "n_chars": [len(r[1]) for r in rows],
        }
    )


def _old_format_call(call_id="AAPL_2006Q1"):
    # Formato antigo (~2005-2010): Operator conduz por "next question" e os
    # oradores do Q&A vêm prefixados "Q -" (analista) / "A -" (gestão).
    rows = [
        ("Operator", "Good day and welcome to Apple's conference call today."),
        ("Peter Oppenheimer, CFO", "Revenue was strong with record gross margin this quarter."),
        ("Operator", "Ben Reitzes of UBS is our next question."),
        ("Q - Ben Reitzes", "Can you talk about gross margin trends into next quarter?"),
        ("A - Peter Oppenheimer", "Margins should stay healthy given our mix and cost discipline."),
        ("Operator", "The next question comes from Bill Shope of J.P. Morgan."),
        ("Q - Bill Shope", "What about international demand this period across regions?"),
        ("A - Tim Cook", "International demand is robust and growing across regions."),
    ]
    return pd.DataFrame(
        {
            "call_id": [call_id] * len(rows),
            "utterance_idx": range(len(rows)),
            "speaker": [r[0] for r in rows],
            "text": [r[1] for r in rows],
            "n_chars": [len(r[1]) for r in rows],
        }
    )


def test_old_format_next_question_and_qa_prefix():
    # Recuperação de calls antigas (ADR-025): o marcador "our next question"
    # abre o Q&A e o prefixo Q-/A- dá o papel EXPLÍCITO.
    out = infer_roles(_old_format_call(), CFG.roles).set_index("utterance_idx")
    assert (out["section"] == "qa").any()  # Q&A detectado (não ficou tudo remarks)
    assert out.loc[1, "role"] == "management"  # CFO nos remarks
    assert out.loc[3, "role"] == "analyst"  # "Q - Ben Reitzes"
    assert out.loc[6, "role"] == "analyst"  # "Q - Bill Shope"
    # "A - Tim Cook" só aparece no Q&A: a heurística posicional o rotularia
    # analista, mas o prefixo "A -" corrige para GESTÃO.
    assert out.loc[7, "role"] == "management"


def test_sections_split_at_qa_marker():
    out = infer_roles(_mini_call(), CFG.roles)
    # Falas 0-2 são remarks; da fala 3 (marcador do Operator) em diante é Q&A.
    assert list(out["section"]) == ["remarks"] * 3 + ["qa"] * 4


def test_roles_management_analyst_operator():
    out = infer_roles(_mini_call(), CFG.roles)
    roles = dict(zip(out["speaker"], out["role"], strict=False))
    assert roles["Tim Cook"] == "management"  # falou antes do Q&A
    assert roles["Luca Maestri"] == "management"  # continua gestão no Q&A
    assert roles["Katy Huberty"] == "analyst"  # estreia no Q&A
    assert roles["Operator"] == "operator"  # neutro


def test_analyst_confirmed_by_operator_announcement():
    out = infer_roles(_mini_call(), CFG.roles)
    katy = out[out["speaker"] == "Katy Huberty"].iloc[0]
    assert bool(katy["operator_confirmed"])  # anunciada: "from Katy Huberty with Morgan Stanley"


def test_no_qa_marker_means_all_remarks_no_analysts():
    df = _mini_call()
    # Remove a fala do Operator com o marcador -> Q&A não detectável.
    df = df[df["utterance_idx"] != 3].reset_index(drop=True)
    df["utterance_idx"] = range(len(df))
    out = infer_roles(df, CFG.roles)
    assert (out["section"] == "remarks").all()
    # Conservador: sem Q&A, ninguém vira analista (Katy fica gestão, pois "falou
    # antes do Q&A" — inexistente — é o único critério restante).
    assert "analyst" not in set(out["role"])


def test_null_speaker_is_unknown():
    df = _mini_call()
    df.loc[4, "speaker"] = None  # a analista perde o nome
    out = infer_roles(df, CFG.roles)
    assert out.loc[4, "role"] == "unknown"


def test_coverage_metrics_counts():
    cov = coverage_metrics(infer_roles(_mini_call(), CFG.roles))
    assert cov.n_calls == 1
    assert cov.pct_calls_qa_detected == 1.0
    assert cov.n_management == 3 and cov.n_analyst == 1 and cov.n_operator == 3
    assert cov.pct_utterances_classified == 1.0  # todas as não-Operator têm papel


def test_validation_sample_deterministic_and_truncated():
    calls = pd.concat(
        [_mini_call(f"TICK{i}_2020Q{i % 4 + 1}") for i in range(30)], ignore_index=True
    )
    annotated = infer_roles(calls, CFG.roles)
    s1, s2 = validation_sample(annotated, CFG.roles), validation_sample(annotated, CFG.roles)
    pd.testing.assert_frame_equal(s1, s2)  # seed fixa -> amostra idêntica
    assert s1["call_id"].nunique() == CFG.roles.validation_sample_size
    # A fala longa do Luca (fixture) excede o limite -> truncamento é exercido de fato.
    assert (s1["primeira_fala_truncada"].str.len() <= CFG.roles.validation_export_chars).all()
    assert (s1["primeira_fala_truncada"].str.len() == CFG.roles.validation_export_chars).any()


# ---------------------------------------------------------------------------
# Casos adicionados após a revisão multi-agente da Fase 2
# ---------------------------------------------------------------------------
def test_opening_boilerplate_does_not_trigger_qa():
    """O bug crítico da revisão: aviso de Q&A na ABERTURA não abre o Q&A."""
    out = infer_roles(_mini_call(), CFG.roles)
    # A fala 0 (boilerplate com "question-and-answer session") é remarks.
    assert out.loc[0, "section"] == "remarks"
    # E a gestão NÃO vira analista (o sintoma do bug era mgmt=0).
    roles = dict(zip(out["speaker"], out["role"], strict=False))
    assert roles["Tim Cook"] == "management"
    assert coverage_metrics(out).n_management == 3


def test_marker_in_management_speech_does_not_trigger_qa():
    df = _mini_call()
    # CEO menciona o jargão do Q&A na própria fala (fala 1, remarks).
    df.loc[1, "text"] = (
        "Thank you. Later we will conduct a question-and-answer session, but "
        "first let me review the quarter results in detail."
    )
    df.loc[1, "n_chars"] = len(df.loc[1, "text"])
    out = infer_roles(df, CFG.roles)
    # O marcador na fala da GESTÃO não abre o Q&A; o corte segue na fala 3.
    assert list(out["section"]) == ["remarks"] * 3 + ["qa"] * 4


def test_roles_are_isolated_per_call():
    # John Smith é ANALISTA na call A (estreia no Q&A)...
    call_a = _mini_call("AAA_2020Q1")
    call_a.loc[4, "speaker"] = "John Smith"
    # ...e GESTÃO na call B (fala nos remarks).
    call_b = _mini_call("BBB_2020Q1")
    call_b.loc[1, "speaker"] = "John Smith"
    both = pd.concat([call_a, call_b], ignore_index=True)
    out = infer_roles(both, CFG.roles)
    smith = out[out["speaker"] == "John Smith"]
    assert set(smith.loc[smith["call_id"] == "AAA_2020Q1", "role"]) == {"analyst"}
    assert set(smith.loc[smith["call_id"] == "BBB_2020Q1", "role"]) == {"management"}


def test_management_is_not_operator_confirmed():
    out = infer_roles(_mini_call(), CFG.roles)
    mgmt = out[out["role"] == "management"]
    assert not mgmt["operator_confirmed"].any()  # regressão "confirma todo mundo"


def test_empty_input_raises_clear_error():
    with pytest.raises(ValueError, match="vazia"):
        infer_roles(_mini_call().iloc[0:0], CFG.roles)


def test_duplicate_index_raises_clear_error():
    df = pd.concat([_mini_call(), _mini_call("MSFT_2020Q1")])  # índice repetido
    with pytest.raises(ValueError, match="[Íí]ndice duplicado"):
        infer_roles(df, CFG.roles)


def test_null_call_id_raises_clear_error():
    # Rejeição explícita (2ª rodada da revisão): agrupar call_id nulo fundiria
    # calls distintas num pseudo-call e quebraria a validation_sample depois.
    df = _mini_call()
    df.loc[6, "call_id"] = None
    with pytest.raises(ValueError, match="call_id nulo"):
        infer_roles(df, CFG.roles)


def test_result_invariant_under_row_permutation():
    df = _mini_call().sample(frac=1.0, random_state=7)  # embaralha as linhas
    out = infer_roles(df, CFG.roles).sort_values("utterance_idx").reset_index(drop=True)
    ref = infer_roles(_mini_call(), CFG.roles)
    pd.testing.assert_series_equal(out["role"], ref["role"])
    pd.testing.assert_series_equal(out["section"], ref["section"])


def test_empty_qa_markers_rejected_at_config_load():
    from tonediv.config import _build_roles

    raw = dict(load_config().raw)
    roles_raw = {**raw["roles"], "qa_start_markers": []}
    with pytest.raises(ValueError, match="qa_start_markers"):
        _build_roles({**raw, "roles": roles_raw})


def test_scalar_string_markers_rejected_at_config_load():
    # Typo real de YAML (esquecer o "-"): string escalar iteraria caractere a
    # caractere e viraria alternância "q|u|e|..." que casa com quase tudo.
    from tonediv.config import _build_roles

    raw = dict(load_config().raw)
    roles_raw = {**raw["roles"], "qa_start_markers": "question-and-answer session"}
    with pytest.raises(ValueError, match="LISTA"):
        _build_roles({**raw, "roles": roles_raw})


def test_non_string_marker_item_rejected_at_config_load():
    from tonediv.config import _build_roles

    raw = dict(load_config().raw)
    roles_raw = {**raw["roles"], "qa_start_markers": ["first question comes from", True, None]}
    with pytest.raises(ValueError, match="não-textuais"):
        _build_roles({**raw, "roles": roles_raw})


# ---------------------------------------------------------------------------
# Regressões da 2ª rodada da revisão (buracos REPRODUZIDOS pelos céticos)
# ---------------------------------------------------------------------------
def test_ir_host_before_boilerplate_does_not_trigger_qa():
    """Buraco A: host de RI fala ANTES do boilerplate; o aviso não pode abrir o Q&A."""
    df = _mini_call()
    host = pd.DataFrame(
        {
            "call_id": ["AAPL_2020Q1"],
            "utterance_idx": [-1],  # antes de todo mundo; reordenado no infer
            "speaker": ["Nancy Paxton"],
            "text": ["Good afternoon, and thank you for joining us today."],
            "n_chars": [51],
        }
    )
    df = pd.concat([host, df], ignore_index=True)
    out = infer_roles(df, CFG.roles)
    # O boilerplate (agora fala 1 na ordem) continua remarks; gestão intacta.
    boiler = out[out["text"].str.contains("listen-only", na=False)].iloc[0]
    assert boiler["section"] == "remarks"
    roles = dict(zip(out["speaker"], out["role"], strict=False))
    assert roles["Tim Cook"] == "management"
    assert roles["Nancy Paxton"] == "management"  # falou antes do Q&A


def test_null_speaker_before_boilerplate_does_not_trigger_qa():
    """Buraco A2: linha com speaker nulo antes do boilerplate não conta como 'gestão já falou'."""
    df = _mini_call()
    ghost = pd.DataFrame(
        {
            "call_id": ["AAPL_2020Q1"],
            "utterance_idx": [-1],
            "speaker": [None],
            "text": ["[Technical difficulty]"],
            "n_chars": [22],
        }
    )
    df = pd.concat([ghost, df], ignore_index=True)
    out = infer_roles(df, CFG.roles)
    roles = dict(zip(out["speaker"], out["role"], strict=False))
    assert roles["Tim Cook"] == "management"  # sintoma do bug seria 'analyst'
    assert coverage_metrics(out).n_management == 3


# ---------------------------------------------------------------------------
# Regressões da 3ª rodada (grafia divergente do MESMO orador — medida na base:
# ~130 calls com gestão convertida em analista; descarte derrubado 3x0)
# ---------------------------------------------------------------------------
def test_glued_name_in_qa_still_management():
    """Artefato sistemático do dataset: Q&A gruda o nome ("MarcBenioff")."""
    df = _mini_call()
    df.loc[5, "speaker"] = "LucaMaestri"  # mesma pessoa, colada, no Q&A
    out = infer_roles(df, CFG.roles)
    assert out.loc[5, "role"] == "management"


def test_nickname_and_middle_initial_variants_still_management():
    df = _mini_call()
    df.loc[1, "speaker"] = "Timothy D. Cook"  # remarks: nome formal c/ inicial
    extra = pd.DataFrame(
        {
            "call_id": ["AAPL_2020Q1"],
            "utterance_idx": [6],
            "speaker": ["Tim Cook"],  # Q&A: apelido, sem inicial
            "text": ["Let me add some color on that point about demand."],
            "n_chars": [50],
        }
    )
    df.loc[6, "utterance_idx"] = 7  # empurra o encerramento do Operator
    df = pd.concat([df, extra], ignore_index=True)
    out = infer_roles(df, CFG.roles)
    tim_qa = out[(out["speaker"] == "Tim Cook") & (out["section"] == "qa")]
    assert set(tim_qa["role"]) == {"management"}  # sobrenome+inicial pareiam


def test_title_suffix_stripped_for_matching():
    df = _mini_call()
    df.loc[1, "speaker"] = "Tim Cook - Chief Executive Officer"
    out = infer_roles(df, CFG.roles)
    assert out.loc[1, "role"] == "management"
    # E a Katy continua analista (sufixo não afeta quem nunca falou nos remarks).
    assert out.loc[4, "role"] == "analyst"


def test_distinct_analysts_are_not_merged():
    # Segurança da chave sobrenome|inicial: pessoas DIFERENTES não pareiam.
    df = _mini_call()
    extra = pd.DataFrame(
        {
            "call_id": ["AAPL_2020Q1"],
            "utterance_idx": [6],
            "speaker": ["Jim Suva"],  # outro analista, sobrenome/inicial distintos
            "text": ["A follow-up on gross margins if I may, please."],
            "n_chars": [47],
        }
    )
    df.loc[6, "utterance_idx"] = 7
    df = pd.concat([df, extra], ignore_index=True)
    out = infer_roles(df, CFG.roles)
    roles = dict(zip(out["speaker"], out["role"], strict=False))
    assert roles["Jim Suva"] == "analyst"
    assert roles["Katy Huberty"] == "analyst"


def test_qa_only_call_strong_marker_fires_at_position_zero():
    """Buraco B: transcrição só-Q&A — marcador FORTE abre o Q&A mesmo na fala 0."""
    rows = [
        (
            "Operator",
            "We will now begin the question-and-answer session. "
            "The first question comes from Katy Huberty with Morgan Stanley.",
        ),
        ("Katy Huberty", "Can you talk about services growth this quarter please?"),
        ("Luca Maestri", "Services set an all-time revenue record again this quarter."),
    ]
    df = pd.DataFrame(
        {
            "call_id": ["AAPL_2021Q1"] * len(rows),
            "utterance_idx": range(len(rows)),
            "speaker": [r[0] for r in rows],
            "text": [r[1] for r in rows],
            "n_chars": [len(r[1]) for r in rows],
        }
    )
    out = infer_roles(df, CFG.roles)
    assert (out["section"] == "qa").all()
    roles = dict(zip(out["speaker"], out["role"], strict=False))
    # Sem remarks, todo não-Operator é analista por estreia no Q&A — inclusive
    # o CFO (limitação conhecida da regra 3, documentada; melhor que o inverso).
    assert roles["Katy Huberty"] == "analyst"
