"""Testes das features (Fase 3) — todos os valores verificáveis no papel."""

import numpy as np
import pandas as pd
import pytest

from tonediv.config import load_config
from tonediv.nlp.features import (
    add_history_features,
    add_idio_features,
    aggregate_calls,
    js_distance,
    residualize_pit,
)

CFG = load_config()


# ---------------------------------------------------------------------------
# js_distance — propriedades e valor calculado à mão
# ---------------------------------------------------------------------------
def test_js_identical_is_zero():
    p = np.array([0.2, 0.3, 0.5])
    assert js_distance(p, p) == pytest.approx(0.0, abs=1e-12)


def test_js_disjoint_is_one():
    assert js_distance(np.array([1.0, 0, 0]), np.array([0, 0, 1.0])) == pytest.approx(1.0)


def test_js_symmetric():
    p, q = np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.3, 0.6])
    assert js_distance(p, q) == pytest.approx(js_distance(q, p))


def test_js_hand_computed_value():
    # P=[.5,.5,0], Q=[0,.5,.5]; M=[.25,.5,.25]
    # KL(P||M) = .5*log2(2) + .5*log2(1) = .5 ; KL(Q||M) idem
    # JSD = (.5+.5)/2 = .5 ; distância = sqrt(.5) ≈ 0.70710678
    p, q = np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.5, 0.5])
    assert js_distance(p, q) == pytest.approx(np.sqrt(0.5), abs=1e-9)


def test_js_invalid_input_is_nan():
    assert np.isnan(js_distance(np.array([0.5, 0.5, 0.5]), np.array([1.0, 0, 0])))


# ---------------------------------------------------------------------------
# aggregate_calls — agregação ponderada e features de comparação calculadas à
# mão. A divergência de tom analistas×gestão (mgmt_analyst_divergence, distância
# de Jensen-Shannon) é a feature ANTIGA da tese Brockman, Li & Price (2015),
# mantida apenas como comparação de legado após o pivô para a tese Angelo — NÃO
# é o sinal central. A tese atual é a distância de tom ENTRE OS GESTORES
# (tone_distance), testada mais abaixo.
# ---------------------------------------------------------------------------
def _scores_one_call():
    # Call com 4 falas pontuadas:
    #  gestão remarks (peso 3): [0.1, 0.2, 0.7]  -> net +0.6
    #  gestão qa      (peso 1): [0.3, 0.4, 0.3]  -> net  0.0
    #  analista qa    (peso 1): [0.6, 0.3, 0.1]  -> net -0.5
    #  analista qa    (peso 1): [0.4, 0.5, 0.1]  -> net -0.3
    return pd.DataFrame(
        {
            "call_id": ["C1"] * 4,
            "utterance_idx": [0, 1, 2, 3],
            "role": ["management", "management", "analyst", "analyst"],
            "section": ["remarks", "qa", "qa", "qa"],
            "p_neg": [0.1, 0.3, 0.6, 0.4],
            "p_neu": [0.2, 0.4, 0.3, 0.5],
            "p_pos": [0.7, 0.3, 0.1, 0.1],
            "net_tone": [0.6, 0.0, -0.5, -0.3],
            "n_tokens": [3, 1, 1, 1],
        }
    )


def test_aggregate_weighted_net_tone_by_hand():
    agg = aggregate_calls(_scores_one_call()).iloc[0]
    # dist_all = (3*[.1,.2,.7] + [.3,.4,.3] + [.6,.3,.1] + [.4,.5,.1]) / 6
    #          = [1.6/6, 1.8/6, 2.6/6] -> net = (2.6-1.6)/6 = 1/6
    assert agg["net_tone"] == pytest.approx(1 / 6, abs=1e-12)


def test_aggregate_divergence_signed_by_hand():
    agg = aggregate_calls(_scores_one_call()).iloc[0]
    # gestão_qa = [.3,.4,.3] (net 0.0); analistas_qa = média de peso igual
    # ([.6,.3,.1]+[.4,.5,.1])/2 = [.5,.4,.1] (net -0.4)
    assert agg["mgmt_analyst_divergence_signed"] == pytest.approx(0.0 - (-0.4), abs=1e-12)
    # Divergência JS > 0 (distribuições distintas) e < 1
    assert 0 < agg["mgmt_analyst_divergence"] < 1
    assert bool(agg["qa_detected"])


def test_aggregate_qa_remarks_gap_by_hand():
    agg = aggregate_calls(_scores_one_call()).iloc[0]
    # net(gestão remarks)=0.6 ; net(gestão qa)=0.0 -> gap = 0.6
    assert agg["qa_remarks_gap"] == pytest.approx(0.6, abs=1e-12)


def test_aggregate_call_without_qa_gives_nan_divergence():
    df = _scores_one_call()
    df["section"] = "remarks"  # sem Q&A
    agg = aggregate_calls(df).iloc[0]
    assert np.isnan(agg["mgmt_analyst_divergence"])
    assert not bool(agg["qa_detected"])


def test_tone_dispersion_population_std():
    agg = aggregate_calls(_scores_one_call()).iloc[0]
    expected = float(np.std([0.6, 0.0, -0.5, -0.3]))  # ddof=0
    assert agg["tone_dispersion"] == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Histórico (delta / js_prev) por empresa
# ---------------------------------------------------------------------------
def _features_panel():
    ts = pd.to_datetime(
        ["2020-01-10 17:00", "2020-04-10 17:00", "2020-02-01 17:00", "2020-05-01 17:00"]
    ).tz_localize("US/Eastern")
    return pd.DataFrame(
        {
            "call_id": ["A1", "A2", "B1", "B2"],
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "call_datetime": ts,
            "net_tone": [0.1, 0.4, -0.2, -0.2],
            "p_neg": [0.2, 0.1, 0.5, 0.5],
            "p_neu": [0.5, 0.4, 0.2, 0.2],
            "p_pos": [0.3, 0.5, 0.3, 0.3],
        }
    )


def test_delta_tone_within_ticker_only():
    out = add_history_features(_features_panel()).set_index("call_id")
    assert np.isnan(out.loc["A1", "delta_tone"])  # primeira call da empresa
    assert out.loc["A2", "delta_tone"] == pytest.approx(0.3)
    assert np.isnan(out.loc["B1", "delta_tone"])
    assert out.loc["B2", "delta_tone"] == pytest.approx(0.0)


def test_js_prev_zero_for_identical_consecutive_calls():
    out = add_history_features(_features_panel()).set_index("call_id")
    assert out.loc["B2", "tone_js_prev"] == pytest.approx(0.0, abs=1e-12)
    assert out.loc["A2", "tone_js_prev"] > 0


# ---------------------------------------------------------------------------
# _idio point-in-time — janela estrita, exclusão da própria empresa, mínimo
# ---------------------------------------------------------------------------
def _idio_cfg(min_calls=1):
    from tonediv.config import FeaturesConfig

    return FeaturesConfig(
        peer_window_days=90, peer_min_calls=min_calls, idio_features=("net_tone",)
    )


def _idio_panel():
    ts = pd.to_datetime(
        [
            "2020-01-01 10:00",  # P1 (peer)
            "2020-02-01 10:00",  # P2 (peer)
            "2020-03-01 10:00",  # ALVO
            "2020-03-01 10:00",  # mesmo instante do alvo (deve ficar FORA)
            "2020-06-15 10:00",  # futuro (deve ficar FORA)
        ]
    ).tz_localize("US/Eastern")
    return pd.DataFrame(
        {
            "call_id": ["P1", "P2", "T", "S", "F"],
            "ticker": ["X", "Y", "Z", "W", "V"],
            "call_datetime": ts,
            "net_tone": [0.2, 0.4, 0.9, 100.0, 100.0],
        }
    )


def test_idio_uses_only_strict_past_window():
    out = add_idio_features(_idio_panel(), _idio_cfg()).set_index("call_id")
    # Pares do alvo: P1 (0.2) e P2 (0.4) — média 0.3. S (mesmo instante) e F
    # (futuro) NÃO entram; se entrassem, o idio explodiria (valores 100).
    assert out.loc["T", "net_tone_idio"] == pytest.approx(0.9 - 0.3, abs=1e-12)


def test_idio_excludes_own_ticker():
    panel = _idio_panel()
    panel.loc[0, "ticker"] = "Z"  # P1 vira call da PRÓPRIA empresa do alvo
    out = add_idio_features(panel, _idio_cfg()).set_index("call_id")
    # Agora o único par é P2 (0.4).
    assert out.loc["T", "net_tone_idio"] == pytest.approx(0.9 - 0.4, abs=1e-12)


def test_idio_min_calls_yields_nan():
    out = add_idio_features(_idio_panel(), _idio_cfg(min_calls=3)).set_index("call_id")
    assert np.isnan(out.loc["T", "net_tone_idio"])  # só 2 pares na janela


def test_idio_left_edge_exactly_90d_is_included():
    # Janela [t−90d, t): fechada à ESQUERDA. Par exatamente 90 dias antes
    # (mesma hora) ENTRA — mata o mutante side="left"->"right" no limite esquerdo.
    panel = _idio_panel()
    target_ts = panel.loc[2, "call_datetime"]
    panel.loc[0, "call_datetime"] = target_ts - pd.Timedelta(days=90)
    out = add_idio_features(panel, _idio_cfg()).set_index("call_id")
    assert out.loc["T", "net_tone_idio"] == pytest.approx(0.9 - 0.3, abs=1e-12)  # P1 e P2


def test_idio_invariant_to_input_order():
    shuffled = _idio_panel().sample(frac=1.0, random_state=11)
    a = add_idio_features(_idio_panel(), _idio_cfg()).set_index("call_id")["net_tone_idio"]
    b = add_idio_features(shuffled, _idio_cfg()).set_index("call_id")["net_tone_idio"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_idio_window_excludes_old_calls():
    panel = _idio_panel()
    # P1 sai da janela de 90 dias (2019-11-01 -> 121 dias antes do alvo).
    panel.loc[0, "call_datetime"] = pd.Timestamp("2019-11-01 10:00", tz="US/Eastern")
    out = add_idio_features(panel, _idio_cfg()).set_index("call_id")
    assert out.loc["T", "net_tone_idio"] == pytest.approx(0.9 - 0.4, abs=1e-12)  # só P2


# ---------------------------------------------------------------------------
# residualize_pit — sinal LIMPO estritamente passado (tese Angelo, ADR-023)
# ---------------------------------------------------------------------------
def test_residualize_pit_strictly_past():
    # y = 2x exatamente. Com min_history=2, a partir da 3ª linha a regressão nas
    # linhas ANTERIORES recupera o beta e o resíduo é ~0. As 2 primeiras (sem
    # histórico) ficam NaN.
    df = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            ),
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 4.0, 6.0, 8.0],
        }
    )
    r = residualize_pit(df, "y", ["x"], min_history=2)
    assert np.isnan(r.iloc[0]) and np.isnan(r.iloc[1])  # sem histórico suficiente
    assert r.iloc[2] == pytest.approx(0.0, abs=1e-9)
    assert r.iloc[3] == pytest.approx(0.0, abs=1e-9)


def test_residualize_pit_does_not_see_future():
    # Se a última linha salta, o resíduo dela usa o ajuste do PASSADO (y=2x ->
    # prevê 8), dando resíduo 92 — NÃO ~0 (que seria se o ajuste 'visse' o salto).
    df = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            ),
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 4.0, 6.0, 100.0],  # salto na última
        }
    )
    r = residualize_pit(df, "y", ["x"], min_history=2)
    assert r.iloc[3] == pytest.approx(92.0, abs=1e-6)  # 100 − previsto(8)


# ---------------------------------------------------------------------------
# tone_distance_per_call — só em calls com Q&A (fidelidade Angelo)
# ---------------------------------------------------------------------------
def test_tone_distance_only_when_qa_detected():
    import dataclasses

    from tonediv.nlp.features import tone_distance_per_call

    # Sem Q&A detectado, a heurística rotula a call inteira como 'management'
    # (analistas incluídos) — a distância de tom NÃO deve ser computada, senão
    # mistura analistas no conjunto de gestores. Call A tem Q&A; call B não.
    fcfg = dataclasses.replace(
        CFG.features, tone_distance_min_managers=2, tone_distance_min_manager_tokens=0
    )
    scores = pd.DataFrame(
        {
            "call_id": ["A", "A", "B", "B"],
            "role": ["management"] * 4,
            "section": ["remarks", "qa", "remarks", "remarks"],
            "speaker": ["Alice Smith", "Bob Jones", "Carol Lee", "Dan Fox"],
            "p_pos": [0.8, 0.2, 0.7, 0.3],
            "p_neg": [0.1, 0.6, 0.2, 0.5],
            "n_tokens": [100, 100, 100, 100],
        }
    )
    out = tone_distance_per_call(scores, fcfg).set_index("call_id")
    assert "A" in out.index  # call com Q&A: distância computada
    assert "B" not in out.index  # call sem Q&A: excluída (não mistura papéis)
    assert out.loc["A", "tone_distance"] > 0
