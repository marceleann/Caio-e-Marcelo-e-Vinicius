"""Features de tom por call, incl. versões ``_idio`` estritamente-passadas.

Por que existe:
    Converte as distribuições FinBERT por fala (script 03) nas features de
    call que o event study e a estratégia consomem.

    O SINAL PRINCIPAL do projeto é a Distância de Tom (Tone Distance, tese
    Angelo 2025), calculada por :func:`tone_distance_per_call`: a distância de
    tom ENTRE OS GESTORES da mesma call, ou seja, o quanto os executivos
    divergem de tom entre si (desacordo gerencial), medida com FinBERT como
    nossa modernização do dicionário Loughran-McDonald. A direção operacional é
    comprar distância alta (prêmio de risco pós-anúncio, ~1-3 meses), e o sinal
    de fato operado é o RESÍDUO estritamente-passado dessa distância sobre os
    confundidores (:func:`residualize_pit`).

    As demais features desta família (calculadas por :func:`aggregate_calls` e
    pelas funções de histórico/idiossincrasia abaixo) servem como controles,
    robustez e comparação — inclusive a antiga feature da tese Brockman, Li &
    Price (2015), ``mgmt_analyst_divergence``, mantida apenas como COMPARAÇÃO de
    legado após o pivô para a tese Angelo (ADR-021/022 superadas, ADR-023).

    Todas as decisões sutis de agregação moram aqui, documentadas:

    - **Agregação por call = média ponderada por TOKENS** das distribuições
      das falas (coerente com a agregação de chunks do scorer, ADR-015): uma
      resposta de 400 tokens carrega mais evidência de tom que um "yes" de 5.
    - **Ordem canônica** ``[p_neg, p_neu, p_pos]`` em todos os vetores.
    - As features restritas ao Q&A (``mgmt_analyst_divergence`` e afins) usam
      SOMENTE falas do Q&A — calls sem Q&A detectado ficam NaN nessa família (e
      são sinalizadas por ``qa_detected``), nunca contaminadas por papéis
      incertos.
    - **Versões ``_idio``** subtraem a média dos PARES em janela rolante de 90
      dias corridos ESTRITAMENTE anterior ao instante da call (ADR-004):
      nenhuma call futura entra na referência — nem calls do MESMO instante.

Features (nomes canônicos das colunas):
    Sinal principal (tese Angelo):
      - ``tone_distance``               distância de tom média entre gestores da call
    Controles/robustez/legado (comparação):
      1. ``net_tone``                     p_pos − p_neg agregado da call
      2. ``delta_tone``                   variação vs. call anterior da MESMA empresa
      3. ``tone_js_prev``                 distância JS vs. call anterior da MESMA empresa
      4. ``tone_dispersion``              desvio-padrão dos net_tones das falas
      5. ``mgmt_analyst_divergence``      JS(analistas_qa, gestão_qa) — feature antiga
                                          da tese Brockman, mantida como comparação
         ``mgmt_analyst_divergence_signed``  net_tone(gestão_qa) − net_tone(analistas_qa)
      6. ``qa_remarks_gap``               net_tone(gestão_remarks) − net_tone(gestão_qa)
      7. ``{f}_idio``                     f − média dos pares na janela PIT (f em 2, 3, 5)

Referências: Angelo (2025, Financial Review) — tese atual; Brockman, Li &
Price (2015, FAJ) — tese antiga superada; docs/DECISIONS.md ADR-004/013/015/023.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tonediv.config import FeaturesConfig

logger = logging.getLogger(__name__)

_DIST_COLS = ["p_neg", "p_neu", "p_pos"]


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Distância de Jensen-Shannon (base 2) entre duas distribuições.

    Raiz quadrada da divergência JS com log base 2 — métrica em ``[0, 1]``:
    0 = distribuições idênticas; 1 = suportes disjuntos. Escolhida sobre a
    divergência KL por ser simétrica e finita mesmo com zeros (ADR-013).

    Args:
        p: Vetor de probabilidades (soma 1).
        q: Vetor de probabilidades (soma 1).

    Returns:
        Distância JS em [0, 1]. NaN se alguma entrada tiver soma inválida.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if not (np.isfinite(p).all() and np.isfinite(q).all()):
        return float("nan")
    if abs(p.sum() - 1.0) > 1e-6 or abs(q.sum() - 1.0) > 1e-6:
        return float("nan")
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0  # 0*log(0/x) = 0 por convenção; b>=a/2 onde a>0, sem div/0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    # Erro numérico pode produzir -1e-17; a métrica é não-negativa por teoria.
    return float(np.sqrt(max(jsd, 0.0)))


def _weighted_dist(sub: pd.DataFrame) -> np.ndarray | None:
    """Distribuição agregada de um conjunto de falas (média ponderada por tokens).

    Retorna ``None`` para conjunto vazio — o chamador decide o NaN da feature.
    """
    if sub.empty:
        return None
    w = sub["n_tokens"].to_numpy(dtype=np.float64)
    probs = sub[_DIST_COLS].to_numpy(dtype=np.float64)
    total = w.sum()
    if total <= 0:
        return None
    return (probs * w[:, None]).sum(axis=0) / total


def _net(dist: np.ndarray | None) -> float:
    """net_tone (p_pos − p_neg) de uma distribuição agregada, ou NaN."""
    if dist is None:
        return float("nan")
    return float(dist[2] - dist[0])


def aggregate_calls(scores: pd.DataFrame) -> pd.DataFrame:
    """Agrega as falas pontuadas em UMA linha por call (features de comparação 1, 4, 5, 6).

    Produz as features de controle/robustez/legado da call (nível de tom,
    dispersão, e a antiga divergência analistas×gestão da tese Brockman, mantida
    como comparação). O sinal principal da tese Angelo (a distância de tom entre
    gestores) é calculado à parte, em :func:`tone_distance_per_call`.

    Args:
        scores: Saída do script 03 (``utterance_scores``): uma linha por fala
            com ``call_id, role, section, p_neg, p_neu, p_pos, net_tone,
            n_tokens``.

    Returns:
        DataFrame indexado por linha com colunas: ``call_id``, distribuição
        agregada da call (``p_neg/p_neu/p_pos``), ``net_tone``,
        ``tone_dispersion``, ``mgmt_analyst_divergence`` (+ ``_signed``),
        ``qa_remarks_gap``, ``qa_detected`` e contagens de falas por lado.
    """
    rows: list[dict[str, object]] = []
    for call_id, g in scores.groupby("call_id", sort=False):
        dist_all = _weighted_dist(g)
        mgmt_qa = g[(g["role"] == "management") & (g["section"] == "qa")]
        anal_qa = g[(g["role"] == "analyst") & (g["section"] == "qa")]
        mgmt_rem = g[(g["role"] == "management") & (g["section"] == "remarks")]
        d_mgmt_qa, d_anal_qa = _weighted_dist(mgmt_qa), _weighted_dist(anal_qa)
        d_mgmt_rem = _weighted_dist(mgmt_rem)

        divergence = (
            js_distance(d_anal_qa, d_mgmt_qa)
            if d_mgmt_qa is not None and d_anal_qa is not None
            else float("nan")
        )
        rows.append(
            {
                "call_id": call_id,
                "p_neg": dist_all[0] if dist_all is not None else float("nan"),
                "p_neu": dist_all[1] if dist_all is not None else float("nan"),
                "p_pos": dist_all[2] if dist_all is not None else float("nan"),
                "net_tone": _net(dist_all),
                # ddof=0: desvio-padrão populacional — a call é o universo
                # inteiro de falas dela, não uma amostra de algo maior.
                "tone_dispersion": float(g["net_tone"].std(ddof=0)) if len(g) > 1 else float("nan"),
                "mgmt_analyst_divergence": divergence,
                "mgmt_analyst_divergence_signed": _net(d_mgmt_qa) - _net(d_anal_qa),
                "qa_remarks_gap": _net(d_mgmt_rem) - _net(d_mgmt_qa),
                "qa_detected": bool((g["section"] == "qa").any()),
                "n_utt_scored": int(len(g)),
                "n_utt_mgmt_qa": int(len(mgmt_qa)),
                "n_utt_analyst_qa": int(len(anal_qa)),
            }
        )
    return pd.DataFrame(rows)


def _manager_key(speaker: object) -> str:
    """Chave de identidade do gestor na call (reusa o casamento de nomes de roles).

    ``_match_key`` gera ``sobrenome|inicial``, unindo grafias do MESMO executivo
    (Mike/Michael, nome colado, cargo grudado). Nome não parseável → ``""``.
    """
    from tonediv.nlp.roles import _match_key

    try:
        key = _match_key(str(speaker))
        return key if isinstance(key, str) else ""
    except Exception:  # nome degenerado não deve derrubar o pipeline
        return ""


def tone_distance_per_call(
    scores: pd.DataFrame,
    fcfg: FeaturesConfig,
    role: str = "management",
    out_col: str = "tone_distance",
    count_col: str = "n_managers_td",
) -> pd.DataFrame:
    """Tone Distance (Angelo 2025) medido com FinBERT — divergência de tom ENTRE gestores.

    Réplica fiel da construção do Angelo, trocando o dicionário Loughran-McDonald
    pelo FinBERT (ADR-023): cada GESTOR que fala na call vira um ponto no plano
    ``(p_pos, p_neg)`` — a intensidade FinBERT agregada das falas dele, ponderada
    por tokens. Calcula-se a distância euclidiana de cada gestor até a MÉDIA dos
    gestores da call; o Tone Distance é a MÉDIA dessas distâncias. Mede o quanto
    os executivos divergem de tom entre si (desacordo gerencial).

    Requer a coluna ``speaker`` (anexada no script 04 a partir de
    ``utterances_roles``). Uma call só recebe Tone Distance se tiver
    ``>= tone_distance_min_managers`` gestores, cada um com
    ``>= tone_distance_min_manager_tokens`` tokens pontuados; caso contrário NaN.

    Args:
        scores: Falas pontuadas (script 03) COM ``speaker``: ``call_id, role,
            speaker, p_neg, p_pos, n_tokens``.
        fcfg: Configuração de features (limiares de gestores/tokens).

    Returns:
        DataFrame ``[call_id, tone_distance, n_managers_td]`` — uma linha por
        call com gestão pontuada; ``tone_distance`` NaN se < min_managers.
    """
    if "speaker" not in scores.columns:
        raise KeyError("tone_distance_per_call requer a coluna 'speaker' (merge de roles).")
    # Só calls com Q&A DETECTADO: sem Q&A, o rótulo de papel não é confiável (a
    # call inteira vira 'management', misturando analistas no conjunto de
    # gestores) — corrige a contaminação da distância de tom (fidelidade Angelo).
    qa_calls = scores.loc[scores["section"] == "qa", "call_id"].unique()
    mgmt = scores[(scores["role"] == role) & (scores["call_id"].isin(qa_calls))].copy()
    mgmt["_mkey"] = mgmt["speaker"].map(_manager_key)
    mgmt = mgmt[mgmt["_mkey"] != ""]  # oradores sem nome parseável ficam de fora
    tok = mgmt["n_tokens"].to_numpy(dtype=np.float64)
    mgmt["_wpos"] = mgmt["p_pos"].to_numpy(dtype=np.float64) * tok
    mgmt["_wneg"] = mgmt["p_neg"].to_numpy(dtype=np.float64) * tok
    per_mgr = (
        mgmt.groupby(["call_id", "_mkey"], sort=False)
        .agg(tok=("n_tokens", "sum"), wpos=("_wpos", "sum"), wneg=("_wneg", "sum"))
        .reset_index()
    )
    per_mgr = per_mgr[per_mgr["tok"] >= fcfg.tone_distance_min_manager_tokens]
    per_mgr["Pos"] = per_mgr["wpos"] / per_mgr["tok"]
    per_mgr["Neg"] = per_mgr["wneg"] / per_mgr["tok"]

    min_m = fcfg.tone_distance_min_managers
    rows: list[dict[str, object]] = []
    for call_id, sub in per_mgr.groupby("call_id", sort=False):
        n = int(len(sub))
        if n < min_m:
            td = float("nan")
        else:
            pos = sub["Pos"].to_numpy(dtype=np.float64)
            neg = sub["Neg"].to_numpy(dtype=np.float64)
            dist = np.sqrt((pos - pos.mean()) ** 2 + (neg - neg.mean()) ** 2)
            td = float(dist.mean())
        rows.append({"call_id": call_id, out_col: td, count_col: n})
    return pd.DataFrame(rows)


def residualize_pit(
    df: pd.DataFrame,
    target: str,
    confounders: list[str],
    min_history: int,
    date_col: str = "decision_date",
) -> pd.Series:
    """Resíduo ESTRITAMENTE PASSADO de ``target`` sobre ``confounders`` (anti-look-ahead).

    Para cada linha (ordenada por ``date_col``), ajusta uma regressão linear do
    ``target`` nos ``confounders`` usando SOMENTE as linhas ANTERIORES no tempo,
    e devolve o resíduo (valor real − previsto). É a forma point-in-time de
    "limpar" o sinal dos confundidores: nenhum coeficiente enxerga o presente ou
    o futuro. As primeiras ``min_history`` linhas (sem histórico suficiente) e as
    linhas com NaN em ``target``/``confounders`` ficam NaN.

    Contexto: a distância de tom é confundida por nível de tom, tom dos analistas,
    tamanho e comprimento da call; operar o resíduo (em vez do valor cru) é o que
    torna o sinal robusto dentro e fora da amostra (tese Angelo — ADR-023).

    Args:
        df: Painel com ``target``, ``date_col`` e ``confounders``.
        target: Coluna a residualizar (ex.: ``tone_distance``).
        confounders: Colunas explicativas contemporâneas.
        min_history: Nº mínimo de linhas passadas para ajustar a regressão.
        date_col: Coluna temporal que define "passado" (ex.: ``decision_date``).

    Returns:
        Série alinhada ao índice de ``df`` com o resíduo (NaN onde indefinido).
    """
    usable = df.dropna(subset=[target, date_col, *confounders]).sort_values(date_col)
    dates = usable[date_col].to_numpy()
    x = np.column_stack([np.ones(len(usable)), usable[confounders].to_numpy(dtype=np.float64)])
    y = usable[target].to_numpy(dtype=np.float64)
    resid = np.full(len(usable), np.nan)
    for i in range(len(usable)):
        # Corte ESTRITAMENTE anterior (searchsorted side="left"): calls do MESMO
        # instante de decisão NÃO entram no ajuste — nenhum coeficiente enxerga o
        # presente nem o futuro (mesmo padrão de _peer_mean_pit).
        hi = int(np.searchsorted(dates, dates[i], side="left"))
        if hi < min_history:
            continue
        beta, *_ = np.linalg.lstsq(x[:hi], y[:hi], rcond=None)
        resid[i] = y[i] - x[i] @ beta
    out = pd.Series(np.nan, index=df.index, dtype=np.float64)
    out.loc[usable.index] = resid
    return out


def add_history_features(features: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta ``delta_tone`` e ``tone_js_prev`` (vs. call anterior da empresa).

    Exige colunas ``ticker`` e ``call_datetime`` já anexadas (ticker CANÔNICO,
    com aliases aplicados na Fase 1 — sem isso a série quebraria na troca
    FB→META etc.). A "call anterior" é a imediatamente anterior NO TEMPO da
    mesma empresa; primeira call de cada empresa fica NaN.
    """
    out = features.sort_values(["ticker", "call_datetime"]).reset_index(drop=True)
    grouped = out.groupby("ticker", sort=False)
    out["delta_tone"] = out["net_tone"] - grouped["net_tone"].shift(1)

    prev_dist = grouped[_DIST_COLS].shift(1)
    js_vals = np.full(len(out), np.nan)
    cur = out[_DIST_COLS].to_numpy(dtype=np.float64)
    prv = prev_dist.to_numpy(dtype=np.float64)
    for i in range(len(out)):
        if np.isfinite(prv[i]).all() and np.isfinite(cur[i]).all():
            js_vals[i] = js_distance(cur[i], prv[i])
    out["tone_js_prev"] = js_vals
    return out


def _peer_mean_pit(
    times: np.ndarray, values: np.ndarray, tickers: np.ndarray, window_days: int, min_calls: int
) -> np.ndarray:
    """Média dos PARES (outras empresas) em janela rolante estritamente anterior.

    Vetorizado com somas acumuladas + busca binária (O(n log n)): para cada
    call i, a janela é ``[t_i − window, t_i)`` — o limite direito é ESTRITO,
    então calls do mesmo instante não entram (PIT rigoroso, ADR-004). A
    contribuição da própria empresa é subtraída via somas acumuladas por
    ticker. Janelas com menos de ``min_calls`` pares retornam NaN.

    A aritmética temporal usa ``datetime64``/``timedelta64`` NATIVOS — nunca
    inteiros crus: a resolução dos timestamps varia com a versão do pandas
    (ns no 2.x, µs no 3.x para strings), e assumir ns fazia a janela de 90
    dias virar 90.000 dias silenciosamente em um dos ambientes (bug pego em
    teste ao rodar a suíte nos DOIS interpretadores do projeto).

    Args:
        times: Instantes ``datetime64`` (qualquer resolução, ORDENADOS asc,
            mesma referência de fuso — usar UTC).
        values: Valores da feature alinhados a ``times`` (NaN ignorado).
        tickers: Ticker de cada call, alinhado.
        window_days: Largura da janela em dias corridos.
        min_calls: Mínimo de calls de pares na janela.

    Returns:
        Vetor de médias dos pares (NaN onde insuficiente).
    """
    valid = np.isfinite(values)
    vals0 = np.where(valid, values, 0.0)
    cum_v = np.concatenate([[0.0], np.cumsum(vals0)])
    cum_c = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])

    window = np.timedelta64(int(window_days), "D")  # promove p/ a resolução de `times`
    left = np.searchsorted(times, times - window, side="left")
    # side="left" no limite direito é OBRIGATÓRIO (janela ESTRITA t_j < t_i):
    # com side="right", a PRÓPRIA call e calls simultâneas de outros tickers
    # entram na "média dos pares" — 38% das calls reais compartilham timestamp
    # exato com outra empresa (achado CRÍTICO da revisão da Fase 3, reproduzido;
    # também mantém simetria com o `hi` da subtração own abaixo).
    right = np.searchsorted(times, times, side="left")

    win_sum = cum_v[right] - cum_v[left]
    win_cnt = cum_c[right] - cum_c[left]

    own_sum = np.zeros(len(times))
    own_cnt = np.zeros(len(times), dtype=np.int64)
    order = np.arange(len(times))
    for _, idx in pd.Series(order).groupby(pd.Series(tickers), sort=False):
        pos = idx.to_numpy()
        t_own = times[pos]
        v_own = vals0[pos]
        c_own = valid[pos].astype(np.int64)
        cv = np.concatenate([[0.0], np.cumsum(v_own)])
        cc = np.concatenate([[0], np.cumsum(c_own)])
        lo = np.searchsorted(t_own, times[pos] - window, side="left")
        hi = np.searchsorted(t_own, times[pos], side="left")
        own_sum[pos] = cv[hi] - cv[lo]
        own_cnt[pos] = cc[hi] - cc[lo]

    peer_sum = win_sum - own_sum
    peer_cnt = win_cnt - own_cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = peer_sum / peer_cnt
    mean[peer_cnt < min_calls] = np.nan
    return mean


def add_idio_features(features: pd.DataFrame, fcfg: FeaturesConfig) -> pd.DataFrame:
    """Acrescenta as versões ``_idio`` (feature − média dos pares na janela PIT).

    Remove a "maré do setor": parte do tom de uma call é humor coletivo do
    momento, não informação específica da empresa (ADR-004). A referência usa
    SOMENTE calls que já aconteceram — janela de 90 dias corridos até o
    instante da call, exclusivo — e exclui a própria empresa.

    Args:
        features: Saída de :func:`add_history_features` (ordenada por ticker/
            data; será reordenada por data internamente).
        fcfg: Configuração (janela, mínimo de pares, lista de features).

    Returns:
        O DataFrame com colunas ``{f}_idio`` para cada ``f`` em
        ``fcfg.idio_features``.
    """
    out = features.sort_values("call_datetime").reset_index(drop=True)
    # UTC + datetime64 nativo: comparações unit-agnostic (ver _peer_mean_pit).
    times = out["call_datetime"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    tickers = out["ticker"].to_numpy()
    for feat in fcfg.idio_features:
        if feat not in out.columns:
            raise KeyError(f"Feature '{feat}' de idio_features não existe nas colunas.")
        peer = _peer_mean_pit(
            times,
            out[feat].to_numpy(dtype=np.float64),
            tickers,
            fcfg.peer_window_days,
            fcfg.peer_min_calls,
        )
        out[f"{feat}_idio"] = out[feat].to_numpy() - peer
    logger.info(
        "Features _idio calculadas para %s (janela=%dd, min_pares=%d).",
        list(fcfg.idio_features),
        fcfg.peer_window_days,
        fcfg.peer_min_calls,
    )
    return out
