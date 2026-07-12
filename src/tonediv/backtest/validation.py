"""Validação temporal: PurgedKFold com embargo, placebo intra-janela e PBO/CSCV.

Por que existe:
    É a linha de defesa contra overfitting e falso positivo:

    - **PurgedKFold + embargo** (López de Prado): dobras temporais em que o
      treino PURGA observações cujo retorno futuro se sobrepõe ao teste, mais
      um embargo posterior — sem isso, posições sobrepostas de H pregões
      vazam informação entre dobras.
    - **Placebo intra-janela** (ADR-009): permuta o sinal DENTRO de cada
      janela cross-section (trimestre — a mesma janela do IC), preservando a
      estrutura temporal. Embaralhar o painel inteiro inflaria falsos
      positivos (armadilha nº 4).
    - **PBO via CSCV** (Bailey et al.): probabilidade de o campeão in-sample
      ficar abaixo da mediana out-of-sample — a métrica anti-cherry-picking
      da grade de robustez.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd

from tonediv.backtest.metrics import rank_ic

logger = logging.getLogger(__name__)


def purged_kfold_indices(
    times: pd.Series, n_splits: int, embargo_days: int, horizon_days: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Gera índices (treino, teste) de K dobras temporais purgadas.

    Cada dobra de teste é um bloco CONTÍGUO no tempo. Do treino são removidos:
    (a) eventos cujo horizonte de retorno [t, t+H] invade o bloco de teste
    (purga à esquerda) e (b) eventos nos ``embargo_days`` dias APÓS o bloco
    (embargo à direita — correlação serial residual).

    Args:
        times: Instantes dos eventos (ordenáveis; tz-aware ok).
        n_splits: Nº de dobras (config ``validation``).
        embargo_days: Dias corridos de embargo pós-teste.
        horizon_days: Horizonte de retorno em dias corridos (para a purga; use
            o horizonte H em pregões convertido com folga — ex.: H×2).

    Yields:
        Pares ``(idx_treino, idx_teste)`` como arrays de posições em ``times``.
    """
    order = np.argsort(times.to_numpy())
    t_sorted = times.to_numpy()[order]
    n = len(t_sorted)
    fold_edges = np.linspace(0, n, n_splits + 1, dtype=int)
    horizon = np.timedelta64(horizon_days, "D")
    embargo = np.timedelta64(embargo_days, "D")

    for k in range(n_splits):
        lo, hi = fold_edges[k], fold_edges[k + 1]
        if hi <= lo:
            continue
        test_pos = order[lo:hi]
        t_start, t_end = t_sorted[lo], t_sorted[hi - 1]
        train_mask = np.ones(n, dtype=bool)
        train_mask[lo:hi] = False
        # Purga: treino anterior cujo retorno [t, t+H] alcança o teste.
        before = np.arange(n) < lo
        reaches_test = t_sorted + horizon >= t_start
        train_mask &= ~(before & reaches_test)
        # Embargo: treino logo após o fim do teste.
        after = np.arange(n) >= hi
        inside_embargo = t_sorted <= t_end + embargo
        train_mask &= ~(after & inside_embargo)
        yield order[np.where(train_mask)[0]], test_pos


def placebo_within_window(
    events: pd.DataFrame,
    feature: str,
    ret_col: str,
    date_col: str,
    n_iter: int,
    rng: np.random.Generator,
    freq: str = "QE",
) -> tuple[float, np.ndarray]:
    """P-valor do IC observado contra permutações do sinal DENTRO de cada janela.

    A permutação embaralha a feature ENTRE eventos da mesma janela temporal
    (trimestre — a mesma cross-section do IC), mantendo datas e retornos no
    lugar: destrói a associação sinal→retorno preservando toda a estrutura
    temporal e as caudas de cada período (ADR-009).

    Returns:
        Tupla ``(p_valor_bicaudal, distribuição_nula_do_IC)``. O p usa a
        correção +1 (nunca zero exato).
    """
    df = events[[date_col, feature, ret_col]].dropna().copy()
    if len(df) < 10:
        return float("nan"), np.empty(0)
    observed = rank_ic(df[feature], df[ret_col])
    window = df[date_col].dt.to_period(freq.replace("E", ""))  # "QE"->"Q", "ME"->"M"

    null = np.empty(n_iter)
    values = df[feature].to_numpy().copy()
    codes = window.factorize()[0]
    for it in range(n_iter):
        permuted = values.copy()
        for g in np.unique(codes):
            mask = codes == g
            permuted[mask] = rng.permutation(permuted[mask])
        null[it] = rank_ic(pd.Series(permuted, index=df.index), df[ret_col])
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(observed):
        return float("nan"), null
    p = (1.0 + float(np.sum(np.abs(null) >= abs(observed)))) / (len(null) + 1.0)
    return p, null


def pbo_cscv(returns_matrix: pd.DataFrame, n_blocks: int) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey et al., 2017).

    Divide o tempo em ``n_blocks`` blocos contíguos; para cada combinação de
    metade dos blocos (treino), elege a estratégia de melhor Sharpe in-sample
    e mede seu rank relativo out-of-sample. PBO = fração das combinações em
    que o campeão fica ABAIXO da mediana OOS.

    Args:
        returns_matrix: DataFrame T×N (linhas = períodos, colunas = variantes
            da grade). NaN tratado como 0 (dia sem posição = caixa).
        n_blocks: Nº PAR de blocos (config; 16 → C(16,8)=12.870 combinações).

    Returns:
        PBO em [0, 1]. NaN se dados insuficientes (< 2 variantes ou blocos
        sem observações).
    """
    if returns_matrix.shape[1] < 2 or returns_matrix.shape[0] < n_blocks:
        return float("nan")
    m = returns_matrix.fillna(0.0).to_numpy()
    blocks = np.array_split(np.arange(m.shape[0]), n_blocks)
    half = n_blocks // 2
    below_median = 0
    total = 0
    for combo in itertools.combinations(range(n_blocks), half):
        train_rows = np.concatenate([blocks[b] for b in combo])
        test_rows = np.concatenate([blocks[b] for b in range(n_blocks) if b not in combo])
        mu_tr, sd_tr = m[train_rows].mean(axis=0), m[train_rows].std(axis=0, ddof=1)
        mu_te, sd_te = m[test_rows].mean(axis=0), m[test_rows].std(axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sr_tr = np.where(sd_tr > 0, mu_tr / sd_tr, -np.inf)
            sr_te = np.where(sd_te > 0, mu_te / sd_te, np.nan)
        best = int(np.argmax(sr_tr))
        if not np.isfinite(sr_te[best]):
            continue
        # Rank relativo OOS do campeão IS: rank/(N+1), a convenção de Bailey
        # et al. — rank/N enviesaria ω para cima em ~1/N e subestimaria o PBO
        # (auditoria da Fase 4).
        finite = sr_te[np.isfinite(sr_te)]
        omega = float(np.sum(finite <= sr_te[best])) / (len(finite) + 1.0)
        below_median += int(omega < 0.5)
        total += 1
    if total == 0:
        return float("nan")
    pbo = below_median / total
    logger.info("PBO/CSCV: %.3f em %d combinações (%d blocos).", pbo, total, n_blocks)
    return float(pbo)


def temporal_holdout_split(
    events: pd.DataFrame, date_col: str, holdout_months: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa o holdout temporal final (últimos N meses INTOCADOS até o fim).

    Toda escolha de hiperparâmetro/feature acontece no bloco de
    desenvolvimento; o holdout só é tocado uma vez, no relatório final.
    """
    cutoff = events[date_col].max() - pd.DateOffset(months=holdout_months)
    dev = events[events[date_col] <= cutoff]
    holdout = events[events[date_col] > cutoff]
    logger.info(
        "Holdout temporal: %d eventos dev (até %s), %d eventos holdout.",
        len(dev),
        cutoff.date(),
        len(holdout),
    )
    return dev, holdout
