"""Análise de decay pré/pós-2015: o sinal sobreviveu à própria publicação?

Por que existe:
    A pergunta secundária do projeto (McLean & Pontiff como referência
    conceitual): o efeito da distância de tom ENTRE OS GESTORES da mesma
    earnings call (tese de Angelo, 2025, Financial Review) perdeu força DEPOIS
    de o tema virar conhecimento público? Comparamos IC, CAR e Sharpe entre o
    regime pré-2015 e o pós-2015, cada um com TESTE DE DIFERENÇA — ambos os
    desfechos (sobreviveu / decaiu) são resultados válidos.

    A escolha de 2015 como fronteira é histórica (é quando a literatura de tom
    em calls ganhou tração pública); NÃO é corte de amostra (ADR-012): a
    amostra é 2005–2025 completa; 2015 é apenas a fronteira dos regimes.

Testes de diferença (escolhidos pela natureza de cada unidade):
    - **IC**: permutação dos rótulos de regime sobre os IC POR PERÍODO
      (trimestre — a unidade natural, quase-i.i.d.). H0: regime não importa.
    - **CAR**: permutação dos rótulos de regime sobre o CAR POR EVENTO.
    - **Sharpe**: bootstrap de BLOCOS MÓVEIS dentro de cada regime (respeita a
      autocorrelação induzida por posições sobrepostas); IC e p-valor da
      diferença ``Sharpe_pós − Sharpe_pré``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tonediv.backtest.metrics import ic_by_period, sharpe_ratio

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeComparison:
    """Resultado de uma comparação pré/pós de uma métrica.

    Attributes:
        metric: Nome da métrica ("ic", "car", "sharpe").
        value_pre: Valor no regime pré-corte.
        value_post: Valor no regime pós-corte.
        diff: ``value_post − value_pre``.
        p_value: P-valor bicaudal da hipótese "sem diferença entre regimes".
        n_pre: Nº de unidades (períodos/eventos/dias) no regime pré.
        n_post: Nº de unidades no regime pós.
    """

    metric: str
    value_pre: float
    value_post: float
    diff: float
    p_value: float
    n_pre: int
    n_post: int


def _perm_diff_pvalue(
    pooled: np.ndarray, n_pre: int, observed: float, n_iter: int, rng: np.random.Generator
) -> float:
    """P-valor bicaudal por permutação dos rótulos de regime.

    Sob H0 (regime não importa), a partição pré/pós é arbitrária: reembaralha-se
    quem é "pré" e recomputa-se a diferença de médias. Correção +1 (nunca zero).
    """
    n = len(pooled)
    if n_pre == 0 or n_pre >= n:
        return float("nan")
    null = np.empty(n_iter)
    for it in range(n_iter):
        perm = rng.permutation(n)
        a, b = pooled[perm[:n_pre]], pooled[perm[n_pre:]]
        null[it] = np.nanmean(b) - np.nanmean(a)
    null = null[np.isfinite(null)]
    if len(null) == 0:
        return float("nan")
    return (1.0 + float(np.sum(np.abs(null) >= abs(observed)))) / (len(null) + 1.0)


def compare_ic(
    events: pd.DataFrame,
    feature: str,
    ret_col: str,
    date_col: str,
    split_year: int,
    n_iter: int,
    rng: np.random.Generator,
    freq: str = "QE",
) -> RegimeComparison:
    """Compara o IC médio por período entre regimes, com teste de permutação."""
    ics = ic_by_period(events, feature, ret_col, date_col, freq)
    if ics.empty:
        return RegimeComparison("ic", float("nan"), float("nan"), float("nan"), float("nan"), 0, 0)
    years = ics.index.year
    pre, post = ics[years < split_year], ics[years >= split_year]
    diff = float(post.mean() - pre.mean())
    p = _perm_diff_pvalue(ics.to_numpy(), len(pre), diff, n_iter, rng)
    return RegimeComparison(
        "ic", float(pre.mean()), float(post.mean()), diff, p, len(pre), len(post)
    )


def compare_car(
    events_car: pd.DataFrame,
    car_col: str,
    date_col: str,
    split_year: int,
    n_iter: int,
    rng: np.random.Generator,
) -> RegimeComparison:
    """Compara o CAR médio por evento entre regimes, com teste de permutação."""
    car = events_car[[date_col, car_col]].dropna()
    years = car[date_col].dt.year
    pre = car.loc[years < split_year, car_col].to_numpy()
    post = car.loc[years >= split_year, car_col].to_numpy()
    if len(pre) == 0 or len(post) == 0:
        return RegimeComparison(
            "car", float("nan"), float("nan"), float("nan"), float("nan"), len(pre), len(post)
        )
    diff = float(post.mean() - pre.mean())
    p = _perm_diff_pvalue(np.concatenate([pre, post]), len(pre), diff, n_iter, rng)
    return RegimeComparison(
        "car", float(pre.mean()), float(post.mean()), diff, p, len(pre), len(post)
    )


def _moving_block_sample(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Reamostra uma série por blocos móveis CIRCULARES (Politis-Romano).

    Blocos circulares: os inícios são sorteados em ``[0, n)`` e o wraparound
    (``% n``) fecha os que passam do fim — assim TODOS os pontos, inclusive as
    bordas, têm cobertura uniforme (o bootstrap não-circular sub-amostra os
    primeiros/últimos ``block−1`` pontos por ~10x; auditoria da Fase 5). Preserva
    a autocorrelação dentro de cada bloco.
    """
    n = len(x)
    if n == 0:
        return x
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)  # circular: qualquer início
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return x[idx % n]


def compare_sharpe(
    returns: pd.Series,
    split_date: pd.Timestamp,
    periods_per_year: int,
    block: int,
    n_iter: int,
    rng: np.random.Generator,
) -> RegimeComparison:
    """Compara o Sharpe entre regimes com bootstrap de blocos móveis.

    O p-valor é a fração das reamostragens em que a diferença cruza zero
    (bicaudal, correção +1) — respeita a autocorrelação de posições sobrepostas.
    """
    r = returns.dropna()
    pre = r[r.index < split_date].to_numpy()
    post = r[r.index >= split_date].to_numpy()
    sr_pre = sharpe_ratio(pd.Series(pre), periods_per_year)
    sr_post = sharpe_ratio(pd.Series(post), periods_per_year)
    diff = sr_post - sr_pre
    if len(pre) < 2 * block or len(post) < 2 * block:
        return RegimeComparison("sharpe", sr_pre, sr_post, diff, float("nan"), len(pre), len(post))

    boot = np.empty(n_iter)
    for it in range(n_iter):
        bp = sharpe_ratio(pd.Series(_moving_block_sample(pre, block, rng)), periods_per_year)
        bq = sharpe_ratio(pd.Series(_moving_block_sample(post, block, rng)), periods_per_year)
        boot[it] = bq - bp
    boot = boot[np.isfinite(boot)]
    # p bicaudal: fração das reamostragens no lado OPOSTO ao sinal da diferença.
    frac = float(np.mean(boot <= 0)) if diff > 0 else float(np.mean(boot >= 0))
    p = min(1.0, 2.0 * (1.0 + frac * len(boot)) / (len(boot) + 1.0))
    return RegimeComparison("sharpe", sr_pre, sr_post, diff, p, len(pre), len(post))


def comparisons_to_frame(comparisons: list[RegimeComparison]) -> pd.DataFrame:
    """Empacota uma lista de :class:`RegimeComparison` numa tabela para salvar."""
    return pd.DataFrame(
        [
            {
                "metric": c.metric,
                "pre": c.value_pre,
                "post": c.value_post,
                "diff_post_minus_pre": c.diff,
                "p_value": c.p_value,
                "n_pre": c.n_pre,
                "n_post": c.n_post,
            }
            for c in comparisons
        ]
    )
