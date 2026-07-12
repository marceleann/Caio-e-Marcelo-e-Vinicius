"""Métricas de desempenho: Sharpe, DSR (Bailey-LdP), IC, t-NW, drawdown, turnover.

Por que existe:
    Concentra TODA a régua de avaliação do projeto num módulo puro e testado
    com casos verificáveis à mão. A métrica mais delicada é o **Deflated
    Sharpe Ratio** (Bailey & López de Prado, 2014): testar uma grade de
    combinações infla o melhor Sharpe por sorte; o DSR desconta esse viés
    usando o número REAL de tentativas e a variância dos Sharpes da grade
    (armadilha nº 7 — nada de aproximação silenciosa; hipóteses documentadas
    no docstring da função).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def sharpe_ratio(returns: pd.Series, periods_per_year: int) -> float:
    """Sharpe anualizado de uma série de retornos por período (rf = 0).

    ``SR = mean/std(ddof=1) × sqrt(P)``. Caixa rende zero no projeto
    (config ``strategy.cash_return``), então o excesso é o próprio retorno.
    Retorna NaN para séries com menos de 2 pontos ou desvio nulo.
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    # nunique()==1 pega série constante cujo std sai ~1e-18 por erro de ponto
    # flutuante (não zero exato) — sem o guard, o Sharpe explodiria p/ ~1e16.
    if sd == 0 or r.nunique() == 1:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def newey_west_tstat(returns: pd.Series, lags: int) -> float:
    """t-stat da média com erro-padrão HAC (Newey-West).

    Regressão da série numa constante com covariância HAC — o erro-padrão
    corrige autocorrelação até ``lags`` defasagens (posições sobrepostas de H
    pregões induzem autocorrelação mecânica de ordem ~H; daí lags ≈ horizonte
    no config). Implementação via ``statsmodels`` (referência da área).
    """
    import statsmodels.api as sm  # import tardio: dependência pesada

    r = returns.dropna().to_numpy(dtype=np.float64)
    if len(r) < max(8, lags + 2):
        return float("nan")
    model = sm.OLS(r, np.ones((len(r), 1)))
    fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0])


def rank_ic(signal: pd.Series, forward_returns: pd.Series) -> float:
    """IC de rank (correlação de Spearman) entre sinal e retorno futuro.

    Pares com NaN em qualquer lado são descartados; menos de 3 pares -> NaN.
    """
    df = pd.DataFrame({"s": signal, "r": forward_returns}).dropna()
    if len(df) < 3 or df["s"].nunique() < 2 or df["r"].nunique() < 2:
        return float("nan")
    rho, _ = stats.spearmanr(df["s"], df["r"])
    return float(rho)


def ic_by_period(
    events: pd.DataFrame, feature: str, ret_col: str, date_col: str, freq: str = "QE"
) -> pd.Series:
    """Série de rank-IC por janela temporal (cross-section por período).

    Por que por período: eventos são esparsos no dia; o IC cross-section
    honesto agrupa eventos numa janela (trimestre por default — casa com o
    ciclo de earnings) e correlaciona sinal × retorno DENTRO dela.

    Returns:
        Série indexada pelo fim do período com o IC de cada janela (períodos
        com menos de 3 eventos válidos ficam NaN e são descartados).
    """
    df = events[[date_col, feature, ret_col]].dropna()
    grouped = df.groupby(pd.Grouper(key=date_col, freq=freq))
    ics = grouped.apply(lambda g: rank_ic(g[feature], g[ret_col]), include_groups=False)
    return ics.dropna()


def ic_ir(ic_series: pd.Series) -> float:
    """IC-IR: média dos ICs por período dividida pelo desvio (consistência)."""
    if len(ic_series) < 2:
        return float("nan")
    sd = float(ic_series.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(ic_series.mean() / sd)


def max_drawdown(returns: pd.Series) -> float:
    """Máximo drawdown (fração negativa) da curva de capital composta."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    curve = (1.0 + r).cumprod()
    peak = curve.cummax()
    return float((curve / peak - 1.0).min())


def annual_turnover(daily_turnover: pd.Series, periods_per_year: int) -> float:
    """Turnover anualizado (média diária × períodos/ano)."""
    if daily_turnover.dropna().empty:
        return float("nan")
    return float(daily_turnover.dropna().mean() * periods_per_year)


def deflated_sharpe_ratio(
    observed_sr: float,
    n_obs: int,
    skew: float,
    kurt: float,
    n_trials: int,
    trials_sr_var: float,
) -> float:
    """Deflated Sharpe Ratio de Bailey & López de Prado (2014).

    Responde: "qual a probabilidade de o Sharpe observado ser positivo DE
    VERDADE, dado que foi o melhor de ``n_trials`` tentativas?". Dois passos:

    1. **Benchmark de sorte** — Sharpe esperado do MÁXIMO de ``n_trials``
       estratégias sem habilidade (aproximação de valores extremos, eq. do
       paper): ``SR0 = sqrt(V) × [(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]``,
       com ``V = trials_sr_var`` (variância dos Sharpes da grade REAL — nunca
       um chute) e γ = constante de Euler-Mascheroni.
    2. **PSR contra SR0** — Probabilistic Sharpe Ratio ajustado a skew e
       curtose dos retornos:
       ``DSR = Φ( ((SR−SR0)·sqrt(n−1)) / sqrt(1 − γ₃·SR + (γ₄−1)/4·SR²) )``.

    Hipóteses (documentadas por exigência da armadilha nº 7): Sharpes da
    grade ~normais entre si; retornos i.i.d. dentro de cada estratégia (a
    correção de skew/curtose relaxa a normalidade, não a independência —
    posições sobrepostas violam parcialmente; por isso o DSR complementa, e
    não substitui, o t-NW e o placebo). SR e SR0 na MESMA base (por período,
    não anualizada).

    Args:
        observed_sr: Sharpe POR PERÍODO da estratégia campeã (não anualizado).
        n_obs: Nº de períodos da série de retornos.
        skew: Assimetria amostral dos retornos.
        kurt: Curtose amostral NÃO-excedente (normal = 3).
        n_trials: Nº REAL de células testadas na grade (script 07).
        trials_sr_var: Variância dos Sharpes (por período) da grade real.

    Returns:
        DSR em [0, 1] — probabilidade de habilidade genuína. NaN se
        parâmetros insuficientes.
    """
    if n_obs < 2 or n_trials < 1 or not np.isfinite(observed_sr):
        return float("nan")
    if n_trials == 1 or trials_sr_var <= 0:
        sr0 = 0.0  # sem grade, o benchmark de sorte é Sharpe zero (PSR clássico)
    else:
        gamma = float(np.euler_gamma)
        e = float(np.e)
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
        sr0 = float(np.sqrt(trials_sr_var) * ((1.0 - gamma) * z1 + gamma * z2))

    denom = 1.0 - skew * observed_sr + ((kurt - 1.0) / 4.0) * observed_sr**2
    if denom <= 0:
        return float("nan")
    z = (observed_sr - sr0) * np.sqrt(n_obs - 1.0) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def performance_summary(
    returns: pd.Series,
    turnover: pd.Series,
    bps_per_side: float,
    periods_per_year: int,
    nw_lags: int,
) -> dict[str, float]:
    """Resumo padrão de uma série de retornos brutos + turnover (para tabelas).

    Retorna Sharpe bruto e líquido, t-NW líquido, drawdown líquido, turnover
    anualizado e arrasto de custos — as colunas mínimas de qualquer tabela de
    resultado do projeto.
    """
    from tonediv.backtest.costs import apply_costs, total_cost_drag

    net = apply_costs(returns, turnover, bps_per_side)
    return {
        "sharpe_gross": sharpe_ratio(returns, periods_per_year),
        "sharpe_net": sharpe_ratio(net, periods_per_year),
        "tstat_nw_net": newey_west_tstat(net, nw_lags),
        "max_drawdown_net": max_drawdown(net),
        "annual_turnover": annual_turnover(turnover, periods_per_year),
        "cost_drag_annual": total_cost_drag(turnover, bps_per_side, periods_per_year),
        "n_days": float(returns.dropna().shape[0]),
    }
