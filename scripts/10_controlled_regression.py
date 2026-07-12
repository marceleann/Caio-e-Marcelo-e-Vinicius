"""10 — Regressão CONTROLADA da distância de tom (método do Angelo), reproduzível.

Testa a Hipótese central do Angelo na nossa base: o retorno futuro regredido na
distância de tom + controles + efeitos fixos de empresa e trimestre, com erros
agrupados por empresa. A distância de tom CRUA não prevê nada (é confundida); o
efeito só aparece controlado. Todas as variáveis são winsorizadas nos percentis
5 e 95 (como o Angelo, Seção 4.1) para robustez a valores extremos.

Uso:
    python scripts/10_controlled_regression.py --config config.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from tonediv.config import configure_logging, load_config

# Controles disponíveis com dados abertos (os de balanço do Angelo exigem
# Compustat; os efeitos fixos de empresa/trimestre absorvem parte — ADR-024).
CONTROLS = [
    "disclosure_tone", "analyst_tone", "analyst_tone_distance",
    "lagged_tone_distance", "length", "momentum", "reversal", "size_proxy",
]
WINSOR_LO, WINSOR_HI = 0.05, 0.95


def _winsorize(s: pd.Series) -> pd.Series:
    return s.clip(s.quantile(WINSOR_LO), s.quantile(WINSOR_HI))


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def _run(df: pd.DataFrame, dep: str, winsor: bool) -> tuple[float, float, float, int]:
    cols = ["tone_distance", dep, *CONTROLS, "ticker", "year_quarter"]
    d = df.dropna(subset=cols).copy()
    if winsor:
        for c in ["tone_distance", dep, *CONTROLS]:
            d[c] = _winsorize(d[c])
    d["td_z"] = _z(d["tone_distance"])
    for c in CONTROLS:
        d[c] = _z(d[c])
    design = pd.concat(
        [
            d[["td_z", *CONTROLS]].reset_index(drop=True),
            pd.get_dummies(d["ticker"], prefix="firm", drop_first=True, dtype=float).reset_index(drop=True),
            pd.get_dummies(d["year_quarter"], prefix="q", drop_first=True, dtype=float).reset_index(drop=True),
        ],
        axis=1,
    )
    design = sm.add_constant(design)
    model = sm.OLS(d[dep].to_numpy(dtype=float), design.to_numpy(dtype=float)).fit(
        cov_type="cluster", cov_kwds={"groups": d["ticker"].to_numpy()}
    )
    j = list(design.columns).index("td_z")
    return float(model.params[j]), float(model.tvalues[j]), float(model.pvalues[j]), len(d)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regressão controlada da distância de tom.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.logging)
    ev = pd.read_parquet(cfg.paths.data_processed / "events.parquet")

    print("[10] Regressão: retorno ~ distância de tom (padronizada) + controles + efeitos fixos")
    print("     coef = retorno por desvio-padrão de distância de tom | erros agrupados por empresa")
    for dep in ["ret_21", "ret_63"]:
        for winsor in (False, True):
            coef, t, p, n = _run(ev, dep, winsor)
            tag = "winsor 5/95" if winsor else "sem winsor "
            print(f"  {dep} | {tag} | coef={coef:+.4f} ({coef*100:+.2f}%) t={t:+.2f} p={p:.4f} n={n}")


if __name__ == "__main__":
    main()
