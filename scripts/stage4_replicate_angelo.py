# -*- coding: utf-8 -*-
"""Stage 4 — replicação do Angelo com a Tone Distance NOVA (por sentença, debiasada).

H1: CAR[-1,+1] ~ Tone Distance + controles, com FIXED EFFECTS de empresa e de
    trimestre e ERROS AGRUPADOS (clustered) por empresa. Sinal esperado: NEGATIVO.
Prêmio (Angelo Tabela 6): ret_21 e ret_63 ~ Tone Distance + controles. Esperado:
    POSITIVO (a distância de tom alta rende mais em ~1–3 meses — direção da estratégia).

Reusa o `car` e os controles já calculados em events/event_study_cars (independem
da TD, pois CAR é preço). Troca a TD antiga (por fala) pela nova (por sentença).
Winsoriza 5/95 (como o Angelo, Seção 4.1). TD padronizada -> coef por desvio-padrão.

Uso: python scripts/stage4_replicate_angelo.py [universo=strict|core|all]
"""
from __future__ import annotations
import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
INT = ROOT / "data" / "interim"
OUT = ROOT / "data" / "outputs"

UNIV = sys.argv[1] if len(sys.argv) > 1 else "strict"
TD_SRC = sys.argv[2] if len(sys.argv) > 2 else "llm"   # heur | llm (qual tone_distance_*.parquet)
CAR_SRC = sys.argv[3] if len(sys.argv) > 3 else "t1"   # t1 (antigo, T+1 [-1,+10]) | ann (anúncio [-1,+1])
CTRL_V = sys.argv[4] if len(sys.argv) > 4 else "v1"    # v1 | v2 (+mktcap/indtone) | v3 (v2 + SUE)
CONTROLS_V1 = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
               "length", "size_proxy", "lagged_tone_distance"]
CONTROLS_V2 = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
               "length", "ln_mktcap", "industry_tone", "lagged_tone_distance"]
CONTROLS_V3 = CONTROLS_V2 + ["sue_pct"]
CONTROLS = {"v1": CONTROLS_V1, "v2": CONTROLS_V2, "v3": CONTROLS_V3}[CTRL_V]
VOL_CONTROLS = ["n_managers", "log_min_sent"]   # controles do viés de volume (robustez)


def winsor(s, lo=0.05, hi=0.95):
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def zscore(s):
    return (s - s.mean()) / s.std()


def run(df, yvar, tdvar, controls, label):
    d = df.dropna(subset=[yvar, tdvar] + controls).copy()
    if len(d) < 50 or d["ticker"].nunique() < 3:
        return dict(spec=label, y=yvar, td=tdvar, n=len(d), coef=np.nan, t=np.nan, p=np.nan, r2=np.nan)
    d["_td"] = zscore(winsor(d[tdvar]))
    d["_y"] = winsor(d[yvar])
    for c in controls:
        d[c] = winsor(d[c])
    rhs = "_td + " + " + ".join(controls) + " + C(ticker) + C(year_quarter)"
    try:
        m = smf.ols(f"_y ~ {rhs}", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["ticker"]})
        return dict(spec=label, y=yvar, td=tdvar, n=len(d),
                    coef=m.params["_td"], t=m.tvalues["_td"], p=m.pvalues["_td"],
                    r2=m.rsquared)
    except Exception as e:
        return dict(spec=label, y=yvar, td=tdvar, n=len(d), coef=np.nan, t=np.nan, p=str(e)[:40])


def main():
    ev = pd.read_parquet(INT.parent / "processed" / "events.parquet")
    if CAR_SRC == "ann":
        car = pd.read_parquet(INT / "announcement_car.parquet")[["call_id", "car"]]
    else:
        car = pd.read_parquet(OUT / "event_study_cars.parquet")[["call_id", "car"]]
    td = pd.read_parquet(INT / f"tone_distance_{TD_SRC}.parquet")   # NOVA TD (por sentença)
    blocks = pd.read_parquet(INT / "call_blocks.parquet")

    # descartar a TD antiga do events; usar a nova
    ev = ev.drop(columns=[c for c in ["tone_distance", "n_managers_td"] if c in ev.columns])
    df = (ev.merge(car, on="call_id", how="left")
            .merge(td, on="call_id", how="inner")     # só calls com TD nova
            .merge(blocks[["call_id", "block", "in_strict_tech", "in_core_pure"]], on="call_id", how="left"))
    if CTRL_V in ("v2", "v3"):
        c2 = pd.read_parquet(INT / "events_controls_v2.parquet")
        df = df.merge(c2, on="call_id", how="left")
    if CTRL_V == "v3":
        sue = pd.read_parquet(INT / "events_sue.parquet")[["call_id", "sue_pct"]]
        df = df.merge(sue, on="call_id", how="left")
    df["log_min_sent"] = np.log(df["min_sent"].clip(lower=1))

    if UNIV == "strict":
        df = df[df["in_strict_tech"]]
    elif UNIV == "core":
        df = df[df["in_core_pure"]]
    print(f"universo={UNIV} | calls com TD nova: {len(df)} | com car: {df['car'].notna().sum()} "
          f"| empresas: {df['ticker'].nunique()} | trimestres: {df['year_quarter'].nunique()}")

    print("\n" + "=" * 84)
    print("H1 — CAR[-1,+1] ~ Tone Distance (esperado NEGATIVO) | FE empresa+trimestre, cluster empresa")
    print("=" * 84)
    print(f"{'spec':30s} {'n':>5s} {'coef(por SD)':>13s} {'t':>7s} {'p':>8s}")
    specs = [
        ("TD crua + controles", "car", "td", CONTROLS),
        ("TD_adj + controles", "car", "td_adj", CONTROLS),
        ("z + controles", "car", "z", CONTROLS),
        ("TD crua + ctrl + vol", "car", "td", CONTROLS + VOL_CONTROLS),
    ]
    for lbl, y, tdv, ctr in specs:
        r = run(df, y, tdv, ctr, lbl)
        pp = f"{r['p']:.3f}" if isinstance(r["p"], float) else r["p"]
        print(f"{lbl:30s} {r['n']:>5d} {r['coef']:>13.4f} {r['t']:>7.2f} {pp:>8s}  R2={r.get('r2', float('nan')):.3f}")

    for horizon, yv in [("1 mês", "ret_21"), ("3 meses", "ret_63")]:
        print("\n" + "=" * 84)
        print(f"PRÊMIO {horizon} — {yv} ~ Tone Distance (esperado POSITIVO) | mesmos FE/cluster")
        print("=" * 84)
        print(f"{'spec':30s} {'n':>5s} {'coef(por SD)':>13s} {'t':>7s} {'p':>8s}")
        for lbl, tdv, ctr in [("TD crua + controles", "td", CONTROLS),
                              ("TD_adj + controles", "td_adj", CONTROLS),
                              ("z + controles", "z", CONTROLS)]:
            r = run(df, yv, tdv, ctr, lbl)
            pp = f"{r['p']:.3f}" if isinstance(r["p"], float) else r["p"]
            print(f"{lbl:30s} {r['n']:>5d} {r['coef']:>13.4f} {r['t']:>7.2f} {pp:>8s}  R2={r.get('r2', float('nan')):.3f}")
    print("\nNota: coef 'por SD' = efeito de +1 desvio-padrão da TD (winsorizada 5/95).")
    print("DONE.")


if __name__ == "__main__":
    main()
