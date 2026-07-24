# -*- coding: utf-8 -*-
"""Backtest de previsão CONDICIONAL por evento (correção apontada pelo Marcelo).

O teste incondicional anterior (ordenar por TD crua) testava uma afirmação
que o painel nunca fez. Aqui o teste casa com a hipótese: em cada corte,
treina-se no passado o modelo SEM o sinal (resultado ~ controles + FE firma
+ FE trimestre); para os eventos do ano seguinte calcula-se o RESÍDUO
(realizado − explicado pelos controles e pela identidade da firma, com
coeficientes e limites de winsorização do treino — leakage-safe); dentro de
cada trimestre futuro, mede-se o spread dos resíduos entre a metade alta e
baixa do sinal. Se a relação condicional é real e antecipável, o spread tem
o sinal previsto pela hipótese.
  H1: resíduo de CAR[-1,+1] menor p/ td_w alta (spread negativo)
  H2: resíduo de vol_120 maior (positivo) | H3: resíduo de |SUE t+1| maior.
Firmas sem histórico no treino ficam de fora (sem FE estimado) — contadas.
FE de trimestre futuro é desnecessário: o split é DENTRO do trimestre.
Uso: python scripts/sp500_oos_conditional.py
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "interim" / "sp500"
CTRL = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
        "length", "ln_mktcap", "industry_tone_ff49", "lagged_td", "sue_pct"]
HOR = {"car_m1p1": 4, "vol_120": 175, "abs_sue_next": 200}


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")
    ev = ev.merge(pd.read_parquet(OUTDIR / "td_variants.parquet")[["call_id", "td_w"]],
                  on="call_id", how="left")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    ev = ev[ev["cdate"] >= "2009-01-01"]
    ev["yq"] = ev["cdate"].dt.to_period("Q")

    for yv, sig, exp, tag in [("car_m1p1", "td_w", -1, "H1 resíduo CAR ~ td_w"),
                              ("vol_120", "td_w", +1, "H2 resíduo vol ~ td_w"),
                              ("abs_sue_next", "td", +1, "H3 resíduo |SUE| ~ td")]:
        print(f"\n===== {tag} | direção esperada: {'negativa' if exp<0 else 'positiva'} =====")
        spreads, hits = [], []
        print(f"{'ano':>5s} | {'tri ok':>6s} | {'spread médio':>12s} | {'eventos usados':>14s}")
        for y in range(2013, 2026):
            cut = pd.Timestamp(f"{y-1}-12-31") - pd.Timedelta(days=HOR[yv])
            tr = ev[ev["cdate"] <= cut].dropna(subset=[yv] + CTRL + ["ticker", "year_quarter"]).copy()
            if len(tr) < 800:
                continue
            # limites de winsor DO TREINO (aplicados também ao futuro)
            lims = {c: (tr[c].quantile(0.05), tr[c].quantile(0.95)) for c in [yv] + CTRL}
            for c in [yv] + CTRL:
                tr[c] = tr[c].clip(*lims[c])
            m = smf.ols(f"{yv} ~ " + " + ".join(CTRL) + " + C(ticker) + C(year_quarter)",
                        data=tr).fit()
            fe = {t: m.params.get(f"C(ticker)[T.{t}]", np.nan) for t in tr["ticker"].unique()}
            base_t = tr["ticker"].iloc[0]  # nível de referência do FE
            fut = ev[(ev["cdate"].dt.year == y)].dropna(subset=[yv, sig] + CTRL).copy()
            fut = fut[fut["ticker"].isin(tr["ticker"].unique())]
            if len(fut) < 200:
                continue
            for c in [yv] + CTRL:
                fut[c] = fut[c].clip(*lims[c])
            pred = m.params["Intercept"] + sum(m.params[c] * fut[c] for c in CTRL)
            pred = pred + fut["ticker"].map(lambda t: 0.0 if t == base_t
                                            else fe.get(t, np.nan)).astype(float)
            fut["resid"] = fut[yv] - pred
            fut = fut.dropna(subset=["resid"])
            year_sp, n_used = [], 0
            for q in range(1, 5):
                sub = fut[fut["yq"] == pd.Period(f"{y}Q{q}")]
                if len(sub) < 60:
                    continue
                med = sub[sig].median()
                sp = float(sub.loc[sub[sig] > med, "resid"].mean()
                           - sub.loc[sub[sig] <= med, "resid"].mean())
                year_sp.append(sp)
                spreads.append(sp)
                hits.append(int(np.sign(sp) == exp))
                n_used += len(sub)
            if year_sp:
                nq = sum(int(np.sign(s) == exp) for s in year_sp)
                print(f"{y:>5d} | {nq}/{len(year_sp):>3d} | {100*np.mean(year_sp):>+11.3f}% | {n_used:>14d}")
        sp = np.array(spreads)
        if len(sp) > 4:
            t = sp.mean() / sp.std() * np.sqrt(len(sp))
            print(f"AGREGADO: {len(sp)} tri | acerto {100*np.mean(hits):.0f}% | "
                  f"spread {100*sp.mean():+.3f}% | t {t:+.2f}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
