# -*- coding: utf-8 -*-
"""Backtest de PREVISÃO por evento (espec do Marcelo, 23/07) — H1, H2, H3.

Para cada momento, prever com SÓ o passado; depois comparar com o realizado,
calculado à parte; agregar sobre todos os momentos. Sem vazamento nem pelo
resultado: o treino em cada corte usa apenas eventos cuja JANELA DE RESULTADO
já estava concluída antes do corte (call + horizonte da variável):
  car_m1p1: +2 dias úteis (~4 corridos) | vol_120: ~175 dias corridos |
  abs_sue_next: até 200 dias corridos.

Protocolo por hipótese:
  1. em cada 31/12 (2012..2024), estima a regressão (controles + FE) no
     treino leakage-safe -> direção prevista = sinal de beta;
  2. em cada trimestre do ano seguinte, calcula À PARTE o spread realizado:
     média(outcome | TD > mediana do tri) - média(outcome | TD <= mediana);
  3. acerto = spread realizado tem o sinal previsto;
  4. agrega: taxa de acerto, spread médio, t da série trimestral de spreads.
Tabela completa por ano (sem seleção). Uso: python scripts/sp500_oos_prediction.py
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
HORIZON_D = {"car_m1p1": 4, "vol_120": 175, "abs_sue_next": 200}


def winsor(s):
    return s.clip(s.quantile(0.05), s.quantile(0.95))


def beta_train(d, yv, sig):
    dd = d.dropna(subset=[yv, sig] + CTRL + ["ticker", "year_quarter"]).copy()
    if len(dd) < 800 or dd["ticker"].nunique() < 50:
        return None
    for c in [yv, sig] + CTRL:
        dd[c] = winsor(dd[c])
    m = smf.ols(f"{yv} ~ {sig} + " + " + ".join(CTRL) + " + C(ticker) + C(year_quarter)",
                data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["ticker"]})
    return float(m.params[sig])


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")
    ev = ev.merge(pd.read_parquet(OUTDIR / "td_variants.parquet")[["call_id", "td_w"]],
                  on="call_id", how="left")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    ev = ev[ev["cdate"] >= "2009-01-01"]
    ev["yq"] = ev["cdate"].dt.to_period("Q")

    for yv, sig, tag in [("car_m1p1", "td_w", "H1 CAR[-1,+1] | sinal td_w"),
                         ("vol_120", "td_w", "H2 vol 120d | sinal td_w"),
                         ("abs_sue_next", "td", "H3 |SUE t+1| | sinal td")]:
        print(f"\n===== {tag} =====")
        print(f"{'ano previsto':>12s} | {'B_treino':>9s} {'previsão':>9s} | "
              f"{'tri acertados':>13s} | {'spread médio do ano':>19s}")
        spreads_all, hits = [], []
        for y in range(2013, 2026):
            cut = pd.Timestamp(f"{y-1}-12-31") - pd.Timedelta(days=HORIZON_D[yv])
            train = ev[ev["cdate"] <= cut]
            b = beta_train(train, yv, sig)
            if b is None:
                print(f"{y:>12d} | treino insuficiente")
                continue
            pred = np.sign(b)
            year_sp = []
            for q in range(1, 5):
                sub = ev[(ev["yq"] == pd.Period(f"{y}Q{q}"))].dropna(subset=[yv, sig])
                if len(sub) < 60:
                    continue
                med = sub[sig].median()
                hi = sub.loc[sub[sig] > med, yv]
                lo = sub.loc[sub[sig] <= med, yv]
                sp = float(winsor(hi).mean() - winsor(lo).mean())
                year_sp.append(sp)
                spreads_all.append(sp)
                hits.append(int(np.sign(sp) == pred))
            if year_sp:
                nq_ok = sum(int(np.sign(s) == pred) for s in year_sp)
                print(f"{y:>12d} | {b:>+9.4f} {'menor' if pred < 0 else 'maior':>9s} | "
                      f"{nq_ok}/{len(year_sp):>10d} | {100*np.mean(year_sp):>+18.3f}%")
        sp = np.array(spreads_all)
        t = sp.mean() / sp.std() * np.sqrt(len(sp)) if len(sp) > 4 else np.nan
        print(f"AGREGADO: {len(sp)} trimestres | acerto {100*np.mean(hits):.0f}% | "
              f"spread médio {100*sp.mean():+.3f}% | t(spreads) {t:+.2f}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
