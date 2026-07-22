# -*- coding: utf-8 -*-
"""Diagnóstico: a TD igual-ponderada morre por MEDIÇÃO ruidosa ou por ESTRUTURA?

Provocação do Marcelo (20/07): "Angelo também tem ruído, e mais. Se a base dele
funciona e a nossa não, o erro é no COMO medimos." Este script testa o mecanismo
proposto (gestores de poucas palavras = pontos ruidosos que atenuam o beta da
igual-ponderada) com evidência, não retórica:

  A) distribuição de nº de gestores e de min_words por call (nós vs bom-senso);
  B) TD igual-ponderada (Eq.3, controles antigos+FF49) SEPARADA por:
       - nº de gestores (<=3 poucos vs >=4 muitos);
       - min_words do gestor mais silencioso (alto = todos falaram bastante);
     se o mecanismo vale, a igual-ponderada revive onde NÃO há falador ruidoso;
  C) comparação com td_ex (exclui < mediana de palavras) e td_w (ponderada).
Se o beta da igual-ponderada aparecer nas calls "limpas" (poucos gestores, todos
falando muito), a causa é medição/estrutura de fala — não um bug de construção.
Uso: python scripts/diag_equalweight_null.py
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
OLD = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
       "length", "ln_mktcap", "industry_tone_ff49", "lagged_td", "sue_pct"]


def winsor(s):
    return s.clip(s.quantile(0.05), s.quantile(0.95))


def run(d, tdv, label):
    d = d.dropna(subset=["car_m1p1", tdv] + OLD + ["ticker", "year_quarter"]).copy()
    if len(d) < 400 or d["ticker"].nunique() < 25:
        print(f"{label:44s} n insuficiente ({len(d)})")
        return
    for c in ["car_m1p1", tdv] + OLD:
        d[c] = winsor(d[c])
    m = smf.ols(f"car_m1p1 ~ {tdv} + " + " + ".join(OLD) + " + C(ticker) + C(year_quarter)",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    print(f"{label:44s} n={len(d):>6d} | coef={m.params[tdv]:+8.4f} "
          f"t={m.tvalues[tdv]:+6.2f} p={m.pvalues[tdv]:.3f}")


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_car3.parquet")
    if "industry_tone_ff49" not in ev.columns:
        evp = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")[["call_id", "industry_tone_ff49"]]
        ev = ev.merge(evp, on="call_id", how="left")
    ev = ev.merge(pd.read_parquet(OUTDIR / "td_variants.parquet")[["call_id", "td_w", "td_ex"]],
                  on="call_id", how="left")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    ev = ev[ev["cdate"] >= "2009-01-01"]

    print("===== A) ESTRUTURA DE FALA POR CALL =====")
    print("nº de gestores por call:")
    print(ev["n_managers"].describe(percentiles=[.25, .5, .75, .9]).round(2).to_string())
    print(f"\n%% de calls com >=4 gestores: {100*(ev['n_managers'] >= 4).mean():.0f}%")
    print(f"min_words (palavras do gestor mais silencioso): "
          f"mediana {ev['min_words'].median():.0f}, "
          f"p25 {ev['min_words'].quantile(.25):.0f}")
    print(f"TD (igual-pond) — média {ev['td'].mean():.4f} mediana {ev['td'].median():.4f} "
          f"sd {ev['td'].std():.4f}  (paper T1: 0.0079/0.0074/0.0048)")

    print("\n===== B) TD IGUAL-PONDERADA por ESTRUTURA da call =====")
    print("(se o beta revive nas calls 'limpas', o nulo é estrutura de fala, não bug)")
    run(ev, "td", "TD igual-pond — TODAS as calls")
    run(ev[ev["n_managers"] <= 3], "td", "  só calls com <=3 gestores")
    run(ev[ev["n_managers"] >= 4], "td", "  só calls com >=4 gestores")
    hi = ev["min_words"].quantile(0.5)
    run(ev[ev["min_words"] >= hi], "td", "  só calls onde todos falaram muito (min_words alto)")
    run(ev[ev["min_words"] < hi], "td", "  só calls com algum gestor silencioso")
    # combinação mais limpa: poucos gestores E todos falando muito
    clean = ev[(ev["n_managers"] <= 3) & (ev["min_words"] >= hi)]
    run(clean, "td", "  LIMPAS: <=3 gestores E todos falando muito")

    print("\n===== C) referência: variantes que rebaixam o falador ruidoso =====")
    run(ev, "td_ex", "TD excluindo gestor < mediana de palavras")
    run(ev, "td_w", "TD ponderada por palavras")
    print("\nDONE.")


if __name__ == "__main__":
    main()
