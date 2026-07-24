# -*- coding: utf-8 -*-
"""Validação walk-forward das hipóteses (exigência do Marcelo, 23/07).

Pergunta do gestor: parado em cada momento t, SÓ com dados até t, eu teria
estimado a relação — e ela teria valido no futuro ainda não visto?

Para cada corte (fim de ano, 2012..2024) e cada hipótese:
  beta_passado = regressão nos eventos com cdate <= corte;
  beta_futuro  = regressão nos eventos com cdate  > corte;
  winsorização e quantis calculados DENTRO de cada janela (nada do futuro
  vaza para o passado); mesmos controles (antigos + FF49) e FE de sempre.
Hipóteses/sinais (fixados): H1 car_m1p1 ~ td_w | H2 vol_120 ~ td_w |
H3 abs_sue_next ~ td (a espec em que H3 foi documentada).
Saída: tabela completa de cortes (sem seleção) + resumo de concordância de
sinal. Uso: python scripts/sp500_walkforward.py
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
CUTS = [f"{y}-12-31" for y in range(2012, 2025)]


def winsor(s):
    return s.clip(s.quantile(0.05), s.quantile(0.95))


def beta(d, yv, sig):
    dd = d.dropna(subset=[yv, sig] + CTRL + ["ticker", "year_quarter"]).copy()
    if len(dd) < 800 or dd["ticker"].nunique() < 50:
        return None
    for c in [yv, sig] + CTRL:
        dd[c] = winsor(dd[c])
    m = smf.ols(f"{yv} ~ {sig} + " + " + ".join(CTRL) + " + C(ticker) + C(year_quarter)",
                data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["ticker"]})
    return m.params[sig], m.tvalues[sig], len(dd)


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")
    ev = ev.merge(pd.read_parquet(OUTDIR / "td_variants.parquet")[["call_id", "td_w"]],
                  on="call_id", how="left")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    ev = ev[ev["cdate"] >= "2009-01-01"]

    for yv, sig, tag in [("car_m1p1", "td_w", "H1 CAR[-1,+1] ~ td_w"),
                         ("vol_120", "td_w", "H2 vol_120 ~ td_w"),
                         ("abs_sue_next", "td", "H3 |SUE t+1| ~ td")]:
        print(f"\n===== {tag} =====")
        print(f"{'corte':>10s} | {'B_passado':>10s} {'t':>6s} {'n':>6s} | "
              f"{'B_futuro':>10s} {'t':>6s} {'n':>6s} | concorda?")
        agree = tot = 0
        for cut in CUTS:
            past = beta(ev[ev["cdate"] <= cut], yv, sig)
            fut = beta(ev[ev["cdate"] > cut], yv, sig)
            if past is None or fut is None:
                print(f"{cut:>10s} | janela insuficiente")
                continue
            ok = np.sign(past[0]) == np.sign(fut[0])
            tot += 1
            agree += int(ok)
            print(f"{cut:>10s} | {past[0]:>+10.4f} {past[1]:>+6.2f} {past[2]:>6d} | "
                  f"{fut[0]:>+10.4f} {fut[1]:>+6.2f} {fut[2]:>6d} | {'SIM' if ok else 'NAO'}")
        print(f"concordância de sinal: {agree}/{tot} cortes")
    print("\nDONE.")


if __name__ == "__main__":
    main()
