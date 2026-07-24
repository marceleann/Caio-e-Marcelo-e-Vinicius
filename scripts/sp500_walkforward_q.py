# -*- coding: utf-8 -*-
"""Walk-forward TRIMESTRAL da H1 condicional (cortes menores — Marcelo, 23/07).

Em cada fim de trimestre (2012Q4..2024Q4): beta da Eq.(3) estimado SÓ no
passado (eventos com resultado concluído antes do corte) vs beta estimado SÓ
no futuro (eventos após o corte). Winsorização dentro de cada janela.
Nota de honestidade impressa junto: cortes adjacentes compartilham dados —
a curva mede ESTABILIDADE, não são ~49 confirmações independentes.
Uso: python scripts/sp500_walkforward_q.py
"""
from __future__ import annotations
import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "interim" / "sp500"
CTRL = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
        "length", "ln_mktcap", "industry_tone_ff49", "lagged_td", "sue_pct"]
# hipótese via argv: h1 (default) | h2 | h3 — com horizonte leakage-safe próprio
HYP = (sys.argv[1] if len(sys.argv) > 1 else "h1").lower()
SPEC = {"h1": ("car_m1p1", "td_w", 4),
        "h2": ("vol_120", "td_w", 175),
        "h3": ("abs_sue_next", "td", 200)}[HYP]
YVAR, SIG, HDAYS = SPEC
HORIZON = pd.Timedelta(days=HDAYS)


def winsor(s):
    return s.clip(s.quantile(0.05), s.quantile(0.95))


def beta(d):
    dd = d.dropna(subset=[YVAR, SIG] + CTRL + ["ticker", "year_quarter"]).copy()
    if len(dd) < 800 or dd["ticker"].nunique() < 50:
        return None
    for c in [YVAR, SIG] + CTRL:
        dd[c] = winsor(dd[c])
    m = smf.ols(f"{YVAR} ~ {SIG} + " + " + ".join(CTRL) + " + C(ticker) + C(year_quarter)",
                data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["ticker"]})
    return float(m.params[SIG]), float(m.tvalues[SIG]), len(dd)


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")
    ev = ev.merge(pd.read_parquet(OUTDIR / "td_variants.parquet")[["call_id", "td_w"]],
                  on="call_id", how="left")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    ev = ev[ev["cdate"] >= "2009-01-01"]

    exp = -1 if HYP == "h1" else +1     # direção esperada pela hipótese
    cuts = pd.period_range("2012Q4", "2024Q4", freq="Q")
    ok_p = ok_f = tot = 0
    print(f"HIPÓTESE {HYP.upper()}: {YVAR} ~ {SIG} | direção esperada: "
          f"{'negativa' if exp < 0 else 'positiva'}")
    print(f"{'corte':>8s} | {'B_pass':>8s} {'t':>6s} | {'B_fut':>8s} {'t':>6s} | pass_ok fut_ok")
    for cq in cuts:
        cut = cq.end_time.normalize()
        past = beta(ev[ev["cdate"] + HORIZON <= cut])
        fut = beta(ev[ev["cdate"] > cut])
        if past is None or fut is None:
            print(f"{str(cq):>8s} | janela insuficiente")
            continue
        tot += 1
        ok_p += int(np.sign(past[0]) == exp)
        ok_f += int(np.sign(fut[0]) == exp)
        print(f"{str(cq):>8s} | {past[0]:>+8.4f} {past[1]:>+6.2f} | "
              f"{fut[0]:>+8.4f} {fut[1]:>+6.2f} | "
              f"{'sim' if np.sign(past[0])==exp else 'NAO':>7s} "
              f"{'sim' if np.sign(fut[0])==exp else 'NAO':>6s}")
    print(f"\nRESUMO {HYP.upper()}: {tot} cortes | passado na direção esperada: {ok_p}/{tot} | "
          f"futuro na direção esperada: {ok_f}/{tot}")
    print("NOTA: cortes adjacentes compartilham dados — curva de ESTABILIDADE, "
          "não confirmações independentes.")
    print("DONE.")


if __name__ == "__main__":
    main()
