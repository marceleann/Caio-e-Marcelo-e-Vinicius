# -*- coding: utf-8 -*-
"""Diagnóstico: o que enfraqueceu H2/H3 na spec completa do paper?

Com os controles ANTIGOS (8, textuais+SUE+lnME) o H2 dava t=+2.2/+2.4 e
|SUE t+1| dava t=+2.5. Com os 17 controles do paper, ambos ~0. Duas causas
candidatas, testadas uma a uma NA MESMA AMOSTRA COMUM (n~14.5k):
  A) composição da amostra (dropna dos 17 controles corta metade);
  B) controles específicos: lagged_avg_td (persistência da TD) e/ou
     std_roa/std_cfo/smooth (risco operacional — canal do próprio H2).
Braços: (1) controles antigos na amostra ANTIGA 2009+; (2) controles antigos
na amostra COMUM; (3) paper menos lagged_avg_td; (4) paper menos
std_roa/std_cfo/smooth; (5) paper menos ambos.
Uso: python scripts/sp500_diag_controls.py
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
       "length", "ln_mktcap", "industry_tone", "lagged_td", "sue_pct"]
PAPER = ["disclosure_tone", "lagged_avg_td", "sue_pct", "btm", "lev",
         "roa", "std_roa", "std_cfo", "rd_at", "etr", "smooth",
         "length", "std_me", "ln_assets",
         "analyst_tone_disp", "analyst_tone", "industry_tone"]


def winsor(s):
    return s.clip(s.quantile(0.05), s.quantile(0.95))


def run(d, yvar, controls, label):
    d = d.dropna(subset=[yvar, "td"] + controls + ["ticker", "year_quarter"]).copy()
    for c in [yvar, "td"] + controls:
        d[c] = winsor(d[c])
    m = smf.ols(f"{yvar} ~ td + " + " + ".join(controls) + " + C(ticker) + C(year_quarter)",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    print(f"{label:44s} n={len(d):>6d} | coef={m.params['td']:+9.4f} "
          f"t={m.tvalues['td']:+6.2f} p={m.pvalues['td']:.3f}", flush=True)


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    ev = ev[ev["cdate"] >= "2009-01-01"]
    common = ev.dropna(subset=["td"] + list(dict.fromkeys(OLD + PAPER)))
    print(f"amostra 2009+: {len(ev)} | amostra comum (todos os controles): {len(common)}\n")

    no_lag = [c for c in PAPER if c != "lagged_avg_td"]
    no_oprisk = [c for c in PAPER if c not in ("std_roa", "std_cfo", "smooth")]
    no_both = [c for c in no_oprisk if c != "lagged_avg_td"]

    for yv, tag in [("vol_20", "H2 vol 20d"), ("vol_120", "H2 vol 120d"),
                    ("abs_sue_next", "H3 |SUE t+1|")]:
        print(f"----- {tag} -----")
        run(ev, yv, OLD, f"{tag}: controles ANTIGOS, amostra 2009+")
        run(common, yv, OLD, f"{tag}: controles ANTIGOS, amostra COMUM")
        run(common, yv, PAPER, f"{tag}: PAPER completo, amostra COMUM")
        run(common, yv, no_lag, f"{tag}: paper SEM lagged_avg_td")
        run(common, yv, no_oprisk, f"{tag}: paper SEM std_roa/cfo/smooth")
        run(common, yv, no_both, f"{tag}: paper SEM ambos")
        print()
    print("DONE.")


if __name__ == "__main__":
    main()
