# -*- coding: utf-8 -*-
"""H2 — rodada FINAL e pré-comprometida (24/07): log-vol, 2 horizontes, 2 testes.

Defeitos identificados nos testes anteriores da H2 (e só neles):
  (a) vol em NÍVEIS com winsorização de níveis através de regimes — clipava a
      vol de 2020 nos limites de 2013-19, esmagando a variação relevante;
      padrão da literatura: log(vol) (processo multiplicativo);
  (b) horizonte 20d (que está na Tabela 4 do paper) nunca passou pela
      validação temporal — só o 120d.
PRÉ-COMPROMISSO declarado ao Marcelo antes de rodar: 4 células —
{log vol_20, log vol_120} x {walk-forward trimestral, previsão condicional
por resíduo} — todas reportadas; resultado encerra a H2, sem redesenhos.
Sinal: td_w. Controles e FE de sempre. Winsor (em log) com limites do treino.
Uso: python scripts/sp500_h2_final.py
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
HOR = {"lv20": 35, "lv120": 175}     # dias corridos p/ conclusão do resultado


def load():
    ev = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")
    ev = ev.merge(pd.read_parquet(OUTDIR / "td_variants.parquet")[["call_id", "td_w"]],
                  on="call_id", how="left")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    ev = ev[ev["cdate"] >= "2009-01-01"].copy()
    for src, dst in [("vol_20", "lv20"), ("vol_120", "lv120")]:
        v = pd.to_numeric(ev[src], errors="coerce")
        ev[dst] = np.log(v.where(v > 0))
    ev["yq"] = ev["cdate"].dt.to_period("Q")
    return ev


def beta(d, yv):
    dd = d.dropna(subset=[yv, "td_w"] + CTRL + ["ticker", "year_quarter"]).copy()
    if len(dd) < 800 or dd["ticker"].nunique() < 50:
        return None
    for c in [yv, "td_w"] + CTRL:
        dd[c] = dd[c].clip(dd[c].quantile(0.05), dd[c].quantile(0.95))
    m = smf.ols(f"{yv} ~ td_w + " + " + ".join(CTRL) + " + C(ticker) + C(year_quarter)",
                data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["ticker"]})
    return float(m.params["td_w"]), float(m.tvalues["td_w"])


def walkforward(ev, yv):
    cuts = pd.period_range("2012Q4", "2024Q4", freq="Q")
    ok_p = ok_f = tot = 0
    H = pd.Timedelta(days=HOR[yv])
    for cq in cuts:
        cut = cq.end_time.normalize()
        p = beta(ev[ev["cdate"] + H <= cut], yv)
        f = beta(ev[ev["cdate"] > cut], yv)
        if p is None or f is None:
            continue
        tot += 1
        ok_p += int(p[0] > 0)
        ok_f += int(f[0] > 0)
    print(f"  walk-forward trimestral {yv}: passado>0 em {ok_p}/{tot} | "
          f"FUTURO>0 em {ok_f}/{tot}")


def conditional(ev, yv):
    spreads, hits = [], []
    for y in range(2013, 2026):
        cut = pd.Timestamp(f"{y-1}-12-31") - pd.Timedelta(days=HOR[yv])
        tr = ev[ev["cdate"] <= cut].dropna(subset=[yv] + CTRL + ["ticker", "year_quarter"]).copy()
        if len(tr) < 800:
            continue
        lims = {c: (tr[c].quantile(0.05), tr[c].quantile(0.95)) for c in [yv] + CTRL}
        for c in [yv] + CTRL:
            tr[c] = tr[c].clip(*lims[c])
        m = smf.ols(f"{yv} ~ " + " + ".join(CTRL) + " + C(ticker) + C(year_quarter)",
                    data=tr).fit()
        fe = {t: m.params.get(f"C(ticker)[T.{t}]", np.nan) for t in tr["ticker"].unique()}
        base_t = tr["ticker"].iloc[0]
        fut = ev[ev["cdate"].dt.year == y].dropna(subset=[yv, "td_w"] + CTRL).copy()
        fut = fut[fut["ticker"].isin(tr["ticker"].unique())]
        if len(fut) < 200:
            continue
        for c in [yv] + CTRL:
            fut[c] = fut[c].clip(*lims[c])
        pred = m.params["Intercept"] + sum(m.params[c] * fut[c] for c in CTRL)
        pred = pred + fut["ticker"].map(
            lambda t: 0.0 if t == base_t else fe.get(t, np.nan)).astype(float)
        fut["resid"] = fut[yv] - pred
        fut = fut.dropna(subset=["resid"])
        for q in range(1, 5):
            sub = fut[fut["yq"] == pd.Period(f"{y}Q{q}")]
            if len(sub) < 60:
                continue
            med = sub["td_w"].median()
            sp = float(sub.loc[sub["td_w"] > med, "resid"].mean()
                       - sub.loc[sub["td_w"] <= med, "resid"].mean())
            spreads.append(sp)
            hits.append(int(sp > 0))
    sp = np.array(spreads)
    t = sp.mean() / sp.std() * np.sqrt(len(sp)) if len(sp) > 4 else np.nan
    print(f"  condicional {yv}: {len(sp)} tri | acerto {100*np.mean(hits):.0f}% | "
          f"spread médio (em log) {sp.mean():+.4f} | t {t:+.2f}")


def main():
    ev = load()
    for yv in ["lv20", "lv120"]:
        print(f"\n===== H2 FINAL — log({yv.replace('lv','vol_')}) ~ td_w =====")
        walkforward(ev, yv)
        conditional(ev, yv)
    print("\nDONE. (encerra a H2 — pré-compromisso de 24/07)")


if __name__ == "__main__":
    main()
