# -*- coding: utf-8 -*-
"""Reconciliação da H2 com o paper: a janela de vol COM o dia +1 (eco do evento).

Paper: vol "over the 20 days following the announcement" => [t0+1, t0+20].
Nossa: [t0+2, t0+21] (fora da janela do evento). Hipótese da reconciliação:
a H2 do paper é, em parte, eco mecânico da turbulência do próprio anúncio
(dia +1), que é maior p/ TD alta (H1); removido o eco, o canal morre.

Testes (rotulados como RECONCILIAÇÃO, não validação — pré-compromisso da H2
segue de pé):
  1. painel em amostra cheia: beta de log vol janela-paper vs janela-nossa
     (e a versão só-eco: vol dos dias [t0+1, t0+2]);
  2. walk-forward trimestral (passado/futuro) da janela-paper;
  3. previsão condicional por resíduo da janela-paper.
Se a janela-paper "ressuscitar" o beta e — só ela — sobreviver ao tempo, o
"onde" está achado: o previsível é a turbulência imediata pós-evento, não o
risco de médio prazo. Uso: python scripts/sp500_h2_reconcile.py
"""
from __future__ import annotations
import glob, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "interim" / "sp500"
CTRL = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
        "length", "ln_mktcap", "industry_tone_ff49", "lagged_td", "sue_pct"]


def build_vols(ev):
    px = pd.concat([pd.read_parquet(p) for p in
                    glob.glob(str(ROOT / "data/raw/sp500/price_shards/*.parquet"))],
                   ignore_index=True).sort_values(["ticker", "date"])
    px["date"] = pd.to_datetime(px["date"])
    tick = {}
    for t, g in px[px["ticker"] != "^GSPC"].groupby("ticker"):
        g = g.dropna(subset=["adj_close"])
        tick[t] = (g["date"].to_numpy("datetime64[ns]"),
                   pd.to_numeric(g["adj_close"], errors="coerce").pct_change().to_numpy(dtype=float))
    vp20 = np.full(len(ev), np.nan)
    vp120 = np.full(len(ev), np.nan)
    veco = np.full(len(ev), np.nan)     # só dias +1 e +2 (o "eco")
    for k, r in enumerate(ev.itertuples()):
        d = tick.get(r.ticker)
        if d is None:
            continue
        dates, ret = d
        t0 = int(np.searchsorted(dates, np.datetime64(r.cdate)))
        if t0 + 121 >= len(ret):
            continue
        w20 = ret[t0 + 1:t0 + 21]
        w120 = ret[t0 + 1:t0 + 121]
        we = ret[t0 + 1:t0 + 3]
        if np.isfinite(w20).sum() >= 15:
            vp20[k] = np.nanstd(w20)
        if np.isfinite(w120).sum() >= 90:
            vp120[k] = np.nanstd(w120)
        if np.isfinite(we).sum() == 2:
            veco[k] = float(np.abs(we).mean())
    ev["lvp20"] = np.log(pd.Series(vp20).where(pd.Series(vp20) > 0)).to_numpy()
    ev["lvp120"] = np.log(pd.Series(vp120).where(pd.Series(vp120) > 0)).to_numpy()
    ev["leco"] = np.log(pd.Series(veco).where(pd.Series(veco) > 0)).to_numpy()
    return ev


def beta(d, yv):
    dd = d.dropna(subset=[yv, "td_w"] + CTRL + ["ticker", "year_quarter"]).copy()
    if len(dd) < 800 or dd["ticker"].nunique() < 50:
        return None
    for c in [yv, "td_w"] + CTRL:
        dd[c] = dd[c].clip(dd[c].quantile(0.05), dd[c].quantile(0.95))
    m = smf.ols(f"{yv} ~ td_w + " + " + ".join(CTRL) + " + C(ticker) + C(year_quarter)",
                data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["ticker"]})
    return float(m.params["td_w"]), float(m.tvalues["td_w"]), len(dd)


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")
    ev = ev.merge(pd.read_parquet(OUTDIR / "td_variants.parquet")[["call_id", "td_w"]],
                  on="call_id", how="left")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    ev = ev[ev["cdate"] >= "2009-01-01"].reset_index(drop=True)
    ev = build_vols(ev)
    for c_out, dst in [("vol_20", "lv20"), ("vol_120", "lv120")]:
        v = pd.to_numeric(ev[c_out], errors="coerce")
        ev[dst] = np.log(v.where(v > 0))
    ev["yq"] = ev["cdate"].dt.to_period("Q")

    print("=== 1) PAINEL AMOSTRA CHEIA — a janela importa? ===")
    for yv, lbl in [("lv20", "nossa [+2,+21]"), ("lvp20", "PAPER [+1,+20]"),
                    ("leco", "só o eco [+1,+2]"), ("lv120", "nossa [+2,+121]"),
                    ("lvp120", "PAPER [+1,+120]")]:
        r = beta(ev, yv)
        if r:
            print(f"  {lbl:>18s}: coef={r[0]:+8.4f} t={r[1]:+6.2f} (n={r[2]})")

    print("\n=== 2) WALK-FORWARD TRIMESTRAL da janela-paper ===")
    H = pd.Timedelta(days=35)
    for yv in ["lvp20"]:
        cuts = pd.period_range("2012Q4", "2024Q4", freq="Q")
        ok_p = ok_f = tot = 0
        for cq in cuts:
            cut = cq.end_time.normalize()
            p = beta(ev[ev["cdate"] + H <= cut], yv)
            f = beta(ev[ev["cdate"] > cut], yv)
            if p is None or f is None:
                continue
            tot += 1
            ok_p += int(p[0] > 0)
            ok_f += int(f[0] > 0)
        print(f"  {yv}: passado>0 em {ok_p}/{tot} | FUTURO>0 em {ok_f}/{tot}")

    print("\n=== 3) PREVISÃO CONDICIONAL da janela-paper (lvp20) ===")
    spreads, hits = [], []
    for y in range(2013, 2026):
        cut = pd.Timestamp(f"{y-1}-12-31") - pd.Timedelta(days=35)
        tr = ev[ev["cdate"] <= cut].dropna(subset=["lvp20"] + CTRL + ["ticker", "year_quarter"]).copy()
        if len(tr) < 800:
            continue
        lims = {c: (tr[c].quantile(0.05), tr[c].quantile(0.95)) for c in ["lvp20"] + CTRL}
        for c in ["lvp20"] + CTRL:
            tr[c] = tr[c].clip(*lims[c])
        m = smf.ols("lvp20 ~ " + " + ".join(CTRL) + " + C(ticker) + C(year_quarter)",
                    data=tr).fit()
        fe = {t: m.params.get(f"C(ticker)[T.{t}]", np.nan) for t in tr["ticker"].unique()}
        base_t = tr["ticker"].iloc[0]
        fut = ev[ev["cdate"].dt.year == y].dropna(subset=["lvp20", "td_w"] + CTRL).copy()
        fut = fut[fut["ticker"].isin(tr["ticker"].unique())]
        if len(fut) < 200:
            continue
        for c in ["lvp20"] + CTRL:
            fut[c] = fut[c].clip(*lims[c])
        pred = m.params["Intercept"] + sum(m.params[c] * fut[c] for c in CTRL)
        pred = pred + fut["ticker"].map(
            lambda t: 0.0 if t == base_t else fe.get(t, np.nan)).astype(float)
        fut["resid"] = fut["lvp20"] - pred
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
    if len(sp) > 4:
        t = sp.mean() / sp.std() * np.sqrt(len(sp))
        print(f"  lvp20: {len(sp)} tri | acerto {100*np.mean(hits):.0f}% | "
              f"spread {sp.mean():+.4f} | t {t:+.2f}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
