# -*- coding: utf-8 -*-
"""Caça ao erro na H2 (ordem do Marcelo, 24/07) — ancoragem pelo DIA DE REAÇÃO.

Suspeito principal remanescente: heterogeneidade de ancoragem. 73% das calls
são pós-fechamento => reação em t0+1; para elas, a janela antiga [t0+2,...]
começa 1 dia após a reação, mas para calls pré-abertura começa 2 dias após.
Janelas heterogêneas em relação ao evento real = ruído sistemático na
variável da H2, herdado por TODOS os testes anteriores.

Correção: r0 = dia de reação efetivo (hora>=16h ET => pregão seguinte; sem
horário => data da call). Janelas re-ancoradas:
  clean : [r0+1, r0+h]   (risco futuro puro, sem o dia da reação)
  echo  : [r0,   r0+h-1] (inclui o dia da reação, p/ contraste)
Bateria pré-especificada (16 células): {clean,echo} x {20,120} x {log,nível}
x {td_w, td_b}; por célula: beta amostra cheia + walk-forward ANUAL
(passado/futuro) + previsão condicional (t). Antes: auditoria independente
do td_w (recomputação das contagens) — último insumo não auditado.
Tudo reportado, seja qual for o resultado. Uso: python scripts/sp500_h2_hunt.py
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


def audit_tdw(ev):
    sc = pd.read_parquet(OUTDIR / "speaker_counts.parquet")
    mg = sc[(sc["role_eff"] == "manager") & (sc["mk"] != "")]
    per = mg.groupby(["call_id", "mk"]).agg(nw=("nw", "sum"), npos=("npos", "sum"),
                                            nneg=("nneg", "sum")).reset_index()
    per = per[per["nw"] > 0]
    per["fp"] = per["npos"] / per["nw"]; per["fn"] = per["nneg"] / per["nw"]
    cm = per.groupby("call_id")[["fp", "fn"]].transform("mean")
    d = np.sqrt((per["fp"] - cm["fp"]) ** 2 + (per["fn"] - cm["fn"]) ** 2)
    w = per["nw"].astype(float)
    num = (d * w).groupby(per["call_id"]).sum()
    den = w.groupby(per["call_id"]).sum()
    tdw_re = (num / den).rename("tdw_re").reset_index()
    nmg = per.groupby("call_id").size()
    tdw_re = tdw_re[tdw_re["call_id"].isin(nmg[nmg >= 2].index)]
    cmp_ = ev[["call_id", "td_w"]].merge(tdw_re, on="call_id", how="inner").dropna()
    diff = (cmp_["td_w"] - cmp_["tdw_re"]).abs()
    print(f"=== AUDITORIA td_w: {len(cmp_)} calls | corr={cmp_['td_w'].corr(cmp_['tdw_re']):.6f} "
          f"| max|diff|={diff.max():.2e} ===", flush=True)


def build_reanchored(ev):
    meta = pd.read_parquet(ROOT / "data" / "raw" / "calls_all.parquet")[["call_id", "has_time"]]
    ev = ev.merge(meta, on="call_id", how="left")
    cdt = pd.to_datetime(ev["call_datetime"], utc=True).dt.tz_convert("US/Eastern")
    ev["hour"] = cdt.dt.hour + cdt.dt.minute / 60.0

    px = pd.concat([pd.read_parquet(p) for p in
                    glob.glob(str(ROOT / "data/raw/sp500/price_shards/*.parquet"))],
                   ignore_index=True).sort_values(["ticker", "date"])
    px["date"] = pd.to_datetime(px["date"])
    tick = {}
    for t, g in px[px["ticker"] != "^GSPC"].groupby("ticker"):
        g = g.dropna(subset=["adj_close"])
        tick[t] = (g["date"].to_numpy("datetime64[ns]"),
                   pd.to_numeric(g["adj_close"], errors="coerce").pct_change().to_numpy(dtype=float))
    cols = {f"{a}{h}": np.full(len(ev), np.nan) for a in ("clean", "echo") for h in (20, 120)}
    n_shift = 0
    for k, r in enumerate(ev.itertuples()):
        d = tick.get(r.ticker)
        if d is None:
            continue
        dates, ret = d
        D = np.datetime64(pd.Timestamp(r.cdate))
        idx = int(np.searchsorted(dates, D))
        if idx >= len(dates):
            continue
        if bool(r.has_time) and dates[idx] == D and r.hour >= 16.0:
            r0 = idx + 1
            n_shift += 1
        else:
            r0 = idx
        if r0 + 121 >= len(ret):
            continue
        for h in (20, 120):
            wc = ret[r0 + 1:r0 + 1 + h]
            we = ret[r0:r0 + h]
            if np.isfinite(wc).sum() >= h * 0.75:
                cols[f"clean{h}"][k] = np.nanstd(wc)
            if np.isfinite(we).sum() >= h * 0.75:
                cols[f"echo{h}"][k] = np.nanstd(we)
    for c, v in cols.items():
        s = pd.Series(v)
        ev[f"lg_{c}"] = np.log(s.where(s > 0)).to_numpy()
        ev[f"lv_{c}"] = s.to_numpy()
    print(f"re-ancoragem: {n_shift} calls pós-16h deslocadas p/ pregão seguinte "
          f"({100*n_shift/len(ev):.0f}%)", flush=True)
    return ev


def beta(d, yv, sig):
    dd = d.dropna(subset=[yv, sig] + CTRL + ["ticker", "year_quarter"]).copy()
    if len(dd) < 800 or dd["ticker"].nunique() < 50:
        return None
    for c in [yv, sig] + CTRL:
        dd[c] = dd[c].clip(dd[c].quantile(0.05), dd[c].quantile(0.95))
    m = smf.ols(f"{yv} ~ {sig} + " + " + ".join(CTRL) + " + C(ticker) + C(year_quarter)",
                data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["ticker"]})
    return float(m.params[sig]), float(m.tvalues[sig]), len(dd)


def wf_annual(ev, yv, sig, hdays):
    H = pd.Timedelta(days=hdays)
    ok_f = tot = 0
    for y in range(2012, 2025):
        cut = pd.Timestamp(f"{y}-12-31")
        p = beta(ev[ev["cdate"] + H <= cut], yv, sig)
        f = beta(ev[ev["cdate"] > cut], yv, sig)
        if p is None or f is None:
            continue
        tot += 1
        ok_f += int(f[0] > 0)
    return f"{ok_f}/{tot}"


def cond(ev, yv, sig, hdays):
    spreads = []
    for y in range(2013, 2026):
        cut = pd.Timestamp(f"{y-1}-12-31") - pd.Timedelta(days=hdays)
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
        fut = ev[ev["cdate"].dt.year == y].dropna(subset=[yv, sig] + CTRL).copy()
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
            med = sub[sig].median()
            spreads.append(float(sub.loc[sub[sig] > med, "resid"].mean()
                                 - sub.loc[sub[sig] <= med, "resid"].mean()))
    sp = np.array(spreads)
    if len(sp) < 5:
        return "n/a"
    return f"t={sp.mean()/sp.std()*np.sqrt(len(sp)):+.2f}"


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")
    ev = ev.merge(pd.read_parquet(OUTDIR / "td_variants.parquet")[["call_id", "td_w"]],
                  on="call_id", how="left")
    ev = ev.merge(pd.read_parquet(OUTDIR / "td_poolcent.parquet")[["call_id", "td_poolcent"]]
                    .rename(columns={"td_poolcent": "td_b"}), on="call_id", how="left")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    ev = ev[ev["cdate"] >= "2009-01-01"].reset_index(drop=True)
    ev["yq"] = ev["cdate"].dt.to_period("Q")
    audit_tdw(ev)
    ev = build_reanchored(ev)

    print("\n=== BATERIA H2 RE-ANCORADA (16 células pré-especificadas) ===")
    print(f"{'célula':>22s} | {'B cheia':>8s} {'t':>6s} | {'WF fut>0':>8s} | {'condicional':>11s}")
    for anchor in ("clean", "echo"):
        for h in (20, 120):
            for tf in ("lg", "lv"):
                yv = f"{tf}_{anchor}{h}"
                for sig in ("td_w", "td_b"):
                    hdays = 35 if h == 20 else 175
                    b = beta(ev, yv, sig)
                    if b is None:
                        print(f"{yv+'~'+sig:>22s} | amostra insuficiente")
                        continue
                    wf = wf_annual(ev, yv, sig, hdays)
                    cd = cond(ev, yv, sig, hdays)
                    print(f"{yv+'~'+sig:>22s} | {b[0]:>+8.4f} {b[1]:>+6.2f} | {wf:>8s} | {cd:>11s}",
                          flush=True)
    print("\nDONE.")


if __name__ == "__main__":
    main()
