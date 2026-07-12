# -*- coding: utf-8 -*-
"""Replicação S&P 500 — FASE C: montagem + sanity checks + regressões do Angelo.

Junta a Fase A (TD_LM + controles textuais) com a Fase B (mercado) e estima:

  H1:      CAR[-1,+1]_i = β·TD_i + γ'Controles_i + FE_firma + FE_trimestre + ε
  Prêmio:  ret_21 / ret_63 (close T+1 -> close T+1+h)  ~  idem
  (cluster por firma; winsor 5/95 em tudo; TD padronizada -> β por SD)

Construções:
  - CAR: market model por evento (estimação [-120,-21], >=60 obs; evento [-1,+1]
    ancorado no 1º pregão >= data da call). Mercado = ^GSPC.
  - SUE: (EPS_reportado − EPS_estimado)/max(|estimado|, 0.01), match ±3 dias.
  - ln_mktcap: raw_close (asof estrito) × shares (asof; pré-1ª obs = extrapolado
    com flag). industry_tone: média PIT 90d do disclosure_tone do MESMO setor
    (sector_cache; estrito, >=3 pares).

Ordem de leitura PRÉ-COMPROMETIDA: sanity checks primeiro (beat rate ~75-80%,
sd do CAR < 7,6% de tech, cobertura); só então os β. Specs idênticas às de tech.

Uso: python scripts/sp500_phase_c.py
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW, INT = ROOT / "data" / "raw", ROOT / "data" / "interim"
SP = RAW / "sp500"
OUTDIR = INT / "sp500"
EST_LO, EST_HI, MIN_EST = -120, -21, 60
E_FLOOR, MAX_GAP_D = 0.01, 3
PEER_WINDOW_D, PEER_MIN = 90, 3
CONTROLS = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
            "length", "ln_mktcap", "industry_tone", "lagged_td", "sue_pct"]


def winsor(s, lo=0.05, hi=0.95):
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def zscore(s):
    return (s - s.mean()) / s.std()


def run(df, yvar, tdvar, label):
    d = df.dropna(subset=[yvar, tdvar] + CONTROLS + ["ticker", "year_quarter"]).copy()
    if len(d) < 100:
        return dict(spec=label, n=len(d), coef=np.nan, t=np.nan, p=np.nan, r2=np.nan)
    d["_td"] = zscore(winsor(d[tdvar]))
    d["_y"] = winsor(d[yvar])
    for c in CONTROLS:
        d[c] = winsor(d[c])
    m = smf.ols("_y ~ _td + " + " + ".join(CONTROLS) + " + C(ticker) + C(year_quarter)",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    return dict(spec=label, n=len(d), coef=m.params["_td"], t=m.tvalues["_td"],
                p=m.pvalues["_td"], r2=m.rsquared, n_firms=d["ticker"].nunique())


def main():
    td = pd.read_parquet(OUTDIR / "tone_distance_sp500.parquet")
    td["cdate"] = pd.to_datetime(td["call_datetime"], utc=True).dt.tz_convert("US/Eastern") \
                    .dt.tz_localize(None).dt.normalize()
    td["year_quarter"] = td["year"].astype(int).astype(str) + "Q" + td["quarter"].astype(int).astype(str)

    px = pd.concat([pd.read_parquet(p) for p in (SP / "price_shards").glob("*.parquet")],
                   ignore_index=True).sort_values(["ticker", "date"])
    mkt = px[px["ticker"] == "^GSPC"].set_index("date")["adj_close"]
    mkt_ret = mkt.pct_change()
    print(f"preços: {px['ticker'].nunique()-1} tickers | mercado ^GSPC {len(mkt)} pregões")

    # ---------- CAR + retornos futuros por evento ----------
    tick_data = {}
    for t, g in px[px["ticker"] != "^GSPC"].groupby("ticker"):
        g = g.dropna(subset=["adj_close"])
        r = g["adj_close"].pct_change().to_numpy()
        rm = mkt_ret.reindex(g["date"]).to_numpy()
        tick_data[t] = (g["date"].to_numpy("datetime64[ns]"), r, rm,
                        g["adj_close"].to_numpy(), g["raw_close"].to_numpy())
    cars, r21s, r63s = [], [], []
    for row in td.itertuples():
        car = r21 = r63 = np.nan
        data = tick_data.get(row.ticker)
        if data is not None:
            dates, r, rm, ac, _ = data
            p = int(np.searchsorted(dates, np.datetime64(row.cdate)))
            if 0 < p + EST_LO and p + 1 < len(r):
                sl = slice(p + EST_LO, p + EST_HI + 1)
                rr, mm = r[sl], rm[sl]
                ok = np.isfinite(rr) & np.isfinite(mm)
                if ok.sum() >= MIN_EST:
                    b, a = np.polyfit(mm[ok], rr[ok], 1)
                    ar = r[p-1:p+2] - (a + b * rm[p-1:p+2])
                    if np.isfinite(ar).all():
                        car = float(ar.sum())
            if p + 1 < len(ac):
                base = ac[p + 1]
                if p + 22 < len(ac):
                    r21 = float(ac[p + 22] / base - 1)
                if p + 64 < len(ac):
                    r63 = float(ac[p + 64] / base - 1)
        cars.append(car); r21s.append(r21); r63s.append(r63)
    td["car"], td["ret_21"], td["ret_63"] = cars, r21s, r63s

    # ---------- SUE ----------
    ea = pd.concat([pd.read_parquet(p) for p in (SP / "earnings_shards").glob("*.parquet")],
                   ignore_index=True).dropna(subset=["eps_est", "eps_act"])
    ea_g = {t: g.sort_values("edate").reset_index(drop=True) for t, g in ea.groupby("ticker")}
    sues = []
    for row in td.itertuples():
        s = np.nan
        g = ea_g.get(row.ticker)
        if g is not None:
            gaps = (g["edate"] - row.cdate).dt.days.abs()
            j = int(gaps.idxmin())
            if gaps.iloc[j] <= MAX_GAP_D:
                a, e = float(g["eps_act"].iloc[j]), float(g["eps_est"].iloc[j])
                s = (a - e) / max(abs(e), E_FLOOR)
        sues.append(s)
    td["sue_pct"] = sues

    # ---------- ln_mktcap ----------
    sh = pd.concat([pd.read_parquet(p) for p in (SP / "shares_shards").glob("*.parquet")],
                   ignore_index=True).sort_values(["ticker", "date"])
    sh_g = {t: g.reset_index(drop=True) for t, g in sh.groupby("ticker")}
    lnmc, extrap = [], []
    for row in td.itertuples():
        v, ex = np.nan, False
        data = tick_data.get(row.ticker)
        g = sh_g.get(row.ticker)
        if data is not None and g is not None and len(g):
            dates, _, _, _, rc = data
            p = int(np.searchsorted(dates, np.datetime64(row.cdate))) - 1
            if p >= 0:
                j = int(np.searchsorted(g["date"].to_numpy("datetime64[ns]"),
                                        np.datetime64(row.cdate), side="right")) - 1
                if j >= 0:
                    n_sh = float(g["shares"].iloc[j])
                else:
                    n_sh, ex = float(g["shares"].iloc[0]), True
                if rc[p] > 0 and n_sh > 0:
                    v = float(np.log(rc[p] * n_sh))
        lnmc.append(v); extrap.append(ex)
    td["ln_mktcap"], td["mktcap_extrapolated"] = lnmc, extrap

    # ---------- industry tone (setor real, PIT 90d estrito) ----------
    sec = pd.read_parquet(RAW / "sector_cache.parquet").set_index("ticker")["sector"].to_dict()
    td["sector"] = td["ticker"].map(sec)
    td = td.sort_values("cdate").reset_index(drop=True)
    ind = np.full(len(td), np.nan)
    for s_name, g in td.groupby("sector"):
        if not isinstance(s_name, str):
            continue
        idx = g.index.to_numpy()
        times = g["cdate"].to_numpy("datetime64[ns]")
        vals = g["disclosure_tone"].to_numpy(float)
        tks = g["ticker"].to_numpy()
        win = np.timedelta64(PEER_WINDOW_D, "D")
        lo = np.searchsorted(times, times - win, side="left")
        hi = np.searchsorted(times, times, side="left")
        for i in range(len(idx)):
            sl = slice(lo[i], hi[i])
            mask = tks[sl] != tks[i]
            v = vals[sl][mask]
            v = v[np.isfinite(v)]
            if len(v) >= PEER_MIN:
                ind[idx[i]] = v.mean()
    td["industry_tone"] = ind

    dest = OUTDIR / "events_sp500.parquet"
    td.to_parquet(dest, index=False)

    # ================= SANITY CHECKS (antes de qualquer β) =================
    print("\n" + "=" * 80)
    print("SANITY CHECKS PRÉ-COMPROMETIDOS")
    print("=" * 80)
    print(f"calls com TD: {len(td)} | com CAR: {td['car'].notna().sum()} | com SUE: "
          f"{td['sue_pct'].notna().sum()} ({100*td['sue_pct'].notna().mean():.0f}%) | "
          f"com mktcap: {td['ln_mktcap'].notna().sum()} (extrap: {int(td['mktcap_extrapolated'].sum())}) | "
          f"com industry_tone: {td['industry_tone'].notna().sum()}")
    print(f"beat rate do SUE: {100*(td['sue_pct'] > 0).mean() / td['sue_pct'].notna().mean():.0f}%"
          f"  (esperado ~75-80%)")
    print(f"sd do CAR[-1,+1]: {100*td['car'].std():.1f}%  (tech era 7,6%; esperado menor)")
    print(f"setores: {td['sector'].nunique()} | sem setor: {td['sector'].isna().sum()} calls")

    # ================= REGRESSÕES =================
    for yv, lbl, expct in [("car", "H1 — CAR[-1,+1] no anúncio", "esperado NEGATIVO"),
                           ("ret_21", "PRÊMIO 1 mês", "esperado POSITIVO"),
                           ("ret_63", "PRÊMIO 3 meses", "esperado POSITIVO")]:
        print("\n" + "=" * 80)
        print(f"{lbl} ({expct}) | FE firma+trimestre, cluster firma, winsor 5/95")
        print("=" * 80)
        print(f"{'spec':22s} {'n':>6s} {'firmas':>6s} {'coef(SD)':>10s} {'t':>7s} {'p':>7s} {'R2':>6s}")
        for tdv, sl in [("td", "TD crua (fiel)"), ("td_adj", "TD_adj (null)"), ("z", "z (null)")]:
            r = run(td, yv, tdv, sl)
            print(f"{sl:22s} {r['n']:>6d} {r.get('n_firms', 0):>6d} {r['coef']:>10.4f} "
                  f"{r['t']:>7.2f} {r['p']:>7.3f} {r['r2']:>6.3f}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
