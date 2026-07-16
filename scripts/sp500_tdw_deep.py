# -*- coding: utf-8 -*-
"""Aprofundamento da TD PONDERADA POR PALAVRAS (Tabela 8 col 1 do Angelo).

Motivação: com a TD ponderada (spec do próprio paper, que lá também é MAIS
forte que a base: -0.77 vs -0.49), o H1 aparece na nossa amostra
(-0.41, t=-2.86, amostra ampla 2009+, controles antigos). Antes de aceitar:
  1. As três janelas de CAR (paper: todas negativas);
  2. Winsorização 1/99 (nota 3 do paper);
  3. Subperíodos (2009-2016 / 2017-2025 / 2005+ completo — o braço de
     controles antigos não depende do EDGAR, então 2005+ é legítimo aqui);
  4. Efeito econômico interquartil (paper: -0.21% no CAR de 1 dia);
  5. H2 (vol futura) e Tabela 6 (retorno mensal) com td_w — a história de
     risco compensado deve valer também na variante ponderada.
Uso: python scripts/sp500_tdw_deep.py
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "data" / "raw" / "sp500"
OUTDIR = ROOT / "data" / "interim" / "sp500"

OLD = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
       "length", "ln_mktcap", "industry_tone", "lagged_td", "sue_pct"]
PAPER = ["disclosure_tone", "lagged_avg_td", "sue_pct", "btm", "lev",
         "roa", "std_roa", "std_cfo", "rd_at", "etr", "smooth",
         "length", "std_me", "ln_assets",
         "analyst_tone_disp", "analyst_tone", "industry_tone"]


def winsor(s, lo=0.05, hi=0.95):
    return s.clip(s.quantile(lo), s.quantile(hi))


def run(df, yvar, tdvar, controls, label, lo=0.05, hi=0.95):
    d = df.dropna(subset=[yvar, tdvar] + controls + ["ticker", "year_quarter"]).copy()
    if len(d) < 500:
        print(f"{label:52s} n insuficiente ({len(d)})", flush=True)
        return
    for c in [yvar, tdvar] + controls:
        d[c] = winsor(d[c], lo, hi)
    iqr = d[tdvar].quantile(0.75) - d[tdvar].quantile(0.25)
    m = smf.ols(f"{yvar} ~ {tdvar} + " + " + ".join(controls)
                + " + C(ticker) + C(year_quarter)",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    b, t, p = m.params[tdvar], m.tvalues[tdvar], m.pvalues[tdvar]
    print(f"{label:52s} n={len(d):>6d} | coef={b:+8.4f} t={t:+6.2f} p={p:.3f} "
          f"| IQR->y={100*b*iqr:+.3f}%", flush=True)


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")
    var = pd.read_parquet(OUTDIR / "td_variants.parquet")
    ev = ev.merge(var, on="call_id", how="left")
    ev["cdate"] = pd.to_datetime(ev["cdate"])

    full = ev                                   # 2005+ (sem exigência EDGAR)
    e09 = ev[ev["cdate"] >= "2009-01-01"]
    e0916 = ev[(ev["cdate"] >= "2009-01-01") & (ev["cdate"] <= "2016-12-31")]
    e17 = ev[ev["cdate"] >= "2017-01-01"]

    print("===== 1-2) Eq.(3) com td_w — janelas e winsorização =====")
    for yv, wl in [("car_m1p1", "CAR[-1,+1]"), ("car_m1p2", "CAR[-1,+2]"), ("car_m1p5", "CAR[-1,+5]")]:
        run(e09, yv, "td_w", OLD, f"{wl} 2009+ ctrl antigos 5/95")
        run(e09, yv, "td_w", OLD, f"{wl} 2009+ ctrl antigos 1/99", lo=0.01, hi=0.99)
    print()
    for yv, wl in [("car_m1p1", "CAR[-1,+1]"), ("car_m1p2", "CAR[-1,+2]"), ("car_m1p5", "CAR[-1,+5]")]:
        run(e09, yv, "td_w", PAPER, f"{wl} 2009+ ctrl PAPER (comum)")

    print("\n===== 3) subperíodos (ctrl antigos, CAR[-1,+1]) =====")
    run(full, "car_m1p1", "td_w", OLD, "2005-2025 (amostra máxima)")
    run(e0916, "car_m1p1", "td_w", OLD, "2009-2016")
    run(e17, "car_m1p1", "td_w", OLD, "2017-2025")

    print("\n===== 5a) H2 com td_w (vol futura; ctrl antigos, amostra ampla) =====")
    for yv in ["vol_20", "vol_120"]:
        run(e09, yv, "td_w", OLD, f"{yv} 2009+")
        run(full, yv, "td_w", OLD, f"{yv} 2005+")

    # ---------- 5b) Tabela 6 com td_w ----------
    print("\n===== 5b) Tabela 6 (retorno mensal, 3m) com td_w =====")
    px = pd.concat([pd.read_parquet(p) for p in (SP / "price_shards").glob("*.parquet")],
                   ignore_index=True).sort_values(["ticker", "date"])
    px = px[px["ticker"] != "^GSPC"].dropna(subset=["adj_close"])
    px["date"] = pd.to_datetime(px["date"])
    px["ym"] = px["date"].dt.to_period("M")
    px["r"] = px.groupby("ticker")["adj_close"].pct_change()
    mret = (px.groupby(["ticker", "ym"])["r"]
              .apply(lambda s: float(np.prod(1.0 + s.dropna()) - 1.0))
              .rename("ret_m").reset_index()).sort_values(["ticker", "ym"])
    g = mret.groupby("ticker")["ret_m"]
    mret["_l1p"] = np.log1p(mret["ret_m"])
    cum = mret.groupby("ticker")["_l1p"].transform(lambda s: s.rolling(11, min_periods=11).sum())
    mret["momentum"] = np.expm1(cum.groupby(mret["ticker"]).shift(2))
    mret["reversal"] = g.shift(1)

    evm = ev.dropna(subset=["td_w"]).copy()
    evm["ym_call"] = evm["cdate"].dt.to_period("M")
    evm = (evm.sort_values(["ticker", "cdate"])
              .drop_duplicates(["ticker", "ym_call"], keep="last"))
    frames = []
    for gap in (1, 2, 3):
        f = evm[["ticker", "ym_call", "td_w", "btm", "ln_mktcap"]].copy()
        f["ym"] = f["ym_call"] + gap
        f["gap"] = gap
        frames.append(f)
    link = (pd.concat(frames, ignore_index=True)
              .sort_values(["ticker", "ym", "gap"])
              .drop_duplicates(["ticker", "ym"], keep="first"))
    panel = mret.merge(link, on=["ticker", "ym"], how="inner")
    panel["size"] = panel["ln_mktcap"]
    panel["ym"] = panel["ym"].astype(str)

    def demean_two_way(df, cols, f1, f2, tol=1e-10, max_iter=50):
        X = df[cols].astype(float).copy()
        for _ in range(max_iter):
            X0 = X.copy()
            X = X - X.groupby(df[f1]).transform("mean")
            X = X - X.groupby(df[f2]).transform("mean")
            if float((X - X0).abs().max().max()) < tol:
                break
        return X

    def run_t6(d, controls, label, wins=(0.05, 0.95)):
        cols = ["ret_m", "td_w"] + controls
        d = d.dropna(subset=cols + ["ticker", "ym"]).copy()
        for c in cols:
            d[c] = winsor(d[c], *wins)
        Xd = demean_two_way(d, cols, "ticker", "ym")
        m = sm.OLS(Xd["ret_m"], Xd[["td_w"] + controls]).fit(
            cov_type="cluster", cov_kwds={"groups": d["ym"]})
        print(f"{label:52s} n={len(d):>6d} | coef={m.params['td_w']:+8.4f} "
              f"t={m.tvalues['td_w']:+6.2f} p={m.pvalues['td_w']:.3f}", flush=True)

    run_t6(panel, [], "T6 col1 (só FE) td_w 5/95")
    run_t6(panel, ["btm", "momentum", "size", "reversal"], "T6 col2 td_w 5/95")
    run_t6(panel, ["btm", "momentum", "size", "reversal"], "T6 col2 td_w 1/99", wins=(0.01, 0.99))
    print("\nDONE.")


if __name__ == "__main__":
    main()
