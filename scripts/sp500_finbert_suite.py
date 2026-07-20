# -*- coding: utf-8 -*-
"""Suíte final do braço FinBERT — comparação LM x FinBERT lado a lado.

O coração do argumento GenAI do projeto: as MESMAS especificações do braço LM
(Eq.3 do Angelo nas 3 janelas de CAR, vol futura 20/120d, Tabela 6 de retorno
mensal), rodadas com os quatro sinais na MESMA amostra de eventos escorados:
  td      = TD LM igual-ponderada (replicação pura do paper)
  td_w    = TD LM ponderada por palavras (T8c1 — achado central do braço LM)
  td_fb   = TD FinBERT igual-ponderada (sentenças, hard argmax)
  td_fb_w = TD FinBERT ponderada por tokens (espelho da td_w)
Specs: winsor 5/95, sinais crus, FE firma+ano-tri (Eq.3/vol; cluster firma) e
FE firma+ano-mês (T6; cluster ano-mês). Amostra: eventos 2009+ com controles
antigos (a subamostra de fundamentos atenua por composição — documentado).

Saída: docs/RESULTADOS_FINBERT_AUTORUN.txt (gerado também pelo vigia noturno).
Uso: python scripts/sp500_finbert_suite.py
"""
from __future__ import annotations
import io, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "data" / "raw" / "sp500"
OUTDIR = ROOT / "data" / "interim" / "sp500"
DEST = ROOT / "docs" / "RESULTADOS_FINBERT_AUTORUN.txt"
OLD = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
       "length", "ln_mktcap", "industry_tone_ff49", "lagged_td", "sue_pct"]
SIGNALS = ["td", "td_w", "td_fb", "td_fb_w"]

buf = io.StringIO()
def out(s=""):
    print(s, flush=True)
    buf.write(s + "\n")


def winsor(s, lo=0.05, hi=0.95):
    return s.clip(s.quantile(lo), s.quantile(hi))


def run_eq3(d, yvar, sig, label):
    dd = d.dropna(subset=[yvar, sig] + OLD + ["ticker", "year_quarter"]).copy()
    if len(dd) < 400:
        out(f"{label:34s} n insuficiente ({len(dd)})")
        return
    for c in [yvar, sig] + OLD:
        dd[c] = winsor(dd[c])
    iqr = dd[sig].quantile(0.75) - dd[sig].quantile(0.25)
    m = smf.ols(f"{yvar} ~ {sig} + " + " + ".join(OLD) + " + C(ticker) + C(year_quarter)",
                data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["ticker"]})
    b, t, p = m.params[sig], m.tvalues[sig], m.pvalues[sig]
    out(f"{label:34s} n={len(dd):>6d} | coef={b:+9.4f} t={t:+6.2f} p={p:.3f} "
        f"| IQR->y={100*b*iqr:+.3f}%")


def demean_two_way(df, cols, f1, f2, tol=1e-10, max_iter=50):
    X = df[cols].astype(float).copy()
    for _ in range(max_iter):
        X0 = X.copy()
        X = X - X.groupby(df[f1]).transform("mean")
        X = X - X.groupby(df[f2]).transform("mean")
        if float((X - X0).abs().max().max()) < tol:
            break
    return X


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_car3.parquet")
    if "industry_tone_ff49" not in ev.columns:   # ff49.py grava nos 2 arquivos
        evp = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")[["call_id", "industry_tone_ff49"]]
        ev = ev.merge(evp, on="call_id", how="left")
    ev = ev.merge(pd.read_parquet(OUTDIR / "td_variants.parquet")[["call_id", "td_w"]],
                  on="call_id", how="left")
    fb = pd.read_parquet(OUTDIR / "tone_distance_finbert.parquet")
    ev = ev.merge(fb[["call_id", "td_fb", "td_fb_w", "is_random"]], on="call_id", how="left")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    scored = ev.dropna(subset=["td_fb_w"])
    e09 = scored[scored["cdate"] >= "2009-01-01"]

    out("=" * 90)
    out("SUÍTE FINAL LM x FinBERT — mesma amostra (eventos escorados), specs do Angelo")
    out(f"eventos escorados com CAR: {len(scored)} | 2009+: {len(e09)} | "
        f"firmas: {e09['ticker'].nunique()}")
    out(f"corr(td, td_fb)={scored['td'].corr(scored['td_fb']):+.3f} | "
        f"corr(td_w, td_fb_w)={scored['td_w'].corr(scored['td_fb_w']):+.3f}")
    out("=" * 90)

    for yv, wl in [("car_m1p1", "CAR[-1,+1]"), ("car_m1p2", "CAR[-1,+2]"),
                   ("car_m1p5", "CAR[-1,+5]"), ("vol_20", "vol 20d"), ("vol_120", "vol 120d")]:
        out(f"\n----- {wl} ~ sinal + controles antigos(FF49) + FE | 2009+ -----")
        for sig in SIGNALS:
            run_eq3(e09, yv, sig, f"{wl} ~ {sig}")

    # ---------------- Tabela 6 (retorno mensal, 3m) ----------------
    out("\n" + "=" * 90)
    out("TABELA 6 — retorno mensal 3m pós-anúncio (FE firma+ano-mês, cluster ano-mês)")
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

    evm = scored.copy()
    evm["ym_call"] = evm["cdate"].dt.to_period("M")
    evm = (evm.sort_values(["ticker", "cdate"])
              .drop_duplicates(["ticker", "ym_call"], keep="last"))
    btm = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")[["call_id", "btm"]]
    evm = evm.merge(btm, on="call_id", how="left")
    frames = []
    for gap in (1, 2, 3):
        f = evm[["ticker", "ym_call", "ln_mktcap", "btm"] + SIGNALS].copy()
        f["ym"] = f["ym_call"] + gap
        f["gap"] = gap
        frames.append(f)
    link = (pd.concat(frames, ignore_index=True)
              .sort_values(["ticker", "ym", "gap"])
              .drop_duplicates(["ticker", "ym"], keep="first"))
    panel = mret.merge(link, on=["ticker", "ym"], how="inner")
    panel["size"] = panel["ln_mktcap"]
    panel["ym"] = panel["ym"].astype(str)

    for sig in SIGNALS:
        for ctrls, lbl in [([], "col1 só FE"), (["btm", "momentum", "size", "reversal"], "col2")]:
            cols = ["ret_m", sig] + ctrls
            d = panel.dropna(subset=cols + ["ticker", "ym"]).copy()
            if len(d) < 2000:
                out(f"T6 {lbl} ~ {sig:8s} n insuficiente ({len(d)})")
                continue
            for c in cols:
                d[c] = winsor(d[c])
            Xd = demean_two_way(d, cols, "ticker", "ym")
            m = sm.OLS(Xd["ret_m"], Xd[[sig] + ctrls]).fit(
                cov_type="cluster", cov_kwds={"groups": d["ym"]})
            out(f"T6 {lbl:12s} ~ {sig:8s} n={len(d):>6d} | coef={m.params[sig]:+8.4f} "
                f"t={m.tvalues[sig]:+6.2f} p={m.pvalues[sig]:.3f}")

    out("\nDONE.")
    DEST.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nresultados gravados em {DEST}")


if __name__ == "__main__":
    main()
