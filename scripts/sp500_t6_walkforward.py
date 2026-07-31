# -*- coding: utf-8 -*-
"""Walk-forward da Tabela 6 (retorno mensal 3m pós-call) — último pendente.

Painel mensal idêntico ao da suíte (FE firma+ano-mês por demeaning, cluster
ano-mês, controles BTM/momentum/size/reversal, winsor 5/95 dentro de cada
janela). Em cada corte (fim de ano, 2014..2024): beta estimado SÓ nos meses
até o corte vs beta SÓ nos meses após. Retorno do mês fecha dentro do mês —
corte por ym é leakage-safe. Sinais: td_w (perna da estratégia) e td (paper).
Direção esperada: positiva. Uso: python scripts/sp500_t6_walkforward.py
"""
from __future__ import annotations
import glob, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "interim" / "sp500"
CTRLS = ["btm", "momentum", "size", "reversal"]


def winsor(s):
    return s.clip(s.quantile(0.05), s.quantile(0.95))


def demean(df, cols, f1, f2, iters=50, tol=1e-10):
    X = df[cols].astype(float).copy()
    for _ in range(iters):
        X0 = X.copy()
        X = X - X.groupby(df[f1]).transform("mean")
        X = X - X.groupby(df[f2]).transform("mean")
        if float((X - X0).abs().max().max()) < tol:
            break
    return X


def fit(d, sig):
    cols = ["ret_m", sig] + CTRLS
    d = d.dropna(subset=cols + ["ticker", "ym"]).copy()
    if len(d) < 5000 or d["ym"].nunique() < 24:
        return None
    for c in cols:
        d[c] = winsor(d[c])
    Xd = demean(d, cols, "ticker", "ym")
    m = sm.OLS(Xd["ret_m"], Xd[[sig] + CTRLS]).fit(
        cov_type="cluster", cov_kwds={"groups": d["ym"]})
    return float(m.params[sig]), float(m.tvalues[sig]), len(d)


def main():
    px = pd.concat([pd.read_parquet(p) for p in
                    glob.glob(str(ROOT / "data/raw/sp500/price_shards/*.parquet"))],
                   ignore_index=True).sort_values(["ticker", "date"])
    px = px[px["ticker"] != "^GSPC"].dropna(subset=["adj_close"])
    px["date"] = pd.to_datetime(px["date"])
    px["adj_close"] = pd.to_numeric(px["adj_close"], errors="coerce")
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

    ev = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")
    ev = ev.merge(pd.read_parquet(OUTDIR / "td_variants.parquet")[["call_id", "td_w"]],
                  on="call_id", how="left")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    ev = ev.dropna(subset=["td"]).copy()
    ev["ym_call"] = ev["cdate"].dt.to_period("M")
    ev = (ev.sort_values(["ticker", "cdate"])
            .drop_duplicates(["ticker", "ym_call"], keep="last"))
    frames = []
    for gap in (1, 2, 3):
        f = ev[["ticker", "ym_call", "td", "td_w", "btm", "ln_mktcap"]].copy()
        f["ym"] = f["ym_call"] + gap
        f["gap"] = gap
        frames.append(f)
    link = (pd.concat(frames, ignore_index=True)
              .sort_values(["ticker", "ym", "gap"])
              .drop_duplicates(["ticker", "ym"], keep="first"))
    panel = mret.merge(link, on=["ticker", "ym"], how="inner")
    panel["size"] = panel["ln_mktcap"]
    print(f"painel: {len(panel)} firma-meses", flush=True)

    for sig in ["td_w", "td"]:
        print(f"\n===== T6 walk-forward | sinal {sig} | esperado: positivo =====")
        print(f"{'corte':>6s} | {'B_pass':>8s} {'t':>6s} {'n':>7s} | {'B_fut':>8s} {'t':>6s} {'n':>7s} | ok?")
        ok_f = tot = 0
        for y in range(2014, 2025):
            cut = pd.Period(f"{y}-12", "M")
            p = fit(panel[panel["ym"] <= cut], sig)
            f_ = fit(panel[panel["ym"] > cut], sig)
            if p is None or f_ is None:
                print(f"{y:>6d} | janela insuficiente")
                continue
            tot += 1
            ok_f += int(f_[0] > 0)
            print(f"{y:>6d} | {p[0]:>+8.4f} {p[1]:>+6.2f} {p[2]:>7d} | "
                  f"{f_[0]:>+8.4f} {f_[1]:>+6.2f} {f_[2]:>7d} | {'sim' if f_[0] > 0 else 'NAO'}")
        print(f"RESUMO {sig}: futuro positivo em {ok_f}/{tot} cortes")
    print("\nDONE.")


if __name__ == "__main__":
    main()
