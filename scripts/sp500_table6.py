# -*- coding: utf-8 -*-
"""Tabela 6 do Angelo (2025) — TD e retornos mensais pós-anúncio (replicação).

Spec do paper (Seção 6 + nota da Tabela 6), item a item:
  - Dependente: retorno MENSAL nos 3 meses seguintes ao anúncio ("the monthly
    return over 3 months following the earnings announcement");
  - Preditor: Tone Distance da call mais recente (CRUA, como no paper);
  - Col 1: só FE de firma + FE ano-mês;
  - Col 2: + BTM (do anúncio mais recente), Momentum (retorno acumulado de
    m-12 a m-2), Size (valor de mercado no anúncio; usamos ln — declarado),
    Reversal (retorno do mês m-1);
  - FE firma + ano-mês (absorvidos por demeaning iterativo; painel grande);
  - Erros-padrão clusterizados por ANO-MÊS (nota da tabela; Petersen 2008);
  - Winsorização 5/95 em todas as variáveis (política declarada na Seção 4.1);
    braço sem winsorizar reportado como sensibilidade.
Resultado do paper: coef 0.2282 (t 2.81) col 1; 0.1693 (t 2.17) col 2 —
TD PREVÊ retorno positivo (compensação de risco), coerente com nosso H2.

Uso: python scripts/sp500_table6.py
"""
from __future__ import annotations
import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "data" / "raw" / "sp500"
OUTDIR = ROOT / "data" / "interim" / "sp500"
SINCE = sys.argv[1] if len(sys.argv) > 1 else None   # ex.: 2013-01 (filtra ym da call)


def winsor(s, lo=0.05, hi=0.95):
    return s.clip(s.quantile(lo), s.quantile(hi))


WLO, WHI = 0.05, 0.95   # sobrescrito no braço 1/99 (nota 3 do paper)


def demean_two_way(df, cols, f1, f2, tol=1e-10, max_iter=50):
    """FE de duas vias por projeções alternadas (Guimarães-Portugal)."""
    X = df[cols].astype(float).copy()
    for _ in range(max_iter):
        X0 = X.copy()
        X = X - X.groupby(df[f1]).transform("mean")
        X = X - X.groupby(df[f2]).transform("mean")
        if float((X - X0).abs().max().max()) < tol:
            break
    return X


def run(d, controls, label, wins=True, lo=0.05, hi=0.95):
    cols = ["ret_m", "td"] + controls
    d = d.dropna(subset=cols + ["ticker", "ym"]).copy()
    if wins:
        for c in cols:
            d[c] = winsor(d[c], lo, hi)
    Xd = demean_two_way(d, cols, "ticker", "ym")
    y = Xd["ret_m"]
    X = Xd[["td"] + controls]
    m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["ym"]})
    print(f"{label:26s} n={len(d):>6d} firmas={d['ticker'].nunique():>4d} "
          f"meses={d['ym'].nunique():>4d} | TD coef={m.params['td']:+8.4f} "
          f"t={m.tvalues['td']:+6.2f} p={m.pvalues['td']:.3f}", flush=True)
    return m


def main():
    # ---------- retornos mensais a partir dos shards diários ----------
    px = pd.concat([pd.read_parquet(p) for p in (SP / "price_shards").glob("*.parquet")],
                   ignore_index=True).sort_values(["ticker", "date"])
    px = px[px["ticker"] != "^GSPC"].dropna(subset=["adj_close"])
    px["date"] = pd.to_datetime(px["date"])
    px["ym"] = px["date"].dt.to_period("M")
    px["r"] = px.groupby("ticker")["adj_close"].pct_change()
    mret = (px.groupby(["ticker", "ym"])["r"]
              .apply(lambda s: float(np.prod(1.0 + s.dropna()) - 1.0))
              .rename("ret_m").reset_index())
    mret = mret.sort_values(["ticker", "ym"]).reset_index(drop=True)
    # momentum m-12..m-2 (11 meses) e reversal m-1, por firma
    g = mret.groupby("ticker")["ret_m"]
    log1p = np.log1p(mret["ret_m"])
    mret["_l1p"] = log1p
    cum = mret.groupby("ticker")["_l1p"].transform(lambda s: s.rolling(11, min_periods=11).sum())
    mret["momentum"] = np.expm1(cum.groupby(mret["ticker"]).shift(2))
    mret["reversal"] = g.shift(1)
    print(f"painel mensal: {len(mret)} firma-meses, {mret['ticker'].nunique()} firmas", flush=True)

    # ---------- TD da call mais recente (1 a 3 meses atrás) ----------
    ev = pd.read_parquet(OUTDIR / "events_sp500_car3.parquet")
    ev = ev.dropna(subset=["td"]).copy()
    ev["ym_call"] = pd.to_datetime(ev["cdate"]).dt.to_period("M")
    ev = (ev.sort_values(["ticker", "cdate"])
            .drop_duplicates(["ticker", "ym_call"], keep="last"))
    frames = []
    for gap in (1, 2, 3):
        f = ev[["ticker", "ym_call", "td", "ln_mktcap"]].copy()
        # btm v2 entra depois (merge com events_sp500_paper se existir)
        f["ym"] = f["ym_call"] + gap
        f["gap"] = gap
        frames.append(f)
    link = pd.concat(frames, ignore_index=True)
    # call mais recente prevalece (gap menor)
    link = (link.sort_values(["ticker", "ym", "gap"])
                .drop_duplicates(["ticker", "ym"], keep="first"))
    panel = mret.merge(link, on=["ticker", "ym"], how="inner")
    if SINCE:
        panel = panel[panel["ym_call"] >= pd.Period(SINCE, "M")]
        print(f"FILTRO: calls a partir de {SINCE}", flush=True)
    panel["ym"] = panel["ym"].astype(str)
    print(f"painel TD x retorno mensal: {len(panel)} obs", flush=True)

    # ---------- Col 1: só FE ----------
    print("\n=== Tabela 6 col 1 (só FE firma + ano-mês) | paper: +0.2282 (t 2.81) ===")
    run(panel, [], "col1 winsor 5/95")
    run(panel, [], "col1 sem winsor", wins=False)

    # ---------- Col 2: + BTM, momentum, size, reversal ----------
    p2 = OUTDIR / "events_sp500_paper.parquet"
    src = p2 if p2.exists() else (OUTDIR / "events_sp500_full.parquet")
    evb = pd.read_parquet(src)[["ticker", "cdate", "btm"]].dropna()
    evb["ym_call"] = pd.to_datetime(evb["cdate"]).dt.to_period("M")
    evb = evb.drop_duplicates(["ticker", "ym_call"], keep="last")[["ticker", "ym_call", "btm"]]
    panel2 = panel.merge(evb, on=["ticker", "ym_call"], how="left")
    panel2["size"] = panel2["ln_mktcap"]
    print(f"\n=== Tabela 6 col 2 (+BTM, momentum, size(ln), reversal) | paper: +0.1693 (t 2.17) ===")
    print(f"(BTM de {src.name}; amostra 2009+ pela cobertura EDGAR)")
    run(panel2, ["btm", "momentum", "size", "reversal"], "col2 winsor 5/95")
    run(panel2, ["btm", "momentum", "size", "reversal"], "col2 winsor 1/99", lo=0.01, hi=0.99)
    run(panel2, ["btm", "momentum", "size", "reversal"], "col2 sem winsor", wins=False)
    run(panel, [], "col1 winsor 1/99", lo=0.01, hi=0.99)
    print("\nDONE.")


if __name__ == "__main__":
    main()
