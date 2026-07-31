# -*- coding: utf-8 -*-
"""Backtest EXECUTADO: carteira TOP-10 Tone Distance, mes a mes.

Desenho (pre-validado, sem lookahead):
  - No fim do mes m-1: universo = empresas com call em m-3..m-1 (call mais
    recente por empresa). TD conhecida no dia da call.
  - Compra as 10 maiores TD em pesos iguais; segura o mes m; rebalanceia.
  - Reguas: S&P 500 (^GSPC), universo elegivel equal-weight, bottom-10 (menor TD).
  - Custo: 10 bps por lado sobre o turnover; series e composicoes salvas em CSV.
Auditoria embutida: max(cdate) das calls usadas < inicio do mes de carteira.
Uso: python scripts/sp500_port10.py
"""
from __future__ import annotations
import glob
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "interim" / "sp500"
COST_SIDE = 0.0010  # 10 bps por lado


def monthly_returns():
    px = pd.concat([pd.read_parquet(p) for p in
                    glob.glob(str(ROOT / "data/raw/sp500/price_shards/*.parquet"))],
                   ignore_index=True)
    px["date"] = pd.to_datetime(px["date"])
    px["adj_close"] = pd.to_numeric(px["adj_close"], errors="coerce")
    px = px.dropna(subset=["adj_close"]).sort_values(["ticker", "date"])
    px["ym"] = px["date"].dt.to_period("M")
    px["r"] = px.groupby("ticker")["adj_close"].pct_change()
    return (px.groupby(["ticker", "ym"])["r"]
              .apply(lambda s: float(np.prod(1.0 + s.dropna()) - 1.0))
              .rename("ret_m").reset_index())


def stats(s: pd.Series, label: str):
    s = s.dropna()
    n = len(s)
    eq = (1.0 + s).cumprod()
    cum = float(eq.iloc[-1] - 1.0)
    cagr = float(eq.iloc[-1] ** (12.0 / n) - 1.0)
    vol = float(s.std() * np.sqrt(12))
    shp = float(s.mean() * 12 / vol) if vol > 0 else np.nan
    mdd = float((eq / eq.cummax() - 1.0).min())
    print(f"  {label:<14s} acum {cum:>+8.1%}  a.a. {cagr:>+6.2%}  "
          f"vol {vol:>5.1%}  Sharpe {shp:>+5.2f}  MDD {mdd:>6.1%}")
    return cagr


def t_excess(a: pd.Series, b: pd.Series, label: str):
    ex = (a - b).dropna()
    t = float(ex.mean() / ex.std() * np.sqrt(len(ex)))
    hit = float((ex > 0).mean())
    print(f"  excesso vs {label:<10s} {ex.mean()*12:>+6.2%} a.a.  "
          f"t={t:>+5.2f}  meses>0: {hit:.0%}")


def main():
    mret = monthly_returns()
    bench = mret[mret["ticker"] == "^GSPC"].set_index("ym")["ret_m"]
    mret = mret[mret["ticker"] != "^GSPC"]

    ev = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")
    ev = ev.merge(pd.read_parquet(OUTDIR / "td_variants.parquet")[["call_id", "td_w"]],
                  on="call_id", how="left")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    ev["ym_call"] = ev["cdate"].dt.to_period("M")

    for sig in ["td_w", "td"]:
        e = ev.dropna(subset=[sig]).copy()
        rows, holds, prev = [], [], set()
        for m in sorted(mret["ym"].unique()):
            win = e[(e["ym_call"] >= m - 3) & (e["ym_call"] <= m - 1)]
            if win.empty:
                continue
            last = win.sort_values("cdate").drop_duplicates("ticker", keep="last")
            rm = mret[mret["ym"] == m].set_index("ticker")["ret_m"]
            last = last[last["ticker"].isin(rm.index)]
            if len(last) < 100:          # universo ainda em rampa: pula
                continue
            assert last["cdate"].max() < m.to_timestamp(), "lookahead na selecao!"
            top = last.nlargest(10, sig)
            bot = last.nsmallest(10, sig)
            turn = len(set(top["ticker"]) - prev) / 10.0
            prev = set(top["ticker"])
            rows.append({"ym": m, "n_uni": len(last), "turnover": turn,
                         "top10": float(rm.loc[top["ticker"]].mean()),
                         "bot10": float(rm.loc[bot["ticker"]].mean()),
                         "uni_ew": float(rm.loc[last["ticker"]].mean()),
                         "spx": float(bench.get(m, np.nan))})
            holds.append({"ym": str(m), "tickers": ",".join(top["ticker"])})

        df = pd.DataFrame(rows).set_index("ym")
        df["top10_net"] = df["top10"] - 2 * COST_SIDE * df["turnover"]
        hd = pd.DataFrame(holds)
        df.to_csv(OUTDIR / f"port10_monthly_{sig}.csv")
        hd.to_csv(OUTDIR / f"port10_holdings_{sig}.csv", index=False)

        print(f"\n===== CARTEIRA TOP-10 por {sig} | {df.index.min()} a {df.index.max()}"
              f" | {len(df)} meses =====")
        print(f"  turnover medio: {df['turnover'].mean():.0%}/mes"
              f"  | universo medio: {df['n_uni'].mean():.0f} empresas")
        stats(df["top10"], "TOP-10 bruto")
        stats(df["top10_net"], "TOP-10 c/custo")
        stats(df["bot10"], "BOTTOM-10")
        stats(df["uni_ew"], "universo EW")
        stats(df["spx"], "S&P 500")
        t_excess(df["top10"], df["uni_ew"], "univ. EW")
        t_excess(df["top10"], df["spx"], "S&P 500")
        t_excess(df["top10"], df["bot10"], "bottom-10")

        print("  --- retorno por ano (carteira executada) ---")
        ann = df.groupby(df.index.year)[["top10", "bot10", "uni_ew", "spx"]].apply(
            lambda g: (1 + g).prod() - 1)
        print(ann.to_string(float_format=lambda x: f"{x:+.1%}"))

        print("  --- composicoes de amostra (conferiveis na mao) ---")
        for i in (0, len(hd) // 2, len(hd) - 1):
            print(f"  {hd.iloc[i]['ym']}: {hd.iloc[i]['tickers']}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
