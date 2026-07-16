# -*- coding: utf-8 -*-
"""Fundamentos contábeis via SEC EDGAR (XBRL companyfacts) -> controles da Eq.(3).

Constrói, da fonte primária oficial e SEM look-ahead (alinhamento pela data de
PROTOCOLO do filing, estritamente anterior à call), os controles contábeis que
o Angelo usa no H1 e que antes nos faltavam:
  BTM (patrimônio/valor de mercado), Lev (passivos/ativos), ROA, Std ROA e
  Std CFO (desvio rolante de 8 tri, mín. 4), RD/ativos (ausente=0, convenção
  Compustat), ETR (imposto/lucro pré-imposto, TTM; NaN se pré-imposto<=0),
  LnAssets. Institutional Ownership: inviável com dados abertos (declarado).

Amostra: eventos de 2009+ (decisão do Marcelo: pré-2009 excluído — sem XBRL).
Depois re-estima a Eq.(3) (CAR v3, TD crua) COM e SEM fundamentos, na MESMA
amostra 2009+, para o delta ser atribuível aos controles.
Uso: python scripts/sp500_fundamentals.py
"""
from __future__ import annotations
import json, zipfile, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTDIR = ROOT / "data" / "interim" / "sp500"
FUND_CACHE = RAW / "sp500" / "fundamentals_facts.parquet"

STOCK_TAGS = {
    "assets": ["Assets"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "liab": ["Liabilities"],
}
FLOW_TAGS = {
    "ni": ["NetIncomeLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "rd": ["ResearchAndDevelopmentExpense"],
    "tax": ["IncomeTaxExpenseBenefit"],
    "pretax": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
}
BASE_CONTROLS = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
                 "length", "ln_mktcap", "industry_tone", "lagged_td", "sue_pct"]
FUND_CONTROLS = ["btm", "lev", "roa", "std_roa", "std_cfo", "rd_at", "etr", "ln_assets"]


def extract_facts():
    if FUND_CACHE.exists():
        return pd.read_parquet(FUND_CACHE)
    cmap = pd.read_parquet(RAW / "cik_map.parquet")
    z = zipfile.ZipFile(RAW / "companyfacts.zip")
    names = set(z.namelist())
    rows = []
    t0 = time.time()
    for i, r in enumerate(cmap.itertuples()):
        fn = f"CIK{int(r.cik):010d}.json"
        if fn not in names:
            continue
        try:
            facts = json.loads(z.read(fn)).get("facts", {}).get("us-gaap", {})
        except Exception:
            continue
        for kind, tagmap in (("stock", STOCK_TAGS), ("flow", FLOW_TAGS)):
            for var, tags in tagmap.items():
                for tag in tags:
                    if tag not in facts:
                        continue
                    for u in facts[tag].get("units", {}).get("USD", []):
                        end = u.get("end"); filed = u.get("filed"); val = u.get("val")
                        if end is None or filed is None or val is None:
                            continue
                        start = u.get("start")
                        if kind == "flow":
                            if start is None:
                                continue
                            dur = (pd.Timestamp(end) - pd.Timestamp(start)).days
                            if not (70 <= dur <= 100):   # só trimestres
                                continue
                        rows.append((r.ticker, var, end, filed, float(val)))
                    break  # usa o primeiro tag disponível da lista
        if (i + 1) % 100 == 0:
            print(f"  extração: {i+1}/{len(cmap)} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows, columns=["ticker", "var", "end", "filed", "val"])
    df["end"] = pd.to_datetime(df["end"]); df["filed"] = pd.to_datetime(df["filed"])
    # as-first-reported: p/ cada (ticker,var,end), a 1ª versão protocolada
    df = df.sort_values("filed").drop_duplicates(["ticker", "var", "end"], keep="first")
    df.to_parquet(FUND_CACHE, index=False)
    print(f"facts: {len(df)} linhas, {df['ticker'].nunique()} tickers -> cache", flush=True)
    return df


def pit_value(series, cdate):
    """Último valor cujo filing é ESTRITAMENTE anterior à call (end mais recente)."""
    ok = series[series["filed"] < cdate]
    if ok.empty:
        return np.nan, None
    row = ok.loc[ok["end"].idxmax()]
    return float(row["val"]), row["end"]


def main():
    facts = extract_facts()
    by = {}
    for (t, v), g in facts.groupby(["ticker", "var"]):
        by[(t, v)] = g.sort_values("end").reset_index(drop=True)

    ev = pd.read_parquet(OUTDIR / "events_sp500_car3.parquet")
    ev = ev[ev["cdate"] >= "2009-01-01"].reset_index(drop=True)  # decisão: pré-2009 fora
    print(f"eventos 2009+: {len(ev)}", flush=True)

    out = {c: np.full(len(ev), np.nan) for c in FUND_CONTROLS}
    for k, row in enumerate(ev.itertuples()):
        t, cd = row.ticker, row.cdate
        a, a_end = pit_value(by.get((t, "assets"), pd.DataFrame(columns=["filed", "end", "val"])), cd) \
            if (t, "assets") in by else (np.nan, None)
        if not np.isfinite(a) or a <= 0:
            continue
        out["ln_assets"][k] = np.log(a)
        if (t, "equity") in by:
            e, _ = pit_value(by[(t, "equity")], cd)
            if np.isfinite(e) and np.isfinite(row.ln_mktcap):
                out["btm"][k] = e / np.exp(row.ln_mktcap)
        L = np.nan
        if (t, "liab") in by:
            L, _ = pit_value(by[(t, "liab")], cd)
        if not np.isfinite(L) and (t, "equity") in by:
            e2, _ = pit_value(by[(t, "equity")], cd)
            if np.isfinite(e2):
                L = a - e2          # identidade contábil: Passivos = Ativos - PL
        if np.isfinite(L):
            out["lev"][k] = L / a
        # séries trimestrais de fluxo (filed < call), últimas 8 obs
        def flow_series(var):
            g = by.get((t, var))
            if g is None:
                return None
            g = g[g["filed"] < cd].sort_values("end")
            return g.tail(9)
        ni = flow_series("ni")
        if ni is not None and len(ni):
            out["roa"][k] = float(ni["val"].iloc[-1]) / a
            if len(ni) >= 4:
                out["std_roa"][k] = float((ni["val"] / a).tail(8).std())
        cfo = flow_series("cfo")
        if cfo is not None and len(cfo) >= 4:
            out["std_cfo"][k] = float((cfo["val"] / a).tail(8).std())
        rd = flow_series("rd")
        out["rd_at"][k] = (float(rd["val"].tail(4).sum()) / a) if rd is not None and len(rd) else 0.0
        tax, pre = flow_series("tax"), flow_series("pretax")
        pt = tx = np.nan
        if tax is not None and len(tax) >= 4:
            tx = float(tax["val"].tail(4).sum())
            if pre is not None and len(pre) >= 4:
                pt = float(pre["val"].tail(4).sum())
            elif ni is not None and len(ni) >= 4:
                pt = float(ni["val"].tail(4).sum()) + tx   # identidade: pré-imposto = LL + imposto
        if np.isfinite(pt) and pt > 0 and np.isfinite(tx):
            out["etr"][k] = tx / pt
    for c in FUND_CONTROLS:
        ev[c] = out[c]
    ev.to_parquet(OUTDIR / "events_sp500_full.parquet", index=False)

    print("\n=== COBERTURA dos fundamentos (eventos 2009+) ===", flush=True)
    for c in FUND_CONTROLS:
        print(f"  {c:10s}: {ev[c].notna().sum():>6d} ({100*ev[c].notna().mean():.0f}%)")

    # ---------- Eq.(3): mesma amostra, sem e com fundamentos ----------
    def winsor(s): return s.clip(s.quantile(0.05), s.quantile(0.95))
    def zscore(s): return (s - s.mean()) / s.std()

    def run(d0, yvar, controls, label):
        d = d0.dropna(subset=[yvar, "td"] + controls + ["ticker", "year_quarter"]).copy()
        d["_td"] = zscore(winsor(d["td"]))
        d["_y"] = winsor(d[yvar])
        for c in controls:
            d[c] = winsor(d[c])
        m = smf.ols("_y ~ _td + " + " + ".join(controls) + " + C(ticker) + C(year_quarter)",
                    data=d).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
        return dict(label=label, n=len(d), nf=d["ticker"].nunique(),
                    coef=m.params["_td"], t=m.tvalues["_td"], p=m.pvalues["_td"])

    # amostra comum: eventos com TODOS os controles (base + fundos) p/ delta limpo
    common = ev.dropna(subset=["td", "car_m1p1"] + BASE_CONTROLS + FUND_CONTROLS)
    print(f"\namostra comum (todos os controles presentes): {len(common)} eventos, "
          f"{common['ticker'].nunique()} firmas")
    print(f"\n{'janela':12s} {'spec':16s} {'n':>6s} {'firmas':>5s} {'coef(SD)':>10s} {'t':>7s} {'p':>7s}")
    for yv, wl in [("car_m1p1", "CAR[-1,+1]"), ("car_m1p2", "CAR[-1,+2]"), ("car_m1p5", "CAR[-1,+5]")]:
        r1 = run(common, yv, BASE_CONTROLS, "sem fundamentos")
        r2 = run(common, yv, BASE_CONTROLS + FUND_CONTROLS, "COM fundamentos")
        for r in (r1, r2):
            print(f"{wl:12s} {r['label']:16s} {r['n']:>6d} {r['nf']:>5d} {r['coef']:>10.4f} "
                  f"{r['t']:>7.2f} {r['p']:>7.3f}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
