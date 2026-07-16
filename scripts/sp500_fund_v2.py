# -*- coding: utf-8 -*-
"""Fundamentos v2 — definições EXATAS do Apêndice A do Angelo (2025).

Correções sobre sp500_fundamentals.py, cada uma ancorada no Apêndice A:
  1. Lev = "total long-term debt divided by total assets" -> dívida de LONGO
     PRAZO (tags LongTermDebtNoncurrent / LongTermDebt), não passivo total.
     (Tabela 1 do paper: média 0.235 — passivo total daria ~0.6.)
  2. Std ROA / Std CFO = desvio-padrão sobre os 16 TRIMESTRES anteriores
     (usávamos 8). Mínimo de 8 obs (paper não declara mínimo; declarado).
  3. Smooth = Std CFO / Std ROA (faltava).
  4. ETR = "total income taxes divided by net income plus total income taxes"
     -> tax/(NI+tax) SEMPRE (a fórmula do paper; sem preferir pretax do XBRL).
     Sem floor: valores extremos ficam para a winsorização (T1: média -2.39).
  5. Standardized ME = valor de mercado padronizado (z-score na amostra).
  6. RD/ativos (ausente=0), BTM, ROA, LnAssets: iguais à v1 (já batiam).
  7. FLUXOS TRIMESTRAIS via diferenciação de YTD: muitas firmas só reportam
     CFO (às vezes NI/tax) ACUMULADO no ano fiscal no 10-Q. Recuperamos o
     trimestre discreto diferenciando valores YTD consecutivos do mesmo início
     de ano fiscal (aceito só se o intervalo entre 'end's for ~1 tri);
     'filed' do trimestre derivado = max(filed) dos dois YTD (point-in-time
     conservador). Prática padrão de tratamento XBRL/Compustat.
Alinhamento point-in-time inalterado: filing protocolado ESTRITAMENTE antes
da call. Amostra 2009+ (decisão registrada: pré-2009 sem XBRL confiável).

Saída: data/raw/sp500/fundamentals_facts_v2.parquet (cache já trimestralizado)
       data/interim/sp500/events_sp500_paper.parquet (eventos + controles)
Uso: python scripts/sp500_fund_v2.py
"""
from __future__ import annotations
import json, zipfile, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTDIR = ROOT / "data" / "interim" / "sp500"
FUND_CACHE = RAW / "sp500" / "fundamentals_facts_v2.parquet"

STOCK_TAGS = {
    "assets": ["Assets"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "liab": ["Liabilities"],
    "ltd": ["LongTermDebtNoncurrent", "LongTermDebt"],   # dívida de longo prazo (paper)
}
FLOW_TAGS = {
    "ni": ["NetIncomeLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "rd": ["ResearchAndDevelopmentExpense"],
    "tax": ["IncomeTaxExpenseBenefit"],
}
FUND_COLS = ["btm", "lev", "roa", "std_roa", "std_cfo", "rd_at", "etr",
             "smooth", "ln_assets", "std_me"]


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
                            if not (70 <= dur <= 380):   # tri OU acumulado YTD
                                continue
                            rows.append((r.ticker, var, end, filed, float(val), start))
                        else:
                            rows.append((r.ticker, var, end, filed, float(val), None))
                    break  # primeiro tag disponível da lista
        if (i + 1) % 100 == 0:
            print(f"  extração: {i+1}/{len(cmap)} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows, columns=["ticker", "var", "end", "filed", "val", "start"])
    df["end"] = pd.to_datetime(df["end"]); df["filed"] = pd.to_datetime(df["filed"])
    df["start"] = pd.to_datetime(df["start"])
    # as-first-reported por período exato (start,end)
    df = df.sort_values("filed").drop_duplicates(["ticker", "var", "start", "end"], keep="first")

    stocks = df[df["start"].isna()].drop(columns=["start"])
    flows = df[df["start"].notna()].copy()
    flows["dur"] = (flows["end"] - flows["start"]).dt.days
    # trimestres discretos direto do filing
    q_direct = flows[flows["dur"] <= 100][["ticker", "var", "end", "filed", "val"]]
    # trimestres derivados: diferença entre YTDs consecutivos do MESMO início
    # de ano fiscal, aceita só se o intervalo entre 'end's tiver ~1 trimestre
    fl = flows.sort_values(["ticker", "var", "start", "end"])
    gk = ["ticker", "var", "start"]
    dv = fl["val"] - fl.groupby(gk)["val"].shift(1)
    dend = (fl["end"] - fl.groupby(gk)["end"].shift(1)).dt.days
    dfil = fl.groupby(gk)["filed"].shift(1)
    okd = dv.notna() & dend.between(70, 100)
    q_diff = pd.DataFrame({
        "ticker": fl.loc[okd, "ticker"], "var": fl.loc[okd, "var"],
        "end": fl.loc[okd, "end"],
        "filed": pd.concat([fl.loc[okd, "filed"], dfil[okd]], axis=1).max(axis=1),
        "val": dv[okd],
    })
    quarters = pd.concat([q_direct.assign(src=0), q_diff.assign(src=1)], ignore_index=True)
    quarters = (quarters.sort_values(["src", "filed"])
                        .drop_duplicates(["ticker", "var", "end"], keep="first")
                        .drop(columns=["src"]))
    out = pd.concat([stocks, quarters], ignore_index=True)
    out.to_parquet(FUND_CACHE, index=False)
    print(f"facts v2: {len(out)} linhas ({len(q_direct)} tri diretos + "
          f"{len(q_diff)} derivados de YTD antes do dedup) -> cache", flush=True)
    return out


def pit_value(series, cdate):
    ok = series[series["filed"] < cdate]
    if ok.empty:
        return np.nan
    return float(ok.loc[ok["end"].idxmax(), "val"])


def main():
    facts = extract_facts()
    by = {}
    for (t, v), g in facts.groupby(["ticker", "var"]):
        by[(t, v)] = g.sort_values("end").reset_index(drop=True)

    ev = pd.read_parquet(OUTDIR / "events_sp500_car3.parquet")
    ev = ev[ev["cdate"] >= "2009-01-01"].reset_index(drop=True)
    print(f"eventos 2009+: {len(ev)}", flush=True)

    out = {c: np.full(len(ev), np.nan) for c in FUND_COLS}
    for k, row in enumerate(ev.itertuples()):
        t, cd = row.ticker, row.cdate
        a = pit_value(by[(t, "assets")], cd) if (t, "assets") in by else np.nan
        if not np.isfinite(a) or a <= 0:
            continue
        out["ln_assets"][k] = np.log(a)
        e = pit_value(by[(t, "equity")], cd) if (t, "equity") in by else np.nan
        if np.isfinite(e) and np.isfinite(row.ln_mktcap):
            out["btm"][k] = e / np.exp(row.ln_mktcap)
        # Lev = dívida de LONGO PRAZO / ativos (Apêndice A); sem fallback p/
        # passivo total (seria outra variável). Firmas sem tag de LTD com
        # Liabilities presente: LTD ausente ~ missing (não zero) — conservador.
        ltd = pit_value(by[(t, "ltd")], cd) if (t, "ltd") in by else np.nan
        if np.isfinite(ltd) and ltd >= 0:
            out["lev"][k] = ltd / a
        # fluxos trimestrais (filed < call), últimos 17 (16 tri de janela + atual)
        def flow_series(var):
            g = by.get((t, var))
            if g is None:
                return None
            g = g[g["filed"] < cd].sort_values("end")
            return g.tail(17)
        ni = flow_series("ni")
        if ni is not None and len(ni):
            out["roa"][k] = float(ni["val"].iloc[-1]) / a
            if len(ni) >= 8:
                out["std_roa"][k] = float((ni["val"] / a).tail(16).std())
        cfo = flow_series("cfo")
        if cfo is not None and len(cfo) >= 8:
            out["std_cfo"][k] = float((cfo["val"] / a).tail(16).std())
        if np.isfinite(out["std_roa"][k]) and np.isfinite(out["std_cfo"][k]) \
                and out["std_roa"][k] > 1e-12:
            out["smooth"][k] = out["std_cfo"][k] / out["std_roa"][k]
        rd = flow_series("rd")
        out["rd_at"][k] = (float(rd["val"].tail(4).sum()) / a) if rd is not None and len(rd) else 0.0
        # ETR = tax/(NI+tax), TTM — fórmula LITERAL do Apêndice A
        tax = flow_series("tax")
        if tax is not None and len(tax) >= 4 and ni is not None and len(ni) >= 4:
            tx = float(tax["val"].tail(4).sum())
            den = float(ni["val"].tail(4).sum()) + tx
            if abs(den) > 1e-9:
                out["etr"][k] = tx / den
    for c in FUND_COLS:
        ev[c] = out[c]
    # Standardized ME (z-score do valor de mercado na amostra; Apêndice A)
    me = np.exp(ev["ln_mktcap"])
    ev["std_me"] = (me - me.mean()) / me.std()

    ev.to_parquet(OUTDIR / "events_sp500_paper.parquet", index=False)
    print("\n=== COBERTURA v2 (eventos 2009+) ===", flush=True)
    for c in FUND_COLS:
        print(f"  {c:10s}: {ev[c].notna().sum():>6d} ({100*ev[c].notna().mean():.0f}%)  "
              f"mean={ev[c].mean():+.4f} median={ev[c].median():+.4f}")
    print("\n(paper T1: lev 0.235/0.189 | std_roa 0.030/0.013 | std_cfo 0.030/0.020 | "
          "etr -2.39/0.212 | smooth 2.58/1.44)", flush=True)
    print("DONE.")


if __name__ == "__main__":
    main()
