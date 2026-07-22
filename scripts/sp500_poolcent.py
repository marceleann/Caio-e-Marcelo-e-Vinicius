# -*- coding: utf-8 -*-
"""TD com centroide AGREGADO (leitura B da Eq.1) — LM e FinBERT.

Motivação (desconfiança do Marcelo, 20/07, confirmada): a frase do paper
"Avg.Pos_t is the average percent positive words spoken across all managers"
admite duas leituras: (A) média simples das frações por gestor (nossa
original) ou (B) fração AGREGADA (total pos/total palavras dos gestores) —
e o rótulo da Figura 1 do paper ("Average Transcript Tone") sugere B.
Teste inicial: leitura B revive a TD base no nosso universo (t=-1.98 em
CAR[-1,+1] vs -0.72 da leitura A; corr A-B=0.96).

Este script constrói td_poolcent (LM, das contagens por fala) e
td_fb_poolcent (FinBERT, dos shards de sentença; shards tech antigos usam
mkey(speaker)), roda a Eq.(3) nas 3 janelas para ambos e salva
data/interim/sp500/td_poolcent.parquet.
Uso: python scripts/sp500_poolcent.py
"""
from __future__ import annotations
import re, unicodedata, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "interim" / "sp500"
SHARDS = ROOT / "data" / "interim" / "sentence_scores_shards"
OLD = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
       "length", "ln_mktcap", "industry_tone_ff49", "lagged_td", "sue_pct"]

_QAp = re.compile(r"^[QA]\s*[-–:]\s*"); _TS = re.compile(r"\s+[-–—]\s+.*$"); _CM = re.compile(r"(?<=[a-z])(?=[A-Z])")
def _clean(n):
    if not isinstance(n, str): return ""
    n = _QAp.sub("", n.strip()); n = _TS.sub("", n); n = _CM.sub(" ", n)
    n = unicodedata.normalize("NFKD", n); n = "".join(c for c in n if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", n).strip().casefold()
def mkey(n):
    c = _clean(n)
    if not c: return ""
    t = [x for x in c.replace(".", " ").split() if len(x) > 1]
    return f"{t[-1]}|{t[0][0]}" if t else c


def winsor(s):
    return s.clip(s.quantile(0.05), s.quantile(0.95))


def main():
    # ---------- LM: leitura B ----------
    sc = pd.read_parquet(OUTDIR / "speaker_counts.parquet")
    mg = sc[(sc["role_eff"] == "manager") & (sc["mk"] != "")]
    per = mg.groupby(["call_id", "mk"]).agg(nw=("nw", "sum"), npos=("npos", "sum"),
                                            nneg=("nneg", "sum")).reset_index()
    per = per[per["nw"] > 0]
    per["fp"] = per["npos"] / per["nw"]; per["fn"] = per["nneg"] / per["nw"]
    tot = per.groupby("call_id").agg(W=("nw", "sum"), P=("npos", "sum"), N=("nneg", "sum"))
    per = per.merge(tot, on="call_id")
    d = np.sqrt((per["fp"] - per["P"] / per["W"]) ** 2 + (per["fn"] - per["N"] / per["W"]) ** 2)
    tdb = d.groupby(per["call_id"]).mean().rename("td_poolcent").reset_index()
    nmg = per.groupby("call_id").size()
    tdb = tdb[tdb["call_id"].isin(nmg[nmg >= 2].index)]
    print(f"LM leitura B: {len(tdb)} calls", flush=True)

    # ---------- FinBERT: leitura B (sentenças, hard argmax) ----------
    rows = []
    for p in SHARDS.glob("call_*.parquet"):
        d = pd.read_parquet(p)
        if "mk" not in d.columns:
            d["mk"] = d["speaker"].map(mkey)
        m = d[d["role"].isin(["manager", "management"]) & (d["mk"] != "")]
        if m.empty:
            continue
        lab = m[["p_neg", "p_neu", "p_pos"]].to_numpy().argmax(1)
        a = (m.assign(isp=(lab == 2).astype(float), isn=(lab == 0).astype(float))
              .groupby("mk").agg(n=("isp", "size"), sp=("isp", "sum"), sn=("isn", "sum")))
        if len(a) < 2:
            continue
        cp, cn = a["sp"].sum() / a["n"].sum(), a["sn"].sum() / a["n"].sum()
        dist = np.sqrt((a["sp"] / a["n"] - cp) ** 2 + (a["sn"] / a["n"] - cn) ** 2)
        rows.append((d["call_id"].iloc[0], float(dist.mean())))
    fbb = pd.DataFrame(rows, columns=["call_id", "td_fb_poolcent"])
    print(f"FinBERT leitura B: {len(fbb)} calls", flush=True)

    out = tdb.merge(fbb, on="call_id", how="outer")
    out.to_parquet(OUTDIR / "td_poolcent.parquet", index=False)

    # ---------- Eq.(3) nas 3 janelas, ambos os sensores ----------
    ev = pd.read_parquet(OUTDIR / "events_sp500_car3.parquet")
    if "industry_tone_ff49" not in ev.columns:
        ev = ev.merge(pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")
                      [["call_id", "industry_tone_ff49"]], on="call_id", how="left")
    ev = ev.merge(out, on="call_id", how="left")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    ev = ev[ev["cdate"] >= "2009-01-01"]
    print(f"corr(td, td_poolcent) = {ev['td'].corr(ev['td_poolcent']):+.3f}")
    for tdv in ["td_poolcent", "td_fb_poolcent"]:
        for yv in ["car_m1p1", "car_m1p2", "car_m1p5"]:
            dd = ev.dropna(subset=[yv, tdv] + OLD + ["ticker", "year_quarter"]).copy()
            if len(dd) < 400:
                print(f"{yv} ~ {tdv}: n insuficiente ({len(dd)})")
                continue
            for c in [yv, tdv] + OLD:
                dd[c] = winsor(dd[c])
            iqr = dd[tdv].quantile(0.75) - dd[tdv].quantile(0.25)
            m = smf.ols(f"{yv} ~ {tdv} + " + " + ".join(OLD) + " + C(ticker) + C(year_quarter)",
                        data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["ticker"]})
            print(f"{yv:10s} ~ {tdv:16s} n={len(dd):>6d} | coef={m.params[tdv]:+8.4f} "
                  f"t={m.tvalues[tdv]:+6.2f} p={m.pvalues[tdv]:.3f} "
                  f"| IQR->y={100*m.params[tdv]*iqr:+.3f}%", flush=True)
    print("DONE.")


if __name__ == "__main__":
    main()
