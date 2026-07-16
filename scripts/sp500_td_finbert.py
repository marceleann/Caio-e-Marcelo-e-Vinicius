# -*- coding: utf-8 -*-
"""TD_FinBERT no S&P 500 — braço GenAI (único desvio intencional do Angelo).

Monta a Tone Distance com tom FinBERT em nível de SENTENÇA a partir dos shards
de data/interim/sentence_scores_shards (tech antigos + S&P novos, pool única):
  - rótulo duro por argmax (convenção do Stage 3): sentença positiva/negativa;
  - por gestor: fp = fração de sentenças positivas, fn = fração negativas
    (espelho exato da fração de palavras LM do Angelo);
  - td_fb = Eq.(1) igual-ponderada (como o paper base);
  - td_fb_w = ponderada pelos TOKENS falados pelo gestor (espelho da Tabela 8
    col 1, que recuperou o H1 no braço LM);
  - SEM filtro de sentenças mínimas por gestor (fidelidade ao paper; braço
    min>=5 como sensibilidade).
Shards antigos (tech) têm coluna `speaker`; novos têm `mk` — normalizados.
Prefixo NOVO é embaralhado (seed 42) => subset aleatório representativo; os
5.452 tech antigos NÃO são aleatórios (só tech) — braço "só aleatórios"
reportado separadamente para evitar viés setorial em leituras interinas.

Depois roda Eq.(3) (CAR v3) e vol futura com controles antigos no subset
pontuado. Saída: data/interim/sp500/tone_distance_finbert.parquet
Uso: python scripts/sp500_td_finbert.py
"""
from __future__ import annotations
import re, unicodedata, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
INT = ROOT / "data" / "interim"
SHARDS = INT / "sentence_scores_shards"
OUTDIR = INT / "sp500"
MIN_MGR = 2

OLD = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
       "length", "ln_mktcap", "industry_tone", "lagged_td", "sue_pct"]

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


def td_from(fp, fn, w=None):
    d = np.sqrt((fp - fp.mean()) ** 2 + (fn - fn.mean()) ** 2)
    if w is None:
        return float(d.mean())
    W = w.sum()
    return float((d * w).sum() / W) if W > 0 else np.nan


def winsor(s):
    return s.clip(s.quantile(0.05), s.quantile(0.95))


def run(df, yvar, tdvar, label):
    d = df.dropna(subset=[yvar, tdvar] + OLD + ["ticker", "year_quarter"]).copy()
    if len(d) < 400 or d["ticker"].nunique() < 30:
        print(f"{label:46s} amostra ainda insuficiente (n={len(d)})", flush=True)
        return
    for c in [yvar, tdvar] + OLD:
        d[c] = winsor(d[c])
    m = smf.ols(f"{yvar} ~ {tdvar} + " + " + ".join(OLD) + " + C(ticker) + C(year_quarter)",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    print(f"{label:46s} n={len(d):>6d} firmas={d['ticker'].nunique():>4d} | "
          f"coef={m.params[tdvar]:+9.4f} t={m.tvalues[tdvar]:+6.2f} p={m.pvalues[tdvar]:.3f}",
          flush=True)


def main():
    files = sorted(SHARDS.glob("call_*.parquet"))
    print(f"shards: {len(files)} calls pontuadas", flush=True)
    rows = []
    for p in files:
        d = pd.read_parquet(p)
        is_new = "mk" in d.columns
        if not is_new:
            d["mk"] = d["speaker"].map(mkey)
        mg = d[d["role"].isin(["manager", "management"]) & (d["mk"] != "")].copy()
        if mg.empty:
            continue
        lab = mg[["p_neg", "p_neu", "p_pos"]].to_numpy().argmax(1)
        mg["is_pos"] = (lab == 2).astype(float)
        mg["is_neg"] = (lab == 0).astype(float)
        agg = mg.groupby("mk").agg(n=("is_pos", "size"), tok=("n_tokens", "sum"),
                                   fp=("is_pos", "mean"), fn=("is_neg", "mean"))
        if len(agg) < MIN_MGR:
            continue
        fp, fn = agg["fp"].to_numpy(), agg["fn"].to_numpy()
        w = agg["tok"].to_numpy(float)
        cid = d["call_id"].iloc[0]
        r = dict(call_id=cid, is_random=bool(is_new),
                 td_fb=td_from(fp, fn), td_fb_w=td_from(fp, fn, w),
                 n_managers=len(agg), min_sent=int(agg["n"].min()))
        a5 = agg[agg["n"] >= 5]
        r["td_fb_min5"] = td_from(a5["fp"].to_numpy(), a5["fn"].to_numpy()) \
            if len(a5) >= MIN_MGR else np.nan
        rows.append(r)
    fb = pd.DataFrame(rows)
    fb.to_parquet(OUTDIR / "tone_distance_finbert.parquet", index=False)
    print(f"TD_FinBERT: {len(fb)} calls ({fb['is_random'].sum()} do sorteio aleatório novo)",
          flush=True)
    print(fb[["td_fb", "td_fb_w"]].describe().loc[["mean", "50%", "std"]].to_string(), flush=True)

    ev = pd.read_parquet(OUTDIR / "events_sp500_car3.parquet").merge(fb, on="call_id", how="inner")
    print(f"\neventos com CAR + TD_FinBERT: {len(ev)}", flush=True)
    print(f"corr(td_lm, td_fb) = {ev['td'].corr(ev['td_fb']):+.3f} | "
          f"corr(td_lm_w?, td_fb_w) via variants:", flush=True)
    tv = OUTDIR / "td_variants.parquet"
    if tv.exists():
        ev = ev.merge(pd.read_parquet(tv)[["call_id", "td_w"]], on="call_id", how="left")
        print(f"corr(td_w, td_fb_w) = {ev['td_w'].corr(ev['td_fb_w']):+.3f}", flush=True)

    for sub, lbl in [(ev, "todos pontuados"), (ev[ev["is_random"]], "só sorteio aleatório")]:
        print(f"\n===== Eq.(3) e risco — {lbl} =====")
        for tdv in ["td_fb", "td_fb_w"]:
            run(sub, "car_m1p1", tdv, f"CAR[-1,+1] ~ {tdv} ({lbl})")
            run(sub, "car_m1p5", tdv, f"CAR[-1,+5] ~ {tdv} ({lbl})")
            run(sub, "vol_120", tdv, f"vol_120 ~ {tdv} ({lbl})")
    print("\nDONE.")


if __name__ == "__main__":
    main()
