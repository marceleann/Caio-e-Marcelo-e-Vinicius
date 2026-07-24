# -*- coding: utf-8 -*-
"""(A) AUDITORIA da variável de vol da H2 + (B) backtest POR EMPRESA (Marcelo).

A) Recomputa vol_20/vol_120 do zero (direto dos shards de preço, janela
   [t0+2, t0+21/121] como no script original) para uma amostra de eventos e
   compara com as colunas armazenadas que TODOS os testes da H2 usaram.
   corr < 0.99 => bug de construção/merge (e a H2 inteira estaria envenenada).

B) Backtest por empresa (um voto por firma, sem pooling):
   - classificação CAUSAL: cada evento é 'alto' se td_w > mediana dos eventos
     ANTERIORES da própria firma (mín. 4 anteriores);
   - por firma (mín. 4 eventos em cada balde): diff = média(resultado|alto)
     − média(resultado|baixo) p/ CAR[-1,+1], log vol_20, log vol_120, |SUE t+1|;
   - agregação entre firmas: % com o sinal esperado, média dos diffs,
     t = média/desvio × sqrt(N_firmas). Cada firma pesa igual.
Uso: python scripts/sp500_perfirm.py
"""
from __future__ import annotations
import glob, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "interim" / "sp500"
rng = np.random.default_rng(42)


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")
    ev = ev.merge(pd.read_parquet(OUTDIR / "td_variants.parquet")[["call_id", "td_w"]],
                  on="call_id", how="left")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    ev = ev[ev["cdate"] >= "2009-01-01"].copy()

    # ---------------- A) AUDITORIA vol ----------------
    px = pd.concat([pd.read_parquet(p) for p in
                    glob.glob(str(ROOT / "data/raw/sp500/price_shards/*.parquet"))],
                   ignore_index=True).sort_values(["ticker", "date"])
    px["date"] = pd.to_datetime(px["date"])
    tick = {}
    for t, g in px[px["ticker"] != "^GSPC"].groupby("ticker"):
        g = g.dropna(subset=["adj_close"])
        tick[t] = (g["date"].to_numpy("datetime64[ns]"),
                   pd.to_numeric(g["adj_close"], errors="coerce").pct_change().to_numpy(dtype=float))
    sample = ev.dropna(subset=["vol_20", "vol_120"]).sample(500, random_state=42)
    rows = []
    for r in sample.itertuples():
        d = tick.get(r.ticker)
        if d is None:
            continue
        dates, ret = d
        t0 = int(np.searchsorted(dates, np.datetime64(r.cdate)))
        if t0 + 122 >= len(ret):
            continue
        w20 = ret[t0 + 2:t0 + 22]
        w120 = ret[t0 + 2:t0 + 122]
        if np.isfinite(w20).sum() >= 15 and np.isfinite(w120).sum() >= 90:
            rows.append((r.call_id, float(np.nanstd(w20)), float(np.nanstd(w120)),
                         r.vol_20, r.vol_120))
    au = pd.DataFrame(rows, columns=["call_id", "re20", "re120", "st20", "st120"])
    print("=== A) AUDITORIA DA CONSTRUÇÃO DA VOL (n=%d eventos re-computados) ===" % len(au))
    for a, b, lbl in [("re20", "st20", "vol_20"), ("re120", "st120", "vol_120")]:
        c = au[a].corr(au[b])
        md = (au[a] - au[b]).abs().max()
        print(f"  {lbl}: corr(recomputada, armazenada) = {c:.4f} | max|diff| = {md:.6f}")

    # ---------------- B) BACKTEST POR EMPRESA ----------------
    for c_out, dst in [("vol_20", "lv20"), ("vol_120", "lv120")]:
        v = pd.to_numeric(ev[c_out], errors="coerce")
        ev[dst] = np.log(v.where(v > 0))
    ev = ev.sort_values(["ticker", "cdate"]).reset_index(drop=True)

    def trailing_side(g):
        sides = np.full(len(g), "", dtype=object)
        vals = g["td_w"].to_numpy(float)
        for i in range(len(g)):
            past = vals[:i]
            past = past[np.isfinite(past)]
            if len(past) >= 4 and np.isfinite(vals[i]):
                sides[i] = "alto" if vals[i] > np.median(past) else "baixo"
        return pd.Series(sides, index=g.index)

    ev["side"] = ev.groupby("ticker", group_keys=False).apply(trailing_side)

    print("\n=== B) BACKTEST POR EMPRESA (classificação causal: mediana passada da própria firma) ===")
    print(f"{'resultado':>14s} | {'esperado':>8s} | {'N firmas':>8s} | {'% sinal esperado':>16s} | "
          f"{'diff médio':>10s} | {'t':>6s}")
    for outc, exp, lbl in [("car_m1p1", -1, "CAR[-1,+1]"), ("lv20", +1, "log vol20"),
                           ("lv120", +1, "log vol120"), ("abs_sue_next", +1, "|SUE t+1|")]:
        diffs = []
        for t, g in ev.dropna(subset=[outc]).groupby("ticker"):
            hi = g.loc[g["side"] == "alto", outc]
            lo = g.loc[g["side"] == "baixo", outc]
            if len(hi) >= 4 and len(lo) >= 4:
                # winsor leve dentro da firma p/ um outlier não dominar o voto
                diffs.append(float(hi.clip(hi.quantile(.05), hi.quantile(.95)).mean()
                                   - lo.clip(lo.quantile(.05), lo.quantile(.95)).mean()))
        d = np.array(diffs)
        if len(d) < 30:
            print(f"{lbl:>14s} | poucas firmas ({len(d)})")
            continue
        tstat = d.mean() / d.std() * np.sqrt(len(d))
        okp = 100 * (np.sign(d) == exp).mean()
        print(f"{lbl:>14s} | {'menor' if exp < 0 else 'maior':>8s} | {len(d):>8d} | "
              f"{okp:>15.0f}% | {d.mean():>+10.4f} | {tstat:>+6.2f}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
