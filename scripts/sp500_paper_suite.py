# -*- coding: utf-8 -*-
"""Suíte de replicação EXATA das Tabelas 3, 4 e 5 do Angelo (2025).

Cada escolha ancorada no texto/Apêndice A do paper:
  - Eq.(3): CAR(CAPM 100d/gap50/min70) ~ TD crua + controles + FE firma +
    FE ano-tri, cluster por firma, winsor 5/95 em TODAS as variáveis.
  - Controles da Tabela 3: Disclosure Tone, Lagged AVERAGE TD (média dos 4
    tri anteriores), SUE, BTM, Lev(=LTD/ativos), ROA, Std ROA(16q),
    Std CFO(16q), RD, ETR(=tax/(NI+tax)), Smooth(=StdCFO/StdROA), Length,
    Standardized ME, LnAssets, Analyst Tone Distance(=DESVIO-PADRÃO do tom
    entre analistas, Apêndice A), Analyst Tone, Industry Tone.
    AUSENTES E DECLARADOS: AQ (Francis et al. 2004; exige accruals),
    Institutional Ownership (13F). SUE é proxy (yfinance vs I/B/E/S);
    Industry Tone por setor GICS (paper: FF49).
  - Tabela 4 (H2): vol realizada 20/120 dias pós-anúncio ~ mesma spec.
    (mean range e vol implícita: sem OHLC/OptionMetrics — declarado.)
  - Tabela 5: |SUE t+1|, vol futura de ROA e CFO (4 tri seguintes) ~ mesma
    spec. (Tobin's Q exige PP&E bruto + intangíveis — declarado.)
Amostras: 2009+ (decisão registrada) e 2009-2022 (fim do período do paper).
Coeficiente reportado na escala CRUA do paper (e também por DP, para nós).

Uso: python scripts/sp500_paper_suite.py
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "interim" / "sp500"

PAPER_CONTROLS = ["disclosure_tone", "lagged_avg_td", "sue_pct", "btm", "lev",
                  "roa", "std_roa", "std_cfo", "rd_at", "etr", "smooth",
                  "length", "std_me", "ln_assets",
                  "analyst_tone_disp", "analyst_tone", "industry_tone"]


def winsor(s, lo=0.05, hi=0.95):
    return s.clip(s.quantile(lo), s.quantile(hi))


def run(df, yvar, tdvar, controls, label):
    d = df.dropna(subset=[yvar, tdvar] + controls + ["ticker", "year_quarter"]).copy()
    if d.empty or d["ticker"].nunique() < 30:
        print(f"{label:34s} amostra insuficiente (n={len(d)})", flush=True)
        return
    for c in [yvar, tdvar] + controls:
        d[c] = winsor(d[c])
    sd_td = d[tdvar].std()
    m = smf.ols(f"{yvar} ~ {tdvar} + " + " + ".join(controls)
                + " + C(ticker) + C(year_quarter)",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    b, t, p = m.params[tdvar], m.tvalues[tdvar], m.pvalues[tdvar]
    print(f"{label:34s} n={len(d):>6d} firmas={d['ticker'].nunique():>4d} | "
          f"coef(cru)={b:+9.4f} t={t:+6.2f} p={p:.3f} | por DP={b*sd_td:+8.5f}", flush=True)


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_paper.parquet")
    ev["cdate"] = pd.to_datetime(ev["cdate"])
    # idempotência: colunas que este script cria são descartadas se já existem
    # (o parquet versionado já veio com elas; sem isso o merge duplica nomes)
    made = ["lagged_avg_td", "lagged_avg_td4", "analyst_tone_disp"]
    ev = ev.drop(columns=[c for c in made if c in ev.columns])

    # ---- Lagged AVERAGE TD: média dos 4 TDs anteriores da firma (Apêndice A) ----
    td_all = pd.read_parquet(OUTDIR / "tone_distance_sp500.parquet")
    td_all["call_datetime"] = pd.to_datetime(td_all["call_datetime"], utc=True)
    td_all = td_all.sort_values(["ticker", "call_datetime"])
    g = td_all.groupby("ticker")["td"]
    td_all["lagged_avg_td"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    td_all["lagged_avg_td4"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=4).mean())
    ev = ev.merge(td_all[["call_id", "lagged_avg_td", "lagged_avg_td4"]], on="call_id", how="left")

    # ---- Analyst Tone Dispersion = std do net tone entre analistas (Apêndice A) ----
    sc_path = OUTDIR / "speaker_counts.parquet"
    if sc_path.exists():
        sc = pd.read_parquet(sc_path)
        an = sc[(sc["role_eff"] == "analyst") & (sc["mk"] != "")]
        per = an.groupby(["call_id", "mk"]).agg(nw=("nw", "sum"), npos=("npos", "sum"),
                                                nneg=("nneg", "sum"))
        per = per[per["nw"] > 0]
        per["net"] = (per["npos"] - per["nneg"]) / per["nw"]
        disp = per.groupby("call_id")["net"].agg(["std", "count"])
        disp = disp[disp["count"] >= 2]["std"].rename("analyst_tone_disp").reset_index()
        ev = ev.merge(disp, on="call_id", how="left")
        print(f"analyst_tone_disp (std entre analistas): {ev['analyst_tone_disp'].notna().sum()} calls")
    else:
        ev["analyst_tone_disp"] = ev["analyst_tone_distance"]
        print("AVISO: speaker_counts.parquet ausente -> usando distância euclidiana como proxy")

    # ---- Tabela 5: vol futura de ROA/CFO (4 tri seguintes) do EDGAR v2 ----
    facts = pd.read_parquet(ROOT / "data" / "raw" / "sp500" / "fundamentals_facts_v2.parquet")
    fl = {}
    for (t, v), gg in facts.groupby(["ticker", "var"]):
        if v in ("ni", "cfo", "assets"):
            fl[(t, v)] = gg.sort_values("end").reset_index(drop=True)
    fut_roa = np.full(len(ev), np.nan)
    fut_cfo = np.full(len(ev), np.nan)
    for k, row in enumerate(ev.itertuples()):
        t, cd = row.ticker, row.cdate
        a = np.exp(row.ln_assets) if np.isfinite(row.ln_assets) else np.nan
        if not np.isfinite(a) or a <= 0:
            continue
        for var, arr in (("ni", fut_roa), ("cfo", fut_cfo)):
            gg = fl.get((t, var))
            if gg is None:
                continue
            nxt = gg[gg["end"] > cd].head(4)
            if len(nxt) == 4:
                arr[k] = float((nxt["val"] / a).std())
    ev["fut_roa_vol"], ev["fut_cfo_vol"] = fut_roa, fut_cfo

    # ---- |SUE t+1| (gap 30-200 dias, como no h3) ----
    ev = ev.sort_values(["ticker", "cdate"])
    ev["sue_next"] = ev.groupby("ticker")["sue_pct"].shift(-1)
    ev["next_date"] = ev.groupby("ticker")["cdate"].shift(-1)
    gap = (ev["next_date"] - ev["cdate"]).dt.days
    ok = (gap >= 30) & (gap <= 200)
    ev.loc[~ok, "sue_next"] = np.nan
    ev["abs_sue_next"] = ev["sue_next"].abs()

    missing = [c for c in PAPER_CONTROLS if c not in ev.columns]
    if missing:
        raise SystemExit(f"controles faltando: {missing}")
    print("\nCONTROLES AUSENTES DECLARADOS: AQ (accruals), Institutional Ownership (13F)")
    print("PROXIES DECLARADOS: SUE (yfinance), Industry Tone (GICS), mercado ^GSPC, precos yfinance\n")

    for sample, lbl in [((ev["cdate"] >= "2009-01-01"), "2009+"),
                        ((ev["cdate"] >= "2009-01-01") & (ev["cdate"] <= "2022-12-31"), "2009-22")]:
        d = ev[sample]
        print(f"===== TABELA 3 (Eq.3) — amostra {lbl} | paper: -0.49/-0.45/-0.54 (t -3.4/-2.9/-3.1) =====")
        for yv, wl in [("car_m1p1", "CAR[-1,+1]"), ("car_m1p2", "CAR[-1,+2]"), ("car_m1p5", "CAR[-1,+5]")]:
            run(d, yv, "td", PAPER_CONTROLS, f"{wl} {lbl}")
        print(f"----- sensibilidade: lag médio ESTRITO de 4 tri ({lbl}) -----")
        ctrl4 = ["lagged_avg_td4" if c == "lagged_avg_td" else c for c in PAPER_CONTROLS]
        run(d, "car_m1p1", "td", ctrl4, f"CAR[-1,+1] {lbl} lag4 estrito")

        print(f"\n===== TABELA 4 (risco) — amostra {lbl} | paper: +0.068 (t 3.5) 20d, +0.050 (t 3.5) 120d =====")
        for yv, wl in [("vol_20", "StdRet 20d"), ("vol_120", "StdRet 120d")]:
            if yv in d.columns:
                run(d, yv, "td", PAPER_CONTROLS, f"{wl} {lbl}")

        print(f"\n===== TABELA 5 (operacional) — amostra {lbl} | paper: |SUE|+0.024 (t 2.4), "
              f"ROAvol +0.109 (t 2.2), CFOvol +0.078 (t 1.8) =====")
        for yv, wl in [("abs_sue_next", "|SUE t+1|"), ("fut_roa_vol", "ROA vol fut"),
                       ("fut_cfo_vol", "CFO vol fut")]:
            run(d, yv, "td", PAPER_CONTROLS, f"{wl} {lbl}")
        print()
    ev.to_parquet(OUTDIR / "events_sp500_paper.parquet", index=False)
    print("events_sp500_paper.parquet atualizado (lagged_avg_td, disp, vols futuras).")
    print("DONE.")


if __name__ == "__main__":
    main()
