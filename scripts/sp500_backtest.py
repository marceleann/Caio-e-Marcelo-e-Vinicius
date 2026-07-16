# -*- coding: utf-8 -*-
"""Backtest S&P 500 — carteira calendar-time por quintis de Tone Distance.

Spec CONGELADA antes de rodar (anti data-mining):
  - Sinal primário: TD CRUA (como o Angelo); braço secundário z (logado, DSR).
  - Direção fixada a priori pelo paper: LONG TD alta / SHORT TD baixa.
  - Ranking point-in-time: percentil do sinal contra os eventos dos 90 dias
    corridos ESTRITAMENTE anteriores (mínimo 20 eventos na janela).
  - Quintis: long se pct >= 0,8; short se pct <= 0,2.
  - Execução: entrada no CLOSE do 1º pregão após a call (T+1), holding 63
    pregões, tranches sobrepostas (Jegadeesh-Titman), pesos iguais diários.
  - Custos: 5 bps por perna (entrada e saída).
Expectativa registrada ANTES de rodar: se o painel estiver certo (prêmio nulo),
spread ~0,1%/mês bruto e Sharpe <= ~0,2. Uso: python scripts/sp500_backtest.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "data" / "raw" / "sp500"
OUTDIR = ROOT / "data" / "interim" / "sp500"
HOLD, RANK_WIN_D, MIN_HIST = 63, 90, 20
TOP, COST = 0.20, 0.0005


def build_portfolio(events, tick, cal, signal_col):
    ev = events.dropna(subset=[signal_col]).sort_values("cdate").reset_index(drop=True)
    times = ev["cdate"].to_numpy("datetime64[ns]")
    sig = ev[signal_col].to_numpy(float)
    win = np.timedelta64(RANK_WIN_D, "D")
    lo = np.searchsorted(times, times - win, side="left")
    hi = np.searchsorted(times, times, side="left")
    side = np.zeros(len(ev), dtype=int)
    for i in range(len(ev)):
        past = sig[lo[i]:hi[i]]
        past = past[np.isfinite(past)]
        if len(past) < MIN_HIST:
            continue
        pct = (past < sig[i]).mean() + 0.5 * (past == sig[i]).mean()  # midrank
        if pct >= 1 - TOP:
            side[i] = 1
        elif pct <= TOP:
            side[i] = -1
    ev["side"] = side

    n_days = len(cal)
    cal_idx = {d: k for k, d in enumerate(cal)}
    long_sum = np.zeros(n_days); long_cnt = np.zeros(n_days)
    short_sum = np.zeros(n_days); short_cnt = np.zeros(n_days)
    used = {1: 0, -1: 0}
    for row in ev[ev["side"] != 0].itertuples():
        d = tick.get(row.ticker)
        if d is None:
            continue
        dates, r = d
        p = int(np.searchsorted(dates, np.datetime64(row.cdate)))
        e = p + 1                      # entrada: close de T+1
        x = min(e + HOLD, len(r) - 1)  # saída: close de e+63
        if e >= len(r) - 2:
            continue
        used[row.side] += 1
        for j in range(e + 1, x + 1):  # retorno acumula de e+1 até x
            k = cal_idx.get(dates[j])
            if k is None:
                continue
            ret = r[j]
            if not np.isfinite(ret):
                continue
            if j == e + 1:
                ret -= COST            # custo de entrada
            if j == x:
                ret -= COST            # custo de saída
            if row.side == 1:
                long_sum[k] += ret; long_cnt[k] += 1
            else:
                short_sum[k] += ret; short_cnt[k] += 1
    with np.errstate(invalid="ignore"):
        long_ret = np.where(long_cnt > 0, long_sum / np.maximum(long_cnt, 1), 0.0)
        short_ret = np.where(short_cnt > 0, short_sum / np.maximum(short_cnt, 1), 0.0)
    active = (long_cnt > 0) | (short_cnt > 0)
    return long_ret, short_ret, active, used


def stats(r, active, label):
    r = r[active]
    if len(r) < 252:
        return f"{label:22s} série curta ({len(r)}d)"
    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    sh = ann / vol if vol > 0 else np.nan
    t = sh * np.sqrt(len(r) / 252)
    return f"{label:22s} ret {100*ann:+6.2f}%aa | vol {100*vol:5.2f}% | Sharpe {sh:+5.2f} | t(Sharpe) {t:+5.2f}"


def main():
    ev = pd.read_parquet(OUTDIR / "events_sp500_h2.parquet")
    tv = OUTDIR / "td_variants.parquet"
    if tv.exists():   # td_w = TD ponderada por palavras (Tabela 8 col 1 do paper)
        ev = ev.merge(pd.read_parquet(tv)[["call_id", "td_w"]], on="call_id", how="left")
        # sinal FE-consistente: percentil do td_w CONTRA O PRÓPRIO HISTÓRICO da
        # firma (estritamente passado, mín. 6 calls). Racional: o efeito da
        # Tabela 6 é identificado COM FE de firma — é desvio do próprio nível,
        # não comparação entre firmas (TD é persistente, Tabela 2 do paper).
        ev = ev.sort_values(["ticker", "cdate"]).reset_index(drop=True)
        def own_pct(s):
            out = np.full(len(s), np.nan)
            v = s.to_numpy(float)
            for i in range(len(v)):
                past = v[:i]
                past = past[np.isfinite(past)]
                if len(past) >= 6 and np.isfinite(v[i]):
                    out[i] = (past < v[i]).mean() + 0.5 * (past == v[i]).mean()
            return pd.Series(out, index=s.index)
        ev["td_w_own"] = ev.groupby("ticker")["td_w"].transform(own_pct)
    px = pd.concat([pd.read_parquet(p) for p in (SP / "price_shards").glob("*.parquet")],
                   ignore_index=True).sort_values(["ticker", "date"])
    mkt = px[px["ticker"] == "^GSPC"].dropna(subset=["adj_close"])
    cal = mkt["date"].to_numpy("datetime64[ns]")
    mret = mkt["adj_close"].pct_change().to_numpy()
    tick = {}
    for t, g in px[px["ticker"] != "^GSPC"].groupby("ticker"):
        g = g.dropna(subset=["adj_close"])
        tick[t] = (g["date"].to_numpy("datetime64[ns]"), g["adj_close"].pct_change().to_numpy())

    for sigcol, lbl in [("td", "SINAL PRIMÁRIO: TD crua (como o Angelo)"),
                        ("td_w", "TD PONDERADA por palavras (T8c1 do paper)"),
                        ("td_w_own", "td_w vs PRÓPRIO histórico (FE-consistente)"),
                        ("z", "braço secundário (log p/ DSR): z")]:
        if sigcol not in ev.columns:
            continue
        lr, sr, act, used = build_portfolio(ev, tick, cal, sigcol)
        ls = lr - sr
        print("\n" + "=" * 84)
        print(f"{lbl} | long {used[1]} eventos, short {used[-1]} | {act.sum()} dias ativos "
              f"({pd.Timestamp(cal[act.argmax()]).date()} em diante)")
        print("=" * 84)
        print(stats(ls, act, "LONG-SHORT (líquido)"))
        print(stats(lr, act, "só perna LONG (líq.)"))
        print(stats(sr, act, "só perna SHORT (líq.)"))
        print(stats(mret, act, "mercado ^GSPC (ref.)"))
    print("\nNota: expectativa pré-registrada (painel nulo): LS ~flat, Sharpe <= 0,2.")
    print("DONE.")


if __name__ == "__main__":
    main()
