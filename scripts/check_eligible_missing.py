# -*- coding: utf-8 -*-
"""Imprime quantas calls ELEGÍVEIS (CAR + controles base) ainda não têm shard
FinBERT — 0 significa fila elegível concluída. Usado pelo vigia noturno."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SH = ROOT / "data" / "interim" / "sentence_scores_shards"
BASE = ["disclosure_tone", "analyst_tone", "analyst_tone_distance",
        "length", "ln_mktcap", "lagged_td", "sue_pct"]
ev = pd.read_parquet(ROOT / "data" / "interim" / "sp500" / "events_sp500_car3.parquet")
elig = set(ev.dropna(subset=["car_m1p1"] + BASE)["call_id"])
done = {p.stem[5:] for p in SH.glob("call_*.parquet")}
print(len(elig - done))
