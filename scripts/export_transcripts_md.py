# -*- coding: utf-8 -*-
"""Exporta uma AMOSTRA de transcrições tratadas em Markdown legível no GitHub.

Objetivo: a orientadora ler no navegador, sem baixar nada. Cada call vira um
.md com: cabeçalho (empresa, data, TD da call), e as falas na ordem original,
cada uma anotada com o papel atribuído pelo nosso classificador
([GESTOR]/[ANALISTA]/[OPERADOR]/[GRUPO]).

Amostra (defaults, ajustáveis na lista PICKS):
  - calls icônicas de empresas conhecidas (AAPL, MSFT, JPM, XOM, WMT);
  - as 5 calls de MAIOR e as 5 de MENOR Tone Distance ponderada (>=3 gestores,
    >=2 analistas) — para visualizar o fenômeno da divergência;
  - 10 aleatórias (seed 42) — sem viés de escolha.
NOTA de licença: amostra pequena para leitura acadêmica; o dataset completo
permanece no HuggingFace (kurry/sp500_earnings_transcripts), citado no README.

Saída: docs/transcricoes_amostra/*.md + README.md (índice)
Uso: python scripts/export_transcripts_md.py
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonediv.config import load_config
from tonediv.data import transcripts as tr

OUT = ROOT / "docs" / "transcricoes_amostra"
PICKS = ["AAPL_2020Q4", "MSFT_2023Q2", "JPM_2019Q1", "XOM_2015Q3", "WMT_2021Q1"]
ROLE_PT = {"manager": "GESTOR", "analyst": "ANALISTA",
           "operator": "OPERADOR", "group": "GRUPO"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "config.yaml"))

    sc = pd.read_parquet(ROOT / "data" / "interim" / "sp500" / "speaker_counts.parquet")
    ev = pd.read_parquet(ROOT / "data" / "interim" / "sp500" / "events_sp500_car3.parquet")
    var = pd.read_parquet(ROOT / "data" / "interim" / "sp500" / "td_variants.parquet")
    ev = ev.merge(var[["call_id", "td_w"]], on="call_id", how="left")

    # extremos de TD ponderada com estrutura rica (>=3 gestores, >=2 analistas)
    n_an = (sc[sc["role_eff"] == "analyst"].groupby("call_id")["mk"].nunique()
            .rename("n_analysts").reset_index())
    cand = ev.dropna(subset=["td_w"]).merge(n_an, on="call_id", how="left")
    cand = cand[(cand["n_managers"] >= 3) & (cand["n_analysts"] >= 2)]
    hi = cand.nlargest(5, "td_w")["call_id"].tolist()
    lo = cand.nsmallest(5, "td_w")["call_id"].tolist()
    rng = np.random.default_rng(42)
    rand = list(rng.choice(cand["call_id"].to_numpy(), 10, replace=False))
    wanted = list(dict.fromkeys(PICKS + hi + lo + rand))

    calls = tr.prepare_calls(tr.load_hf_transcripts(cfg), cfg)
    calls = calls[calls["call_id"].isin(wanted)]
    roles = {(r.call_id, r.utterance_idx): r.role_eff for r in
             sc[sc["call_id"].isin(wanted)].itertuples()}
    meta = ev.set_index("call_id")

    utts = tr.build_utterances(calls)
    index_rows = []
    for cid, g in utts.groupby("call_id"):
        m = meta.loc[cid] if cid in meta.index else None
        tk, per = cid.split("_")
        lines = [f"# {tk} — {per}", ""]
        if m is not None:
            lines += [f"**Data da call:** {m['cdate']}  ",
                      f"**Tone Distance (LM):** {m['td']:.4f} | **ponderada por palavras:** "
                      f"{m['td_w']:.4f} | **gestores:** {int(m['n_managers'])}", "",
                      "_Papéis atribuídos pelo classificador do projeto "
                      "(LLM + regras determinísticas, acurácia ~90% validada "
                      "por matriz de confusão)._", "", "---", ""]
        for r in g.sort_values("utterance_idx").itertuples():
            role = ROLE_PT.get(roles.get((cid, r.utterance_idx), "?"), "?")
            txt = str(r.text).replace("\n", " ").strip()
            lines.append(f"**[{role}] {r.speaker}:** {txt}")
            lines.append("")
        (OUT / f"{cid}.md").write_text("\n".join(lines), encoding="utf-8")
        tag = ("alta TD" if cid in hi else "baixa TD" if cid in lo
               else "icônica" if cid in PICKS else "aleatória")
        td_val = f"{m['td_w']:.4f}" if m is not None else "—"
        index_rows.append((cid, tag, td_val))

    idx = ["# Amostra de transcrições tratadas", "",
           "Cada arquivo é uma earnings call completa com o papel de cada orador",
           "atribuído pelo pipeline do projeto. Amostra: 5 icônicas, 5 de maior e",
           "5 de menor Tone Distance ponderada (>=3 gestores), 10 aleatórias (seed 42).",
           "Dataset completo (33.362 calls): "
           "[kurry/sp500_earnings_transcripts](https://huggingface.co/datasets/kurry/sp500_earnings_transcripts).",
           "", "| call | seleção | TD ponderada |", "|---|---|---|"]
    for cid, tag, td_val in sorted(index_rows):
        idx.append(f"| [{cid}]({cid}.md) | {tag} | {td_val} |")
    (OUT / "README.md").write_text("\n".join(idx), encoding="utf-8")
    print(f"exportadas {len(index_rows)} calls -> {OUT}")


if __name__ == "__main__":
    main()
