# -*- coding: utf-8 -*-
"""Stage 1 (validação) — matriz de confusão da classificação de papel do LLM.

Método exigido: amostra de 300 headers rotulados de forma INDEPENDENTE do LLM que
está sendo testado (padrão-ouro humano, ou modelo independente lendo as falas),
comparados com o papel do LLM (Haiku) → matriz de confusão + acurácia + P/R.

Dois modos:
  sample : sorteia 300 headers (seed 42) estratificados por papel do LLM, anexa
           uma fala de exemplo, e exporta o CSV de rotulagem SEM mostrar o papel do
           LLM (para não enviesar quem rotula). Coluna `gold_role` fica em branco.
  matrix : lê o CSV com `gold_role` preenchido, cruza com o papel do LLM e imprime
           a matriz de confusão + métricas.

Uso:
  python scripts/stage1_confusion_matrix.py sample [n=300]
  python scripts/stage1_confusion_matrix.py matrix
"""
from __future__ import annotations
import sys, json, re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INT = ROOT / "data" / "interim"
CACHE = INT / "llm_roles_cache.jsonl"
CSV = INT / "speaker_validation_sample.csv"
SEED = 42
CLASSES = ["manager", "analyst", "operator", "group"]
# quantos sortear por classe (gestor/analista dominam a importância; operador é trivial)
QUOTA = {"manager": 130, "analyst": 130, "group": 30, "operator": 10}


def load_llm():
    done = {}
    if CACHE.exists():
        for ln in CACHE.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(ln); done[o["h"]] = o["role"]
            except Exception:
                pass
    return done


def sample_headers(n):
    # estratifica pelo classificador DETERMINÍSTICO (independente do LLM testado),
    # cobrindo todos os 12.678 headers — a amostra não depende da saída do Haiku.
    det = pd.read_parquet(INT / "speaker_roles.parquet")[["speaker", "role"]]
    norm = {"management": "manager", "manager": "manager", "analyst": "analyst",
            "operator": "operator", "group": "group", "unknown": "analyst"}
    det["det"] = det["role"].map(norm)
    r = pd.read_parquet(INT / "utterances_roles.parquet")
    r = r.assign(sp=r["speaker"].astype(str).str.strip(),
                 tk=r["call_id"].astype(str).str.split("_").str[0])
    # feats + fala de exemplo (a mais longa, evidência p/ julgar o papel)
    def example(txts):
        cand = sorted((str(t) for t in txts), key=len, reverse=True)[:1]
        t = cand[0] if cand else ""
        return re.sub(r"\s+", " ", t)[:240]
    g = r.groupby("sp").agg(n=("sp", "size"), n_tk=("tk", "nunique"),
                            tickers=("tk", lambda x: ",".join(sorted(set(x))[:4])),
                            sample=("text", example)).reset_index()
    g = g.merge(det[["speaker", "det"]], left_on="sp", right_on="speaker", how="left")
    parts = []
    for c in CLASSES:
        pool = g[g["det"] == c]
        k = min(QUOTA[c], len(pool))
        if k > 0:
            parts.append(pool.sample(n=k, random_state=SEED))
    sample = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    out = sample[["sp", "tickers", "n", "n_tk", "sample"]].copy()
    out.columns = ["header", "tickers", "n_falas", "n_empresas", "fala_exemplo"]
    out["gold_role"] = ""   # preencher: manager | analyst | operator | group
    out.to_csv(CSV, index=False, encoding="utf-8-sig")
    print(f"Exportado {CSV} — {len(out)} headers para rotular (gold_role em branco).")
    print("Distribuição por classe DETERMINÍSTICA (só para balancear a amostra):")
    print(sample["det"].value_counts().to_string())
    print("\nComo preencher: julgue cada um pela fala_exemplo + tickers; escreva")
    print("manager/analyst/operator/group em gold_role. Depois rode: matrix")


def confusion():
    if not CSV.exists():
        print("CSV não existe — rode o modo 'sample' antes."); return
    s = pd.read_csv(CSV, encoding="utf-8-sig")
    llm = load_llm()  # papel do LLM vindo do cache (completo no momento da matriz)
    s["llm_role"] = s["header"].map(llm)
    s["gold_role"] = s["gold_role"].astype(str).str.strip().str.lower()
    miss = s["llm_role"].isna().sum()
    if miss:
        print(f"aviso: {miss} headers ainda sem papel do LLM (classificação incompleta).")
    s = s[s["gold_role"].isin(CLASSES) & s["llm_role"].notna()]
    if len(s) == 0:
        print("nenhuma linha com gold_role preenchido (manager/analyst/operator/group)."); return
    print(f"headers rotulados: {len(s)}")
    cm = pd.crosstab(s["gold_role"], s["llm_role"], rownames=["HUMANO"], colnames=["LLM"], dropna=False)
    cm = cm.reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    print("\nMATRIZ DE CONFUSÃO (linha=humano, coluna=LLM):")
    print(cm.to_string())
    acc = (s["gold_role"] == s["llm_role"]).mean()
    print(f"\nacurácia global: {100*acc:.1f}% ({int((s['gold_role']==s['llm_role']).sum())}/{len(s)})")
    print("\npor classe (precision / recall):")
    for c in CLASSES:
        tp = int(((s["llm_role"] == c) & (s["gold_role"] == c)).sum())
        pred = int((s["llm_role"] == c).sum()); true = int((s["gold_role"] == c).sum())
        prec = tp / pred if pred else float("nan")
        rec = tp / true if true else float("nan")
        print(f"  {c:9s}: P={prec:.2f} R={rec:.2f}  (n_humano={true})")
    # foco tese: gestor x analista (o que corrompe a TD se errado)
    ma = s[s["gold_role"].isin(["manager", "analyst"]) & s["llm_role"].isin(["manager", "analyst"])]
    if len(ma):
        acc_ma = (ma["gold_role"] == ma["llm_role"]).mean()
        print(f"\nacurácia gestor↔analista (o que importa p/ a TD): {100*acc_ma:.1f}% (n={len(ma)})")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sample"
    if mode == "sample":
        sample_headers(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    elif mode == "matrix":
        confusion()
    else:
        print("modo inválido: use 'sample' ou 'matrix'")


if __name__ == "__main__":
    main()
