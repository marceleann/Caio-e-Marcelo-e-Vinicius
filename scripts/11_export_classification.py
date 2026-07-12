"""11 — Exporta a classificação FIEL de TODAS as calls da amostra (não uma amostra de 20).

Aplica a metodologia do Angelo (papéis + distância de tom entre gestores) a cada
call que tem disponibilidade e materializa o resultado, call a call, em dois CSVs
auditáveis:

- ``classificacao_por_call.csv`` — uma linha por call: nº de gestores e analistas
  identificados, distância de tom (crua e o sinal limpo), tom da call, retornos
  futuros, e se a call entrou no sinal operável (com o motivo quando não entrou).
- ``papeis_por_orador.csv`` — o papel inferido de CADA orador em CADA call
  (gestão/analista/moderador), verificável na amostra INTEIRA.

Uso:
    python scripts/11_export_classification.py --config config.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from tonediv.config import configure_logging, load_config


def _motivo(row: pd.Series) -> str:
    if pd.notna(row["tone_distance_clean"]):
        return "classificada — no sinal operável"
    if pd.isna(row["tone_distance"]):
        return "sem distância de tom: <2 gestores identificados ou Q&A não detectado"
    if pd.isna(row["decision_date"]):
        return "distância de tom OK, mas sem pregão de decisão (call não alinhada a preços)"
    faltando = [
        c
        for c in ("analyst_tone", "analyst_tone_distance", "size_proxy", "length")
        if c in row.index and pd.isna(row[c])
    ]
    if faltando:
        return f"distância de tom OK, mas confundidor ausente ({', '.join(faltando)})"
    return "distância de tom OK, mas histórico curto para residualizar (<250 calls anteriores)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta a classificação de TODAS as calls.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.logging)
    p = cfg.paths

    ev = pd.read_parquet(p.data_processed / "events.parquet")
    roles = pd.read_parquet(p.data_interim / "utterances_roles.parquet")

    # ---- 1) classificação por call (a amostra inteira) ----
    cols = {
        "call_id": ev["call_id"], "ticker": ev["ticker"],
        "data_call": ev["call_datetime"].dt.tz_convert(cfg.sample.timezone).dt.date,
        "n_gestores": ev["n_managers_td"], "n_analistas": ev.get("n_analysts_td"),
        "distancia_de_tom": ev["tone_distance"].round(4),
        "sinal_limpo": ev["tone_distance_clean"].round(4),
        "tom_da_call": ev["net_tone"].round(4),
        "ret_1mes": ev["ret_21"].round(4), "ret_3meses": ev["ret_63"].round(4),
    }
    per_call = pd.DataFrame(cols)
    per_call["no_sinal_operavel"] = ev["tone_distance_clean"].notna()
    per_call["motivo"] = ev.apply(_motivo, axis=1)
    per_call = per_call.sort_values(["ticker", "data_call"])
    per_call.to_csv(p.data_outputs / "classificacao_por_call.csv", index=False)

    # ---- 2) papéis por orador em TODAS as calls ----
    spk = (
        roles.sort_values(["call_id", "utterance_idx"])
        .groupby(["call_id", "speaker"], dropna=False, sort=True)
        .agg(papel=("role", "first"), secao_1a_fala=("section", "first"),
             n_falas=("role", "size"), primeira_fala=("text", "first"))
        .reset_index()
    )
    spk["primeira_fala"] = spk["primeira_fala"].str.slice(0, 120)
    spk.to_csv(p.data_outputs / "papeis_por_orador.csv", index=False)

    n = len(per_call)
    td = int(per_call["distancia_de_tom"].notna().sum())
    clean = int(per_call["no_sinal_operavel"].sum())
    print(f"[11] {n} calls classificadas | distância de tom em {td} ({td/n:.1%}) "
          f"| no sinal operável {clean} ({clean/n:.1%})")
    print(f"     papéis inferidos para {len(spk)} pares (call, orador) na amostra inteira")
    print(f"     -> data/outputs/classificacao_por_call.csv e papeis_por_orador.csv")
    print("     motivos (calls fora do sinal operável):")
    for motivo, c in per_call.loc[~per_call["no_sinal_operavel"], "motivo"].value_counts().items():
        print(f"       {c:>4}  {motivo}")


if __name__ == "__main__":
    main()
