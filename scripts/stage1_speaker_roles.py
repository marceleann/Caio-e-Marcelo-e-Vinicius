# -*- coding: utf-8 -*-
"""Stage 1 (base determinística) — dicionário de papéis por orador.

Classifica cada header de orador em {manager, analyst, operator, group} com
regras de alta confiança, medidas na base real (ver docs). O LLM (Claude) entra
DEPOIS, só no resíduo ambíguo + no mapa de identidade + na validação — não aqui.

Sinais (ordem de prioridade):
  1. "Operator" exato -> operator.
  2. rótulo de grupo ("Executives"/"Analysts"/"Unidentified...") -> group (fora do sinal).
  3. prefixo "Q -"/"A -" -> analista/gestão (formato antigo).
  4. cargo no header ("... Chief Financial Officer") -> manager.
  5. firma sell-side no header ("... Morgan Stanley") -> analyst.
  6. aparece em prepared remarks alguma vez E sob <=2 tickers -> manager
     (analista NUNCA fala em remarks; gestor fala sob a própria empresa).
  7. aparece sob >=3 tickers -> analyst (cobre várias empresas).
  8. aparece em remarks -> manager. Senão -> AMBÍGUO (só-Q&A, poucos tickers):
     rótulo provisório = papel majoritário da heurística atual + needs_llm=True.

Saída: data/interim/speaker_roles.parquet
  [speaker, role, source, needs_llm, n, n_tk, pct_rem, match_key, heur_role]
"""
from __future__ import annotations
import re, unicodedata
import pandas as pd
from pathlib import Path

INT = Path(__file__).resolve().parents[1] / "data" / "interim"

TITLE = re.compile(r"\b(chief|officer|ceo|cfo|coo|cto|president|vice president|vp|chairman|"
                   r"founder|treasurer|controller|general counsel|head of|director of|"
                   r"investor relations|executive|principal accounting)\b", re.I)
FIRM = re.compile(r"\b(securities|capital|partners|research|brokerage|& co|llc|l\.l\.c|"
                  r"morgan|goldman|sachs|jpmorgan|j\.p\. morgan|merrill|barclays|credit suisse|"
                  r"ubs|deutsche|citigroup|citi|wells fargo|jefferies|baird|cowen|piper|wedbush|"
                  r"raymond james|stifel|oppenheimer|needham|canaccord|bernstein|evercore|mizuho|"
                  r"nomura|rbc|bmo|keybanc|susquehanna|william blair|cantor|guggenheim|truist|"
                  r"loop capital|rosenblatt|davidson|macquarie|scotiabank|bank of america|bofa|"
                  r"new street|wolfe|redburn|northland|roth|craig|benchmark|mkm|argus|moffett|"
                  r"arete|melius|tigress|sanford|bnp|societe generale|hsbc|td cowen|itau|bradesco)\b", re.I)
GROUP = re.compile(r"^(executives?|analysts?|unidentified|company representative|"
                   r"conference call participants?|multiple speakers?|participants?)", re.I)
QP = re.compile(r"^Q\s*[-–:]"); AP = re.compile(r"^A\s*[-–:]")

# --- identidade (roles._match_key, replicado) ---
_QA = re.compile(r"^[QA]\s*[-–:]\s*"); _TS = re.compile(r"\s+[-–—]\s+.*$"); _CM = re.compile(r"(?<=[a-z])(?=[A-Z])")
def _clean(n):
    if not isinstance(n, str): return ""
    n = _QA.sub("", n.strip()); n = _TS.sub("", n); n = _CM.sub(" ", n)
    n = unicodedata.normalize("NFKD", n); n = "".join(c for c in n if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", n).strip().casefold()
def match_key(n):
    c = _clean(n)
    if not c: return ""
    t = [x for x in c.replace(".", " ").split() if len(x) > 1]
    return f"{t[-1]}|{t[0][0]}" if t else c


def main():
    r = pd.read_parquet(INT / "utterances_roles.parquet")
    qa_calls = set(r.loc[r["section"] == "qa", "call_id"].unique())
    s = r.assign(sp=r["speaker"].astype(str).str.strip(),
                 tk=r["call_id"].astype(str).str.split("_").str[0],
                 rem_clean=((r["section"] == "remarks") & (r["call_id"].isin(qa_calls))))
    heur = s.groupby("sp")["role"].agg(lambda x: x.value_counts().index[0]).rename("heur_role")
    g = s.groupby("sp").agg(n=("sp", "size"), n_tk=("tk", "nunique"),
                            n_rem=("section", lambda x: (x == "remarks").sum()),
                            n_rem_clean=("rem_clean", "sum"),
                            pct_rem=("section", lambda x: (x == "remarks").mean())).reset_index()
    g = g.merge(heur, left_on="sp", right_index=True)

    def _org_part(h):
        """Parte do header APÓS o 1º delimitador (nome | org). Evita bater firma no sobrenome."""
        parts = re.split(r"\s*[-–—,/]\s*", h, maxsplit=1)
        return parts[1] if len(parts) > 1 else ""

    def classify(row):
        # Hierarquia por CONFIABILIDADE do sinal (medida na base real):
        #  - cargo/firma no header e n_tk são robustos;
        #  - o rótulo de SEÇÃO (remarks/qa) é RUIDOSO (analistas acumulam remarks
        #    falsos em calls sem Q&A detectado), então só conta como sinal FRACO
        #    de gestor quando n_tk é baixo (single-company + remarks reais).
        h = row["sp"]
        if h.lower() == "operator": return "operator", "operator"
        if GROUP.match(h): return "group", "grupo"
        if QP.match(h): return "analyst", "prefixo_Q"
        if AP.match(h): return "manager", "prefixo_A"
        if TITLE.search(h): return "manager", "cargo_header"
        if FIRM.search(_org_part(h)): return "analyst", "firma_header"
        if row["n_tk"] >= 4: return "analyst", "multi_ticker"        # cobre >=4 empresas
        if row["n_tk"] <= 2 and row["n_rem_clean"] >= 1: return "manager", "1ticker+remarks"
        return "ambiguous", "residuo"                                 # n_tk 2-3 sem remarks: LLM decide

    res = g.apply(lambda row: pd.Series(classify(row)), axis=1)
    g["role"], g["source"] = res[0], res[1]
    g["needs_llm"] = g["role"] == "ambiguous"
    # rótulo provisório do resíduo = heurística atual (para ser usável já)
    g.loc[g["needs_llm"], "role"] = g.loc[g["needs_llm"], "heur_role"]
    g["match_key"] = g["sp"].map(match_key)
    g = g.rename(columns={"sp": "speaker"})

    dest = INT / "speaker_roles.parquet"
    g[["speaker", "role", "source", "needs_llm", "n", "n_tk", "n_rem", "n_rem_clean", "match_key", "heur_role"]].to_parquet(dest, index=False)
    print(f"Escrito {dest}: {len(g)} headers únicos")
    print("\npapel final (headers únicos):")
    print(g["role"].value_counts().to_string())
    print("\npapel final (ponderado por FALAS):")
    print(g.groupby("role")["n"].sum().sort_values(ascending=False).to_string())
    print("\nfonte da decisão (headers):")
    print(g["source"].value_counts().to_string())
    amb = g[g["needs_llm"]]
    print(f"\nprecisam do LLM (needs_llm): {len(amb)} headers, {int(amb['n'].sum())} falas "
          f"({100*amb['n'].sum()/g['n'].sum():.1f}%) — rótulo provisório = heurística atual")
    # quantos discordam entre determinístico e heurística antiga (onde melhoramos)
    conf = g[(~g["needs_llm"]) & (g["role"] != g["heur_role"])]
    print(f"headers onde o determinístico DIVERGE da heurística antiga: {len(conf)} "
          f"({int(conf['n'].sum())} falas) — melhorias/correções")
    print("DONE.")


if __name__ == "__main__":
    main()
