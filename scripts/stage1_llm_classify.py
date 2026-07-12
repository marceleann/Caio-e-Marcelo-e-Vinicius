# -*- coding: utf-8 -*-
"""Stage 1 (GenAI) — classificação de papel do orador via LLM, como ROBÔ reproduzível.

Um script que chama o Claude (CLI headless `claude -p`, modelo Haiku) em lotes
para classificar CADA header único de orador em {manager, analyst, operator,
group}, usando sinais determinísticos (nº de empresas, dá remarks, firma/cargo no
header) apenas como DICA no prompt. Cacheia por header (resumível) e cruza com o
classificador determinístico (`speaker_roles.parquet`) como validação.

Este É o uso de IA generativa no NÚCLEO do pipeline (edital, 15%): reproduzível
(qualquer um re-roda), com impacto efetivo (a tese inteira depende de separar
gestores de analistas), não declaratório. Prompt e modelo documentados abaixo e
em docs/GENAI_USAGE.md.

Uso:
  python scripts/stage1_llm_classify.py [batch=60] [max_headers=0(=todos)]
"""
from __future__ import annotations
import subprocess, json, re, sys, time, unicodedata
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INT = ROOT / "data" / "interim"
CACHE = INT / "llm_roles_cache.jsonl"
MODEL = "haiku"
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MAX_H = int(sys.argv[2]) if len(sys.argv) > 2 else 0

SYS = (
    "Você é uma função de rotulagem de dados de earnings calls. Classifique cada "
    "orador em EXATAMENTE um papel:\n"
    "- manager: executivo, fundador ou pessoa de RI da PRÓPRIA empresa que reporta\n"
    "- analyst: analista sell-side (equity research) que faz perguntas na call\n"
    "- operator: moderador/telefonista da call\n"
    "- group: rótulo coletivo, não um indivíduo (ex.: 'Executives', 'Unidentified "
    "Analyst', 'Conference Call Participants')\n"
    "Dicas fornecidas: nº de empresas distintas em que o nome aparece (analista "
    "cobre MUITAS; executivo aparece em 1, ou 2-3 se trocou de emprego), se dá "
    "prepared remarks (só a empresa dá remarks), e se há cargo/firma no header.\n"
    "Responda SOMENTE um array JSON, um objeto por orador, na ordem dada: "
    '[{"i": <número>, "role": <papel>}]. Sem texto, sem perguntas, sem explicação.'
)

TITLE = re.compile(r"\b(chief|officer|ceo|cfo|coo|cto|president|vice president|vp|chairman|"
                   r"founder|treasurer|controller|general counsel|head of|director of|"
                   r"investor relations|executive)\b", re.I)
FIRM = re.compile(r"[,\-–—/].*\b(securities|capital|partners|research|& co|morgan|goldman|"
                  r"merrill|barclays|ubs|deutsche|citi|jefferies|baird|cowen|piper|stifel|"
                  r"oppenheimer|needham|bernstein|evercore|mizuho|nomura|rbc|bmo|keybanc|"
                  r"susquehanna|william blair|guggenheim|bank of america|wells fargo)\b", re.I)


def features():
    r = pd.read_parquet(INT / "utterances_roles.parquet")
    qa = set(r.loc[r["section"] == "qa", "call_id"].unique())
    r = r.assign(sp=r["speaker"].astype(str).str.strip(),
                 tk=r["call_id"].astype(str).str.split("_").str[0],
                 rc=((r["section"] == "remarks") & (r["call_id"].isin(qa))))
    g = r.groupby("sp").agg(n=("sp", "size"), n_tk=("tk", "nunique"),
                            rc=("rc", "sum"),
                            tickers=("tk", lambda x: ",".join(sorted(set(x))[:4]))).reset_index()
    g["hint"] = g.apply(lambda x: (
        f'{x["n_tk"]} empresa(s) [{x["tickers"]}]'
        f'{"; dá remarks" if x["rc"] >= 1 else "; nunca dá remarks"}'
        f'{"; cargo no header" if TITLE.search(x["sp"]) else ""}'
        f'{"; firma no header" if FIRM.search(x["sp"]) else ""}'), axis=1)
    return g.sort_values("n", ascending=False).reset_index(drop=True)


def call_llm(lines):
    prompt = SYS + "\n\nOradores:\n" + "\n".join(lines)
    r = subprocess.run("claude -p --model " + MODEL, input=prompt, shell=True,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=180)
    out = r.stdout or ""
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def load_cache():
    done = {}
    if CACHE.exists():
        for ln in CACHE.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(ln); done[o["h"]] = o["role"]
            except Exception:
                pass
    return done


def main():
    g = features()
    if MAX_H:
        g = g.head(MAX_H)
    done = load_cache()
    todo = g[~g["sp"].isin(done)].reset_index(drop=True)
    print(f"headers: {len(g)} | já no cache: {len(done)} | a fazer: {len(todo)} | modelo: {MODEL}")
    t0 = time.time()
    with CACHE.open("a", encoding="utf-8") as fh:
        for start in range(0, len(todo), BATCH):
            chunk = todo.iloc[start:start + BATCH]
            lines = [f'{i+1}) "{row.sp}" — {row.hint}' for i, row in enumerate(chunk.itertuples())]
            res = call_llm(lines)
            ok = 0
            if res:
                by_i = {int(o["i"]): o.get("role", "") for o in res if isinstance(o, dict) and "i" in o}
                for i, row in enumerate(chunk.itertuples()):
                    role = by_i.get(i + 1, "")
                    if role in ("manager", "analyst", "operator", "group"):
                        fh.write(json.dumps({"h": row.sp, "role": role}, ensure_ascii=False) + "\n")
                        ok += 1
                fh.flush()
            el = time.time() - t0
            nb = start // BATCH + 1
            print(f"  lote {nb} ({start+len(chunk)}/{len(todo)}): {ok}/{len(chunk)} ok | {el:.0f}s")
    # -------- merge + validação --------
    done = load_cache()
    g["llm_role"] = g["sp"].map(done)
    det = pd.read_parquet(INT / "speaker_roles.parquet")[["speaker", "role"]]
    det = det.rename(columns={"role": "det_role"})
    norm = {"management": "manager", "manager": "manager", "analyst": "analyst",
            "operator": "operator", "group": "group", "unknown": "unknown"}
    m = g.merge(det, left_on="sp", right_on="speaker", how="left")
    m["det_n"] = m["det_role"].map(norm)
    out = INT / "speaker_roles_llm.parquet"
    m[["sp", "n_tk", "n", "hint", "llm_role", "det_role"]].rename(columns={"sp": "speaker"}).to_parquet(out, index=False)
    cov = m["llm_role"].notna()
    print(f"\nclassificados pelo LLM: {int(cov.sum())}/{len(m)} ({100*cov.mean():.1f}%) — escrito {out}")
    both = m[cov & m["det_n"].notna() & (m["det_n"] != "unknown")]
    agree = (both["llm_role"] == both["det_n"])
    print(f"concordância LLM×determinístico (onde ambos decidem): {100*agree.mean():.1f}% "
          f"({int(agree.sum())}/{len(both)} headers; {int(both.loc[agree,'n'].sum())}/{int(both['n'].sum())} falas)")
    print("discordâncias por tipo (por FALAS):")
    dis = both[~agree]
    print(dis.groupby(["det_n", "llm_role"])["n"].sum().sort_values(ascending=False).head(8).to_string())
    print(f"\ntempo total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
