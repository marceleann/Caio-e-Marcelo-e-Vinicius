# -*- coding: utf-8 -*-
"""Repassada no texto do S&P 500 -> contagens LM POR FALA (ordem preservada).

Por que existe: a Fase A agrega as contagens por call e descarta o nível de
orador/fala, mas três construções do Angelo (2025) precisam dele:
  - Tabela 7 (Adjusted TD): subtrai as palavras de tom da PERGUNTA do analista
    da RESPOSTA da gestão -> exige o pareamento pergunta->resposta em ordem;
  - Tabela 8 (TD ponderada por palavras / excluindo oradores abaixo da mediana)
    -> exige contagens por gestor;
  - "Analyst Tone Dispersion" (Apêndice A) = DESVIO-PADRÃO do net tone entre
    analistas (a Fase A calculou distância euclidiana; aqui dá para corrigir).

Mesma maquinaria da Fase A (mesmos papéis: cache LLM + determinístico validado;
mesmo dicionário LM oficial; mesmos filtros de qualidade), mas a saída é uma
linha POR FALA: call_id, utterance_idx, mk (chave do orador), role_eff, section,
nw, npos, nneg. Texto descartado após contagem (memória bounded).

Saída: data/interim/sp500/speaker_counts.parquet
Uso: python scripts/sp500_speaker_counts.py [chunk=2000]
"""
from __future__ import annotations
import json, re, sys, time, unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonediv.config import load_config
from tonediv.data import transcripts as tr

INT, RAW = ROOT / "data" / "interim", ROOT / "data" / "raw"
OUTDIR = INT / "sp500"
CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
TOKEN = re.compile(r"[A-Za-z]+")

# ---------- identidade e classificador: IDÊNTICOS à Fase A ----------
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
                  r"arete|melius|tigress|sanford|bnp|hsbc|prudential|bear stearns|wachovia)\b", re.I)
GROUP = re.compile(r"^(executives?|analysts?|unidentified|company representative|"
                   r"conference call participants?|multiple speakers?|participants?)", re.I)
QP = re.compile(r"^Q\s*[-–:]"); AP = re.compile(r"^A\s*[-–:]")

def org_part(h):
    parts = re.split(r"\s*[-–—,/]\s*", h, maxsplit=1)
    return parts[1] if len(parts) > 1 else ""

def det_classify(h, n_tk, n_rem_clean):
    if h.lower() == "operator": return "operator"
    if GROUP.match(h): return "group"
    if QP.match(h): return "analyst"
    if AP.match(h): return "manager"
    if TITLE.search(h): return "manager"
    if FIRM.search(org_part(h)): return "analyst"
    if n_tk >= 4: return "analyst"
    if n_tk <= 2 and n_rem_clean >= 1: return "manager"
    if n_rem_clean >= 1: return "manager"
    return "analyst"


def main():
    t00 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "config.yaml"))

    lmd = pd.read_csv(RAW / "LM_MasterDictionary.csv")
    POS = set(lmd.loc[lmd["Positive"] != 0, "Word"].str.upper())
    NEG = set(lmd.loc[lmd["Negative"] != 0, "Word"].str.upper())
    print(f"LM: {len(POS)} pos, {len(NEG)} neg", flush=True)

    llm_map = {}
    cache = INT / "llm_roles_cache.jsonl"
    if cache.exists():
        for ln in cache.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(ln); llm_map[o["h"]] = o["role"]
            except Exception:
                pass
    print(f"cache LLM: {len(llm_map)} headers", flush=True)

    print("carregando HF (cache local)...", flush=True)
    calls = tr.prepare_calls(tr.load_hf_transcripts(cfg), cfg)
    print(f"calls: {len(calls)} ({time.time()-t00:.0f}s)", flush=True)

    q = cfg.quality
    reduced = []
    for i0 in range(0, len(calls), CHUNK):
        chunk = calls.iloc[i0:i0 + CHUNK]
        utts = tr.build_utterances(chunk)
        meta = tr.calls_metadata(chunk, utts)
        keep_ids = set(meta.loc[(meta["n_chars_total"] >= q.min_chars_call)
                                & (meta["n_utterances"] >= q.min_utterances_call), "call_id"])
        utts = utts[utts["call_id"].isin(keep_ids)].reset_index(drop=True)
        if utts.empty:
            continue
        from tonediv.nlp.roles import infer_roles
        ann = infer_roles(utts, cfg.roles)
        nw = np.empty(len(ann), dtype=np.int32)
        npos = np.empty(len(ann), dtype=np.int32)
        nneg = np.empty(len(ann), dtype=np.int32)
        for j, txt in enumerate(ann["text"].astype(str)):
            toks = [w.upper() for w in TOKEN.findall(txt)]
            nw[j] = len(toks)
            npos[j] = sum(1 for w in toks if w in POS)
            nneg[j] = sum(1 for w in toks if w in NEG)
        red = ann[["call_id", "utterance_idx", "speaker", "section"]].copy()
        red["sp"] = red["speaker"].astype(str).str.strip()
        red["nw"], red["npos"], red["nneg"] = nw, npos, nneg
        red = red.drop(columns=["speaker"])
        reduced.append(red)
        print(f"  bloco {i0//CHUNK+1}/{(len(calls)+CHUNK-1)//CHUNK} ({time.time()-t00:.0f}s)", flush=True)
    df = pd.concat(reduced, ignore_index=True)
    del reduced
    df["tk"] = df["call_id"].astype(str).str.split("_").str[0]

    # papéis globais (idêntico à Fase A)
    qa_calls = set(df.loc[df["section"] == "qa", "call_id"].unique())
    df["rem_clean"] = (df["section"] == "remarks") & df["call_id"].isin(qa_calls)
    stats = df.groupby("sp").agg(n_tk=("tk", "nunique"), n_rem_clean=("rem_clean", "sum"))
    role_map = {}
    for sp, row in stats.iterrows():
        r = llm_map.get(sp)
        if r is None:
            r = det_classify(sp, int(row["n_tk"]), int(row["n_rem_clean"]))
        role_map[sp] = r
    df["role_eff"] = df["sp"].map(role_map)
    df["mk"] = df["sp"].map(mkey)

    out = df[["call_id", "utterance_idx", "mk", "role_eff", "section", "nw", "npos", "nneg"]]
    dest = OUTDIR / "speaker_counts.parquet"
    out.to_parquet(dest, index=False)
    print(f"\nEscrito {dest}: {len(out)} falas, {out['call_id'].nunique()} calls "
          f"({(time.time()-t00)/60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
