# -*- coding: utf-8 -*-
"""Stage 3 — Tone Distance (Angelo) a partir dos scores por SENTENÇA + debias.

Constrói a métrica fiel à Eq.(1) do Angelo, medindo o tom com FinBERT por
sentença e agregando ao gestor como FRAÇÃO DE SENTENÇAS positivas/negativas
(análogo direto da fração de palavras L-M). Depois:

- mede o viés de volume (corr da TD com a nº de sentenças do gestor mais quieto);
- aplica o PERMUTATION NULL (teste de randomização condicional, Fisher 1935):
  para cada call, embaralha as sentenças entre os gestores PRESERVANDO quantas
  cada um falou; sob esse nulo ninguém discorda -> E[TD_null]. Sinal debiasado
  TD_adj = TD - E[TD_null]; padronizado z = (TD - E)/sd. Custo: só numpy.
- sanity: corr(TD_adj, sentenças do gestor mais quieto) deve cair a ~0.

Saída: data/interim/tone_distance.parquet (uma linha por call).
Uso: python scripts/stage3_tone_distance.py [min_manager_sentences=5] [n_perm=200]
"""
from __future__ import annotations
import sys, re, unicodedata, time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INT = ROOT / "data" / "interim"
SHARDS = INT / "sentence_scores_shards"
SEED = 42

MIN_SENT = int(sys.argv[1]) if len(sys.argv) > 1 else 5   # mín. de sentenças por gestor
N_PERM = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ROLE_SRC = sys.argv[3] if len(sys.argv) > 3 else "heur"   # heur (heurística) | llm (Stage 1 GenAI)
MIN_MGR = 2

# ---- identidade do gestor (mesma lógica de roles._match_key) ----
_QA = re.compile(r"^[QA]\s*[-–:]\s*"); _TS = re.compile(r"\s+[-–—]\s+.*$"); _CM = re.compile(r"(?<=[a-z])(?=[A-Z])")
def _clean(n):
    if not isinstance(n, str): return ""
    n = _QA.sub("", n.strip()); n = _TS.sub("", n); n = _CM.sub(" ", n)
    n = unicodedata.normalize("NFKD", n); n = "".join(c for c in n if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", n).strip().casefold()
def mkey(n):
    c = _clean(n)
    if not c: return ""
    t = [x for x in c.replace(".", " ").split() if len(x) > 1]
    return f"{t[-1]}|{t[0][0]}" if t else c

def td_from_points(fp, fn):
    """Tone Distance = média das distâncias euclidianas ao centroide."""
    cp, cn = fp.mean(), fn.mean()
    return float(np.sqrt((fp - cp) ** 2 + (fn - cn) ** 2).mean())

def main():
    files = sorted(SHARDS.glob("call_*.parquet"))
    print(f"shards disponíveis: {len(files)} | MIN_SENT={MIN_SENT} N_PERM={N_PERM}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # papel EFETIVO: heurística (coluna role), LLM (Stage 1) ou LLM de ALTA
    # CONFIANÇA (llm_hc: LLM e determinístico CONCORDAM que é gestor — dois
    # métodos independentes; corta o ruído de medida da classificação).
    if ROLE_SRC in ("llm", "llm_hc"):
        sr = pd.read_parquet(INT / "speaker_roles_llm.parquet")
        if ROLE_SRC == "llm_hc":
            det_mgr = sr["det_role"].isin(["manager", "management"])
            sr = sr.assign(eff=np.where((sr["llm_role"] == "manager") & det_mgr, "manager",
                                        sr["llm_role"].where(sr["llm_role"] != "manager", "uncertain")))
            role_map = sr.set_index("speaker")["eff"].to_dict()
        else:
            role_map = sr.set_index("speaker")["llm_role"].to_dict()
        heur_map = {"management": "manager", "analyst": "analyst", "operator": "operator", "unknown": "analyst"}
        df["role_eff"] = df["speaker"].map(role_map).fillna(df["role"].map(heur_map))
        mgmt_val = "manager"
    else:
        df["role_eff"] = df["role"]
        mgmt_val = "management"
    mg = df[df["role_eff"] == mgmt_val].copy()
    mg["mk"] = mg["speaker"].map(mkey)
    mg = mg[mg["mk"] != ""]
    lab = mg[["p_neg", "p_neu", "p_pos"]].to_numpy().argmax(1)  # 0=neg,1=neu,2=pos
    mg["is_pos"] = (lab == 2).astype(np.float64)
    mg["is_neg"] = (lab == 0).astype(np.float64)

    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    for i, (cid, g) in enumerate(mg.groupby("call_id", sort=True)):
        # contagem por gestor
        gm = g.groupby("mk")
        cnt = gm.size()
        keep = cnt[cnt >= MIN_SENT].index
        gg = g[g["mk"].isin(keep)]
        n_mgr = len(keep)
        if n_mgr < MIN_MGR:
            continue
        # coord real (hard-label) + soft
        agg = gg.groupby("mk").agg(n=("is_pos", "size"),
                                   fp=("is_pos", "mean"), fn=("is_neg", "mean"),
                                   sp=("p_pos", "mean"), sn=("p_neg", "mean"))
        td_hard = td_from_points(agg["fp"].to_numpy(), agg["fn"].to_numpy())
        td_soft = td_from_points(agg["sp"].to_numpy(), agg["sn"].to_numpy())
        counts = agg["n"].to_numpy()
        min_sent = int(counts.min())
        avg_pos = float(gg["is_pos"].mean())
        # ---- permutation null (hard-label) ----
        pos = gg["is_pos"].to_numpy(); neg = gg["is_neg"].to_numpy()
        N = len(pos)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        tds = np.empty(N_PERM)
        for b in range(N_PERM):
            perm = rng.permutation(N)
            fp = np.add.reduceat(pos[perm], starts) / counts
            fn = np.add.reduceat(neg[perm], starts) / counts
            tds[b] = td_from_points(fp, fn)
        e_null, sd_null = float(tds.mean()), float(tds.std())
        rows.append(dict(call_id=cid, n_managers=n_mgr, min_sent=min_sent,
                         avg_pos=avg_pos, td=td_hard, td_soft=td_soft,
                         e_td_null=e_null, sd_td_null=sd_null,
                         td_adj=td_hard - e_null,
                         z=(td_hard - e_null) / sd_null if sd_null > 1e-9 else np.nan))
        if (i + 1) % 300 == 0:
            print(f"  {i+1} calls processadas ({time.time()-t0:.0f}s)")
    out = pd.DataFrame(rows)
    dest = INT / f"tone_distance_{ROLE_SRC}.parquet"
    out.to_parquet(dest, index=False)
    out.to_parquet(INT / "tone_distance.parquet", index=False)  # default p/ Stage 4
    print(f"\nEscrito {dest} ({ROLE_SRC}): {len(out)} calls com TD")

    # ---------- relatório ----------
    def stats(s):
        return f"mean={s.mean():.4f} median={s.median():.4f} sd={s.std():.4f} CV={s.std()/s.mean():.3f}"
    print("\n== TONE DISTANCE (hard-label, fração de sentenças) ==")
    print(" ", stats(out["td"]))
    print("   paper Angelo (L-M): mean 0.0079 median 0.0074 sd 0.0048 CV 0.61")
    print("== TD debiasada (TD_adj = TD - E[null]) ==")
    print(" ", stats(out["td_adj"].clip(lower=-1)))
    print(f"   E[null] médio={out['e_td_null'].mean():.4f}  (fração da TD crua que é ruído)")
    print("\n== CORRELAÇÕES (viés de volume) ==")
    for col, lbl in [("min_sent", "sentenças do gestor mais quieto"),
                     ("n_managers", "nº de gestores"), ("avg_pos", "tom positivo médio")]:
        print(f"  corr(TD_crua, {lbl:32s}): {out['td'].corr(out[col]):+.3f}")
    print("  --- após debias ---")
    for col, lbl in [("min_sent", "sentenças do gestor mais quieto"),
                     ("n_managers", "nº de gestores"), ("avg_pos", "tom positivo médio")]:
        print(f"  corr(TD_adj , {lbl:32s}): {out['td_adj'].corr(out[col]):+.3f}")
    print(f"  corr(z      , sentenças do gestor mais quieto): {out['z'].corr(out['min_sent']):+.3f}  (deve ~0)")
    c = out["td"].corr(out["min_sent"])
    rule = ("controles bastam" if abs(c) < 0.10 else
            "null é sinal primário" if abs(c) > 0.25 else "rodar ambos (zona 0.10-0.25)")
    print(f"\n  DECISION RULE (|corr TD_crua x quietest|={abs(c):.3f}) -> {rule}")
    print("\n== TD média por nº de gestores (checar monotonia/viés) ==")
    print(out.groupby("n_managers")["td"].agg(["mean", "count"]).head(8).to_string())
    print("\nDONE.")

if __name__ == "__main__":
    main()
