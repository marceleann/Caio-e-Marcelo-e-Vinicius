# -*- coding: utf-8 -*-
"""Gold labels ROBUSTOS: regras confiáveis (cargo/firma/operator/grupo/prefixo/
'Analyst for') = ground truth correto; nomes pelados casados por HEADER EXATO via
OVERRIDES (sem risco de alinhamento posicional). Preenche gold_role e roda a matriz.

Uso:
  python scripts/stage1_gold_rules.py residual   # lista os pelados p/ rotular
  python scripts/stage1_gold_rules.py fill       # aplica regras+overrides, escreve gold
"""
import sys, csv, re, json
from pathlib import Path
from collections import Counter

INT = Path(r"C:\Users\Marcelo\Desafio Itaú QAI 26\data\interim")
CSV = INT / "speaker_validation_sample.csv"

TITLE = re.compile(r"\b(chief|officer|ceo|cfo|coo|cto|president|vice president|vp|chairman|"
                   r"founder|treasurer|controller|general counsel|head of|director of|"
                   r"investor relations|executive|group president)\b", re.I)
FIRM = re.compile(r"[,\-–—/].*\b(securities|capital|partners|research|& co|llc|inc\.|morgan|"
                  r"goldman|sachs|merrill|barclays|ubs|deutsche|citi|jefferies|baird|cowen|"
                  r"piper|wedbush|raymond james|stifel|oppenheimer|needham|canaccord|bernstein|"
                  r"evercore|mizuho|nomura|rbc|bmo|keybanc|susquehanna|william blair|cantor|"
                  r"guggenheim|bank of america|wells fargo|wachovia|bear stearns|caris|kaufman|"
                  r"gleacher|kessler|northland|roth|benchmark|thinkequity|isi|fbr|prudential|"
                  r"agricole|weisel|stephens|avian|charter equity|a\. g\. edwards|ag edwards|"
                  r"centennial|first manhattan|lime rock|standard & poor|sanders morris|"
                  r"pacific crest|pacific growth|auriga|us steel|friedman|ROK|Millman|Global Crown|"
                  r"bnp|scotiabank|d\.a\. davidson|davidson|morgan keegan|new street|melius|"
                  r"tigress|redburn)\b", re.I)
GROUP = re.compile(r"^(executives?|analysts?|unidentified|company representative|"
                   r"conference call|multiple speakers?|participants?)", re.I)


def rule_label(h):
    hl = h.strip()
    low = hl.lower()
    if low in ("operator",): return "operator"
    # analista ESPECÍFICO (mesmo sem nome) -> analyst, ANTES do teste de coletivo
    if hl.startswith("Analyst for") or low.startswith("analyst –") or low.startswith("analyst -"): return "analyst"
    if low.startswith("unidentified analyst"): return "analyst"
    # coletivos VERDADEIROS / lado-empresa não identificado -> group
    if re.match(r"^(executives?$|analysts?$|conference call|multiple speakers?|participants?$|"
                r"unidentified (participant|compan|conference|speaker))", hl, re.I): return "group"
    if re.match(r"^[Qq]\s*[-–:]", hl): return "analyst"
    if re.match(r"^[Aa]\s*[-–:]", hl): return "manager"
    if TITLE.search(hl): return "manager"
    # firma na parte após o 1º delimitador
    parts = re.split(r"\s*[-–—,/]\s*", hl, maxsplit=1)
    org = parts[1] if len(parts) > 1 else ""
    if FIRM.search(org): return "analyst"
    # header que é SÓ um nome de firma (sem delimitador)
    if FIRM.search(" - " + hl): return "analyst"
    return None


# nomes pelados / artefatos -> rótulo, julgados pela fala de exemplo (header EXATO)
OVERRIDES = {}  # preenchido no modo 'fill' abaixo


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "residual"
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    if mode == "residual":
        res = [r for r in rows if rule_label(r["header"]) is None]
        print(f"pelados/artefatos a rotular à mão: {len(res)}")
        for r in res:
            print(f"::{r['header']}:: [{r['tickers']}] n={r['n_falas']} | {r['fala_exemplo'][:70]}")
        return
    # fill
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gold_overrides import OV  # dict header->role
    filled = 0
    for r in rows:
        h = r["header"]
        lab = rule_label(h) or OV.get(h)
        if lab is None and len(h) > 80:
            lab = "manager"  # header-artefato longo = fragmento de prepared remarks (empresa)
        r["gold_role"] = lab or ""
        filled += bool(lab)
    with open(CSV, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"gold preenchido: {filled}/{len(rows)} | {dict(Counter(r['gold_role'] for r in rows))}")


if __name__ == "__main__":
    main()
