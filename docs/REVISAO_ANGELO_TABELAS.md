# Revisão de fidelidade contra Angelo et al. (2025) — rodada completa

**Data:** 15-16/07/2026. **Mandato:** revisar tudo contra o PDF do paper, pesquisar
soluções, avançar autonomamente; cada saída ancorada no artigo. **Diretriz de
desenho (Marcelo):** replicação exata do modelo do Angelo; único desvio
intencional = tom por FinBERT (braço GenAI).

## 1. Auditoria — o que já estava fiel

| Item | Paper (onde) | Nosso estado |
|---|---|---|
| TD Eq.(1): frações de palavras LM por gestor, transcrição inteira, euclidiana ao centroide, média | §4.2, Apêndice A | idêntico; distribuição bate (0.0086/0.0080/0.0047 vs 0.0079/0.0074/0.0048) |
| CAR: CAPM (sem alfa), estimação 100d, mín. 70, gap 50, janelas [-1,+1],[-1,+2],[-1,+5] no dia do anúncio | §5.1 + nota T3 | `sp500_car_v3.py` — exato |
| Eq.(3): FE firma + FE ano-tri, cluster por firma | nota T3 | idêntico |
| Winsorização 5/95 em todas as variáveis (1/99 similar) | §4.1 + nota 3 | idêntico (1/99 testado) |
| TD crua como variável principal | T3 | idêntico |
| RD ausente = 0 | Apêndice A | idêntico |
| Sem filtro de palavras mínimas por gestor (todos entram) | §4.2 (T8c3 é a alternativa) | idêntico |

Correção de registro: a spec principal (T3) inclui TANTO fundamentos QUANTO
Analyst Tone/Dispersion/Industry Tone — nossa regressão sempre teve os dois
blocos; apenas a explicação anterior estava imprecisa.

## 2. Correções implementadas nesta rodada (Apêndice A, item a item)

1. **Lev** = dívida de LONGO PRAZO / ativos (tags `LongTermDebtNoncurrent`/
   `LongTermDebt`). Antes: passivo total (média 0.68 ≠ 0.235 do paper).
   Depois: **0.244/0.223 vs 0.235/0.189 — bate**.
2. **Std ROA / Std CFO** sobre 16 trimestres (antes 8; mín. 8, declarado).
3. **CFO trimestral por diferenciação de YTD** (10-Q reporta CFO acumulado):
   cobertura Std CFO 42%→77%; `filed` do trimestre derivado = max(filed) dos
   dois YTD (PIT conservador). Prática padrão XBRL/Compustat.
4. **Smooth** = Std CFO/Std ROA (faltava). Agora 2.44/1.59 vs 2.58/1.44 do paper.
5. **ETR** = imposto/(LL+imposto) — fórmula literal do Apêndice A, sempre.
6. **Standardized ME** = z-score do valor de mercado (antes só lnME).
7. **Lagged Average TD** = média dos 4 TDs anteriores da firma (antes 1 lag);
   braço estrito (exatamente 4) como sensibilidade.
8. **Analyst Tone Dispersion** = desvio-padrão do net tone entre analistas
   (Apêndice A; antes usávamos distância euclidiana) — de `speaker_counts`.

**Segue ausente e declarado:** AQ (Francis et al. 2004, exige accruals),
Institutional Ownership (13F), SUE do I/B/E/S (proxy yfinance), vol implícita
(OptionMetrics), mean range (sem OHLC), FF49 (proxy GICS), CRSP/Capital IQ
(yfinance/HF kurry). Scripts: `sp500_fund_v2.py`, `sp500_paper_suite.py`.

## 3. Resultados por tabela do paper (amostra S&P 500)

### Tabela 3 (CAR ~ TD base) — segue NULA com a spec completa
17 controles, amostra comum 2009+ (n=14.481, 395 firmas): t = −0.30/−0.27/−0.32
nas três janelas (2009-22: −0.52/−0.62/−0.77). Consistente com as 9 rodadas
anteriores. **Mas ver §4: a variante ponderada do próprio paper muda o quadro.**

### Tabela 4 (vol futura) — sinal concentrado no pós-crise
Com controles antigos, amostra ampla 2009+: vol20 t=+1.95, vol120 t=+1.87.
Diagnóstico formal (`sp500_diag_controls.py` + decomposição): a spec completa
não mata por CONTROLES, mata por AMOSTRA — a exigência de 16 tri de XBRL
elimina 2009-2010 (0-8% de cobertura). Restringindo às mesmas 395 firmas o
sinal sobrevive (t=+2.31/+2.08); cortando 2009-10 morre em qualquer conjunto
de firmas. **Leitura: em large caps, TD→risco é fenômeno de regime de
estresse.** (O paper inclui 2006-2008.)

### Tabela 5 (operacional) — |SUE t+1| idem (t=+2.21 ampla; morre sem 2009-10);
ROA vol futura marginal (t=+1.64); CFO vol futura nula.

### Tabela 6 (retorno mensal 3m pós-anúncio) — **REPLICA**
Painel firma-mês, FE firma+ano-mês, cluster ano-mês, `sp500_table6.py`:

| spec | paper | nós |
|---|---|---|
| col 2 (BTM, momentum, size, reversal), 5/95 | +0.1693 (t 2.17) | **+0.1614 (t 2.02)** |
| col 2, winsor 1/99 | "similar" (nota 3) | +0.1569 (t 2.00) |
| col 1 (só FE) | +0.2282 (t 2.81) | +0.0910 (t 1.20) |

Estável em subperíodos (2013+: +0.12; 2018+: +0.20, t 1.82). Sem winsorização
nenhuma o efeito some (nenhum painel de retorno roda assim; declarado).

### Tabelas 7 e 8 (variantes de TD) — **o achado da rodada**
`sp500_table78.py` + `sp500_tdw_deep.py`, de `speaker_counts.parquet`:

**TD ponderada por palavras (T8 col 1 do paper)**, CAR, controles antigos,
amostra ampla 2009+ (n=21.245):

| janela | paper (T8c1) | nós | IQR→CAR |
|---|---|---|---|
| CAR[-1,+1] | −0.7678 (t −4.72) | **−0.4082 (t −2.86)** | −0.15% (paper: −0.21%) |
| CAR[-1,+2] | — | −0.4324 (t −2.84) | −0.16% |
| CAR[-1,+5] | — | −0.5051 (t −3.07) | −0.19% |

Robusto a 1/99 (t −2.08/−2.10/−2.51); subperíodos 2009-16 (−0.39, t −1.75) e
2017-25 (−0.42, t −2.16); amostra 2005-2025 completa: −0.33 (t −2.40).
Na subamostra de fundamentos atenua para t≈−1.3/−1.5 (efeito de composição de
amostra, não dos controles — mesmo padrão do H2).

E o arco completo com td_w:
- **Vol futura:** vol20 t=+2.49; vol120 **t=+3.62** (paper: +3.46/+3.50);
- **Retorno mensal (T6):** col1 +0.2142 (t 2.21) vs +0.2282 (t 2.81) paper;
  col2 +0.2242 (t 2.32) vs +0.1693 (t 2.17); robusto 1/99.

**Adjusted TD (T7):** ponto −0.155 vs −0.134 do paper (sinal e magnitude
certos), t −1.55 (ns — nossos clusters são 477 firmas vs 3.860 do paper).
T8c3 (exclui < mediana): −0.18 (t −1.17), sinal certo, ns.

**Interpretação (com base no próprio paper):** a TD igual-ponderada dá o mesmo
peso a quem fala 200 palavras e a quem fala 5.000 — em calls de large caps
(muitos participantes secundários) isso injeta ruído na média. O próprio
Angelo reporta a versão ponderada como MAIS forte (−0.77 vs −0.49) e conclui
que "investors seem to place greater emphasis on the managers that speak
more". Não é fishing nosso: é a spec alternativa pré-existente do paper, e o
nosso resultado replica o padrão dele (ponderada > base).

## 4. Backtests (spec congelada, `sp500_backtest.py`)

| sinal | LS líquido | Sharpe | t |
|---|---|---|---|
| td cru | +2.0% aa | +0.18 | +0.83 |
| td_w | +0.3% aa | +0.05 | +0.23 |
| td_w vs próprio histórico (FE-consistente) | +0.6% aa | +0.07 | +0.32 |

Leitura honesta: o prêmio da T6 é identificado DENTRO da firma com controles;
quintis incondicionais não o monetizam em large caps. A informação da TD está
mais em RISCO (vol futura, t +3.6) do que em retorno explorável isolado.
Implicações de estratégia: (a) overlay de risco; (b) sinal combinado; (c)
universo small/mid (Motley Fool) — decisão com o Marcelo e a orientadora.

## 5. Braço FinBERT (único desvio intencional — diretriz do projeto)

Ambiente reconstruído (Python 3.14: pyarrow, datasets, torch 2.13 CPU,
transformers 5.14). `sp500_score_finbert.py` lançado: pontua sentença a
sentença as ~27,7k calls restantes (5.452 do universo tech reaproveitadas,
mesmo segmentador/modelo), retomável (1 parquet/call), ordem embaralhada
determinística (qualquer prefixo é representativo), sessões de 12h. Ao
completar (ou com prefixo grande), montar TD_FinBERT e rodar toda a suíte.

## 5b. Adendo (madrugada de 16/07) — FF49 e otimização do scorer

- **Industry Tone agora é FF49 exato** (`sp500_ff49.py`): SIC de cada CIK via
  EDGAR submissions (603/603) + Siccodes49 do Ken French; média leave-one-out
  por indústria-trimestre (>=3 pares). Cobertura 76%→85% (o GICS do yfinance
  tinha `sector` nulo em 23% dos eventos — era um dreno de amostra em TODAS as
  regressões). Com FF49 e n=24.953 (544 firmas), o achado central sustenta:
  td_w CAR t=−2.34/−2.35/−1.94; vol futura t=+2.30/+3.01; TD base segue nula.
- **Scorer FinBERT otimizado:** quantização dinâmica int8 (validada: argmax
  100% igual ao fp32, max|Δp|=1e-4) + fila com prioridade para as 22.246 calls
  elegíveis à regressão. Taxa 19,5→42,6 sent/s; ETA da fila elegível ~63h
  (sessões de 11-12h, retomável). `sp500_td_finbert.py` monta TD_FinBERT
  (igual-ponderada e ponderada por tokens, espelhos exatos do braço LM) e
  roda as regressões no subset pontuado — leitura interina viável já com o
  prefixo aleatório da primeira noite (~4k calls).
- Interinos FinBERT (5.419 calls tech antigas): corr(td_lm, td_fb)=+0.19,
  corr(td_w, td_fb_w)=+0.21 — correlação baixa entre as medidas de tom é
  esperada (léxico vs modelo) e é exatamente o que torna o braço FinBERT um
  teste informativo, não redundante.

## 5c. Adendo (20/07) — a ambiguidade do centroide (desconfiança do Marcelo)

O Marcelo insistiu que a TD base nula podia ser erro de mensuração. Auditoria:
(i) aplicação da fórmula EXONERADA — exemplo manual (AAPL 2020Q4) e
reimplementação independente nas 32.387 calls, diferença máxima 8e-17;
(ii) MAS a frase da Eq.(1) ("Avg.Pos = average percent positive words spoken
across all managers") admite duas leituras: A = média simples das frações por
gestor (nossa original); B = fração AGREGADA dos gestores (total pos/total
palavras) — e o rótulo da Figura 1 do paper ("Average Transcript Tone")
sugere B. Na leitura A, um orador marginal (ex.: RI lendo o disclaimer
jurídico, 283 palavras, 6 "negativas" de boilerplate — caso real da AAPL
2020Q4) desloca o centro; na B, o centro é o tom de quem efetivamente fala.

**Resultado (Eq.3, controles antigos+FF49, 2009+, n~23,6k): a leitura B
REVIVE a TD base nos dois sensores** — LM: t=-1.98/-2.05/-1.73; FinBERT:
t=-1.88/-2.29/-2.34 (IQR ~-0.11% a -0.14%). Cadeia monotônica completa:

| espec | peso da fala | t LM | t FinBERT |
|---|---|---|---|
| leitura A (centro = média simples) | nenhum | -0.72 | +0.09 |
| leitura B (centro = tom agregado) | no centro | -1.98 | -1.88 |
| T8c1 (pondera centro E distâncias) | total | -2.83 | -2.37 |

Conclusão revisada: a Eq.(1) do Angelo REPLICA em large caps sob a leitura B
do centroide (defensável pelo próprio paper), e fortalece monotonicamente com
mais peso de fala. O nulo da leitura A tem diagnóstico preciso: centro
contaminado por oradores marginais (boilerplate de RI conta como tom negativo
no LM). Scripts: `diag_equalweight_null.py`, `sp500_poolcent.py`; dados:
`td_poolcent.parquet`.

## 6. Arquivos novos desta rodada

`sp500_speaker_counts.py`, `sp500_fund_v2.py`, `sp500_paper_suite.py`,
`sp500_diag_controls.py`, `sp500_table6.py`, `sp500_table78.py`,
`sp500_tdw_deep.py`, `sp500_score_finbert.py`, `sp500_td_finbert.py`,
`sp500_ff49.py`; dados: `speaker_counts.parquet`,
`fundamentals_facts_v2.parquet`, `events_sp500_paper.parquet`,
`td_variants.parquet`, `cik_sic.parquet`, `tone_distance_finbert.parquet`.
