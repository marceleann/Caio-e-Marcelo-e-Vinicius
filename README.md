# Tone Distance

**Distância de tom entre os gestores dentro da mesma earnings call como sinal
quantitativo.**
Desafio Quant AI — Itaú Asset 2026 · 100% open-source.

---

## ⭐ Estado atual (agosto/2026) — entrega do pré-relatório

**O relatório oficial da equipe está em
[docs/RELATORIO_PRE.md](docs/RELATORIO_PRE.md)** (também em
[PDF](docs/RELATORIO_PRE.pdf) e [DOCX](docs/RELATORIO_PRE.docx)). O robô
chama-se **CORO**: ele mede a divergência de tom entre os gestores da mesma
earnings call (Tone Distance, TD) em 33.362 calls do S&P 500 (2005–2025,
685 empresas), com dois medidores de tom independentes (dicionário
Loughran-McDonald e FinBERT).

Resultados centrais, todos reproduzíveis a partir deste repositório:

- **H1 (precificação)**: o CAR do anúncio cai com a TD ponderada pelo volume
  de fala (t até −2,8; mesmo quadro no FinBERT); a versão igual-ponderada é
  nula — a ponderação é a informação;
- **Retornos subsequentes**: coeficiente positivo no painel mensal (t≈+2,0);
- **H3 (incerteza operacional)**: a TD prevê a magnitude da surpresa de
  lucro do trimestre seguinte (t=+3,58 fora da amostra, o resultado
  preditivo mais forte do projeto);
- **H2 (risco)**: descartada como sinal após a validação temporal — a
  associação em amostra cheia vem do eco do próprio anúncio e nenhuma das
  16 especificações predefinidas prevê prospectivamente;
- **Carteira executada**: top-10 por TD, rebalanceamento mensal, **+20,4%
  a.a. bruto contra +13,2% do S&P 500** (fev/2009–ago/2025), com a leitura
  honesta contra o universo equal-weight no relatório.

### Como replicar (sem reprocessar nada)

Todos os dados derivados estão versionados no repo. Basta:

```bash
git clone https://github.com/marceleann/Caio-e-Marcelo-e-Vinicius.git
cd Caio-e-Marcelo-e-Vinicius
pip install pandas numpy statsmodels pyarrow          # o suficiente p/ análises
python scripts/sp500_finbert_suite.py   # H1 nos dois medidores (tabela central)
python scripts/sp500_table6.py          # retornos mensais subsequentes
python scripts/sp500_port10.py          # carteira executada top-10 (estratégia)
python scripts/sp500_backtest.py        # long-short por quintis
```

Cada script imprime as regressões e carteiras com n, coeficiente, t e p — os
mesmos números do relatório. A bateria completa de robustez segue o mesmo
padrão, um comando cada: `sp500_oos_conditional.py` (previsão condicional
fora da amostra), `sp500_h2_reconcile.py` e `sp500_h2_final.py` (dissecação
da H2), `sp500_walkforward.py` / `sp500_walkforward_q.py` /
`sp500_t6_walkforward.py` (estabilidade temporal) e `sp500_paper_suite.py`
(Tabelas 3/4/5 na especificação estrita do paper). Para refazer **do zero**
(texto → TD → CAR → fundamentos), fontes e dependências completas estão em
[data/README.md](data/README.md) — o mapa arquivo → script → fonte primária;
as transcrições baixam sozinhas do HuggingFace na primeira execução. A
revisão de fidelidade tabela a tabela contra o paper (registro de julho)
está em [docs/REVISAO_ANGELO_TABELAS.md](docs/REVISAO_ANGELO_TABELAS.md);
em caso de divergência, vale o relatório.

---

## Estudo-piloto original (universo tech) — histórico

## A tese

O desacordo de tom **entre os próprios gestores** que falam numa mesma earnings
call contém informação sobre retornos futuros? Replicamos e modernizamos Angelo
et al. (2025, *Financial Review*), "Tone Distance: Managerial Tone Divergence and
Market Reaction to Earnings Announcements". O sinal é a **distância de tom entre
os executivos** dentro da mesma call (não analistas versus gestão).

- **Método:** FinBERT (`yiyanghkust/finbert-tone`) gera distribuições
  `[negativo, neutro, positivo]` por fala. Por gestor, agrega-se (ponderado por
  tokens) as coordenadas (probabilidade positiva, probabilidade negativa);
  calcula-se a distância euclidiana de cada gestor até a média dos gestores da
  call; a **Tone Distance** é a média dessas distâncias (exige ≥2 gestores).
- **Nossa contribuição:** medir o tom com **FinBERT** no lugar do dicionário de
  palavras Loughran-McDonald usado no paper original — modernização do sensor de
  tom.
- **Insight central:** a distância de tom **crua** não prevê nada (é confundida
  por nível de tom, tamanho, setor, tom dos analistas). O efeito só aparece com o
  método de Angelo: regressão com controles, **efeitos fixos de empresa e de
  trimestre** e **erros agrupados por empresa**. O sinal operável é o resíduo
  estritamente-passado da distância de tom sobre os confundidores — o **sinal
  limpo** (`tone_distance_clean`).
- **Direção (a priori, fixada pelo paper):** comprar (long) distância de tom
  **alta** — mais desacordo = mais incerteza = maior retorno exigido no horizonte
  de ~1–3 meses.
- **Universo:** empresas de tecnologia dos EUA, por **lista explícita** (não por
  setor GICS, instável no período). Ver [docs/DECISIONS.md](docs/DECISIONS.md) ADR-001.
- **Entregável principal:** uma **estratégia operável** (portfólio calendar-time,
  market-neutral), líquida de custos, comparada a QQQ/SPY. O event study é a
  evidência de suporte.

> **Histórico honesto:** o projeto começou na tese de Brockman, Li & Price
> (2015) — divergência de tom analistas×gestão — e **pivotou** para Angelo (2025)
> por ser mais nova, mais inovadora e alinhada à orientadora. As decisões antigas
> ADR-021 (manter só Brockman) e ADR-022 (direção do sinal Brockman) ficam
> **superadas** pelo pivô.

---

## Arquitetura do repositório

```
config.yaml            # TODOS os parâmetros, comentados (fonte única de verdade)
requirements.txt       # versões pinadas
pyproject.toml         # config de ruff/black/pytest + metadados do pacote
Makefile               # alvos de qualidade + um por etapa do pipeline
docs/                  # DECISIONS, data_schemas, quality_report, strategy_spec
src/tonediv/           # biblioteca PURA e testável (sem I/O implícito)
  config.py            # carregamento tipado do config
  data/                # transcripts, prices, universe
  nlp/                 # roles, segment, scorer (FinBERT), features
  align/               # pit (point-in-time + guarda anti-look-ahead)
  backtest/            # event_study, calendar_portfolio, metrics, validation, costs, decay
scripts/               # orquestração fina (00–09): carrega config, chama lib, salva parquet
tests/                 # unitários, vazamento, labels, harness de alpha sintético
data/                  # raw / interim / processed / outputs (gerado; não versionado)
```

Separação estrita: `src/` é biblioteca testável; `scripts/` é orquestração fina
(< 60 linhas cada). Nenhum número mágico fora do `config.yaml`.

---

## Setup

Requer **Python 3.11** (as versões pinadas em `requirements.txt` foram testadas
nele; 3.12+ pode não ter wheels para todas as libs pesadas — torch/transformers).

Ambiente canônico: **venv `.venv311` na raiz do projeto**.

```powershell
# Windows (PowerShell) — criar o venv e instalar tudo:
py -3.11 -m venv .venv311
.venv311\Scripts\python -m pip install -r requirements.txt
.venv311\Scripts\python -m pip install -e .

# Rodar qualquer etapa com o python do venv:
.venv311\Scripts\python scripts\01_download_data.py --config config.yaml
```

Com `make` (Git Bash/WSL): `make install PYTHON=.venv311/Scripts/python` e os
demais alvos idem (`make download PYTHON=...`).

---

## Como rodar o pipeline

Cada etapa é um alvo do Makefile e um script em `scripts/`. Rode em ordem.
Tempos são **estimativas** (a preencher/afinar conforme as fases são construídas;
dependem de CPU/GPU e da presença de FinBERT em GPU).

| Etapa | Alvo | Script | O que faz | Tempo estimado |
|------:|------|--------|-----------|----------------|
| 00 | `make diagnostics` | `00_diagnostics.py` | Horários das calls, cobertura por ano, qualidade e **teste de survivorship** | ~min |
| 01 | `make download`    | `01_download_data.py` | Transcrições (HF) + preços (yfinance) | ~10–30 min (rede) |
| 02 | `make roles`       | `02_infer_roles.py` | Papéis analista×gestão + CSV de validação manual | ~min |
| 03 | `make score`       | `03_score_tone.py` | FinBERT: distribuições de tom por fala | **horas em CPU / ~min em GPU** |
| 04 | `make features`    | `04_build_features.py` | As 7 features (incl. `_idio` PIT), alinhadas a preço | ~min |
| 05 | `make event-study` | `05_event_study.py` | CAR (market model) + placebo de datas falsas | ~min |
| 06 | `make backtest`    | `06_backtest_portfolio.py` | Portfólio calendar-time (long-short e long-only) | ~min |
| 07 | `make robustness`  | `07_robustness_grid.py` | Grade completa + DSR com `n_trials` real | ~min–h |
| 08 | `make decay`       | `08_decay_analysis.py` | Estabilidade no tempo (pré/pós-2015) | ~min |
| 09 | `make report`      | `09_strategy_report.py` | Tearsheet: curva de capital vs. QQQ/SPY, tabelas, drawdown | ~min |

`make all` roda 01→09. `make lint` e `make test` verificam qualidade e correção.

---

## Qualidade e reprodutibilidade

- `make lint` — `ruff` + `black --check`, limpos.
- `make test` — `pytest`: unitários (casos calculados à mão), testes de
  vazamento (inclui o caso "call às 19h → decisão no mesmo dia = ERRO") e o
  harness que recupera alpha sintético e NÃO inventa alpha em ruído.
- Determinismo: toda aleatoriedade deriva de `seed` no `config.yaml`.
- Decisões metodológicas: [docs/DECISIONS.md](docs/DECISIONS.md).

---

## Status de construção (por fase)

- [x] **Fase 0** — Esqueleto: estrutura, `config.yaml`, Makefile, requirements, DECISIONS.
- [x] **Fase 1** — Dados e diagnóstico (`config`, `transcripts`, `prices`, `universe`, `diagnostics`, scripts 00–01, testes offline).
- [x] **Fase 2** — Papéis e NLP (`roles`, `segment`, `scorer`, scripts 02–03, testes; revisão multi-agente).
- [x] **Fase 3** — Features e alinhamento (`features`, `pit`, script 04; auditoria multi-agente).
- [x] **Fase 4** — Provas (`event_study`, `calendar_portfolio`, `metrics`, `validation`, scripts 05–07; auditoria).
- [x] **Fase 5** — Decay e fechamento (`decay`, `report`, scripts 08–09, `strategy_spec.md`).

**Todo o código está pronto — 159 testes verdes.** O pipeline já rodou nos dados
reais; o pivô para Angelo (sinal de distância de tom entre gestores) está
concluído, o sinal limpo validado (event study, regressão controlada e backtest)
e o holdout de 18 meses fora da amostra ficou positivo.
