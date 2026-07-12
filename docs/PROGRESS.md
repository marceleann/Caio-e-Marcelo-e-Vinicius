# PROGRESS.md — Estado do projeto "Tone Distance"

> Resumo vivo para alinhamento do time. Atualizado ao fim de cada fase.
> Última atualização: 2026-07-08 (pivô para Angelo concluído; sinal limpo
> validado; holdout positivo).
> Regras oficiais do desafio: ver [REGRAS_DESAFIO.md](REGRAS_DESAFIO.md).

## A tese (atual, definitiva)

Transformar a *distância de tom entre os gestores* dentro de uma mesma earnings
call num sinal quantitativo testável — replicando e modernizando **Angelo et al.
(2025, *Financial Review*)**, "Tone Distance: Managerial Tone Divergence and
Market Reaction to Earnings Announcements".

- **Pergunta central:** quando os executivos de uma mesma call divergem de tom
  entre si, isso antecipa retornos futuros? O sinal é o desacordo de tom **entre
  os gestores** (não analistas versus gestão).
- **Direção (a priori, fixada pelo paper — Angelo, Tabela 6):** comprar (long)
  distância de tom **alta**. Ela cai no anúncio, mas rende mais nos ~1–3 meses
  seguintes (prêmio de risco: mais desacordo = mais incerteza = maior retorno
  exigido).

### O pivô (registrado honestamente)

O projeto começou na tese de Brockman, Li & Price (2015, FAJ) — divergência de
tom analistas×gestão — e **pivotou** para Angelo (2025) por ser mais nova, mais
inovadora e alinhada à orientadora. As decisões antigas **ADR-021** (manter só
Brockman) e **ADR-022** (direção do sinal Brockman) ficam **superadas** pelo
pivô. As fases 0–2 de infraestrutura (dados, papéis, scoring FinBERT) foram
reaproveitadas; a mudança recai sobre a construção da feature e as provas.

## O método

- **FinBERT** (`yiyanghkust/finbert-tone`) gera distribuições
  [negativo, neutro, positivo] por fala. Por gestor, agrega-se (ponderado por
  tokens) as coordenadas (probabilidade positiva, probabilidade negativa);
  a **Tone Distance** é a média das distâncias euclidianas de cada gestor até a
  média dos gestores da call (exige ≥2 gestores — cobertura 97,5% das calls,
  mediana de 4 gestores/call).
- **Nossa contribuição:** medir o tom com FinBERT no lugar do dicionário de
  palavras Loughran-McDonald usado no paper original.
- **Insight central (por que controles):** a distância de tom **crua** não prevê
  nada. O efeito só aparece com o método de Angelo — regressão com controles,
  **efeitos fixos de empresa e de trimestre** e **erros agrupados por empresa**.
  O sinal operável é o resíduo estritamente-passado da distância de tom sobre os
  confundidores [nível de tom da call, tom dos analistas, distância de tom dos
  analistas, tamanho, comprimento] — o **sinal limpo** (`tone_distance_clean`).
- **Universo:** empresas tech dos EUA no S&P 500, 2005–2025, dados 100%
  open-source (transcrições `kurry/sp500_earnings_transcripts` do HuggingFace;
  preços via yfinance).
- **Entregável final:** uma **estratégia operável** (portfólio calendar-time
  market-neutral, líquido de custos, vs. QQQ/SPY) — o event study é a evidência
  de suporte, não o produto.

## O que já está pronto

### Fase 0 — Esqueleto ✅

Repositório estruturado: `config.yaml` com todos os parâmetros comentados
(zero números mágicos no código), Makefile, requirements pinados, e
[DECISIONS.md](DECISIONS.md) com ~15 ADRs registrando cada decisão metodológica
(regra T+1 anti-look-ahead, placebo permutado dentro da data, Deflated Sharpe,
etc.) — a base da defesa perante a banca.

### Decisão conjunta sobre o universo (ADR-001)

Em vez de filtrar por setor GICS (instável: mudou em 2018 e 2023), definimos um
método híbrido: **rede programática** (setor atual via yfinance) ∪ **lista
curada** (incluindo deslistadas como Yahoo, Sun, PeopleSoft — que servem de
teste de survivorship). Núcleo "sempre dentro" = IT clássico + internet/mídia
interativa; casos debatíveis (pagamentos, Amazon/Tesla/Uber, solar-tech, EMS)
viram **blocos de sensibilidade** que rodam com e sem — a discussão "isso é
tech?" vira resultado empírico.

### Fase 1 — Dados e diagnóstico ✅

Biblioteca `src/tonediv/` implementada e testada:

- `config.py` — carregamento tipado do config (dataclasses congeladas);
- `data/transcripts.py` — parse defensivo das transcrições → tabelas `calls` e
  `utterances` + filtros de qualidade;
- `data/prices.py` — download de preços ticker a ticker com retry (deslistadas
  falham graciosamente e alimentam o survivorship);
- `data/universe.py` — a lógica do universo híbrido com regra de subtração;
- `data/diagnostics.py` — histograma de horários das calls, cobertura,
  survivorship e geração do `quality_report.md`;
- scripts `00_diagnostics.py` e `01_download_data.py` prontos.

**Qualidade:** 13 testes passando, `ruff` e `black` limpos.
**Bônus:** um teste pegou um bug real — o ticker `ON` virava booleano `True`
no YAML; corrigido e blindado (ADR-014).

### Fase 2 — Papéis e NLP ✅ (com 2 rodadas de revisão multi-agente)

`roles.py` (heurística analista×gestão com marcadores em 3 classes —
fortes/fracos/anti-aviso), `segment.py` (chunking O(n) ≤512 tokens),
`scorer.py` (FinBERT com `id2label` em runtime), scripts 02–03. Duas rodadas
de revisão adversarial multi-agente encontraram e corrigiram um **bug crítico
reproduzido** (boilerplate de abertura fazia toda a gestão virar analista) e
~10 endurecimentos; suíte com **48 testes**. Ver ADR-006 (revisões 1 e 2) e
ADR-015.

### Realidade da base (1º contato com dados reais — ADR-016)

- 33.362 calls no dataset; **99 tickers** do nosso universo presentes;
  **4.914 calls** de universo aprovadas na qualidade (361k falas).
- **Survivorship parcial**: só 5 de 37 deslistadas têm transcrição (ALTR,
  XLNX, MXIM, LSI, TWTR). Limitação declarada — será reportada no relatório.
- **Incidente yfinance**: pin 0.2.50 quebrou com a API atual do Yahoo (0
  preços/setores); atualizado para 1.5.1 e repinado.

### Fases 3, 4 e 5 — TODO O CÓDIGO PRONTO ✅ (cada uma com auditoria multi-agente)

- **Fase 3** (`features.py`, `pit.py`, script 04): 7 features + `_idio` PIT;
  regra T+1 com buffer de duração da call; guarda de timestamps completos;
  `merge_asof` estrito. Auditoria pegou vazamentos de borda (janela `_idio`,
  duração da call, match exato do ADV) — corrigidos.
- **Fase 4** (`event_study`, `calendar_portfolio`, `metrics`, `validation`,
  scripts 05–07): CAR em painel flat + placebo de datas falsas; regressão com
  efeitos fixos de empresa+trimestre e erros agrupados por empresa; portfólio
  calendar-time market-neutral; DSR/PBO/IC/t-NW; PurgedKFold + placebo
  intra-janela. Auditoria VALIDOU o núcleo numericamente e corrigiu a casca
  (exposição líquida residual, midrank no gatilho, calendário real).
- **Fase 5** (`decay.py`, `report.py`, scripts 08–09, `strategy_spec.md`):
  estabilidade pré/pós-2015 com testes de diferença (permutação + block
  bootstrap); tearsheet vs QQQ/SPY; a lâmina da estratégia.

**Suíte: 159 testes verdes** nos dois interpretadores (incl. sentinela FinBERT
real e harness de alpha sintético). `ruff`/`black` limpos.

## Estado atual — pivô concluído e sinal limpo validado ✅

O pipeline rodou nos dados reais e o pivô para Angelo está fechado. Base final:
**5.452 calls, 116 empresas, ~408 mil falas**; distância de tom em 97,5% das
calls; sinal limpo em ~63% (exige fala de analista + histórico). Resultados do
**sinal limpo** (`tone_distance_clean`):

- **Event study:** retorno anormal do tercil alto menos o do tercil baixo =
  **+1,5%**, significância forte (p ≈ 0,000); placebo de datas falsas p = 0,001.
- **Regressão controlada** (efeitos fixos empresa+trimestre, erros agrupados,
  distância padronizada): retorno de 1 mês **+0,41%/desvio-padrão** (t = 2,39;
  p = 0,017); 3 meses **+1,01%/desvio-padrão** (t = 2,50; p = 0,013). n = 3584.
  No teste **cru** (sem controles) não há nada (t < 0,4).
- **Backtest market-neutral** (sinal limpo, 3 meses) — período todo incl.
  holdout: Sharpe líquido **0,47**, drawdown máximo −7%, retorno anualizado
  +0,9%, total +16%. Long-only: Sharpe **0,94**, drawdown −6%, anualizado +3,2%.
  Referências: QQQ Sharpe 0,93; SPY 0,78.
- **Holdout (18 meses lacrados, fora da amostra):** a estratégia market-neutral
  fica **positiva** (~+0,3 em 3 meses, +0,45 em 1 mês). O sinal **cru**, ao
  contrário, **vira negativo** no holdout — limpar os confundidores dá robustez.
- **Robustez:** probabilidade de sobreajuste 0,26 (aceitável); o sinal limpo é a
  **campeã** de uma grade de 60 combinações; Sharpe deflacionado (best-de-60)
  0,18; placebo do próprio sinal p = 0,069 (marginal).
- **Verificação independente:** um agente reproduziu a regressão principal do
  zero e bateu exatamente; sem vazamento de futuro nos controles.

### Próximos passos

- [ ] **Pré-relatório 31/07** e **relatório final 17/08** — redigir a partir de
  DECISIONS/PROGRESS, com os números acima.
- [ ] **Nome/identidade do robô** (gap do regulamento, critério 1).
- [ ] **Validação manual** das 20 calls do CSV de papéis (conferência humana).

### Limitações honestas a declarar

Efeito **concentrado pós-2015** (fenômeno da era moderna, não dos 20 anos
inteiros — poder preditivo vira de negativo para positivo, p = 0,015; Sharpe de
−0,16 para +0,78, p = 0,054); placebo próprio marginal (0,069); sinal limpo cobre
~63% das calls; tamanho é proxy de liquidez (não capitalização, por falta de
dados de balanço abertos); magnitude modesta (alpha descorrelacionado de baixo
risco, não bate o índice em termos absolutos); o efeito de anúncio de
curtíssimo prazo de Angelo não é observável aqui (entramos no dia seguinte).

## Conformidade com o regulamento oficial (lido em 2026-07-04)

Os três documentos oficiais (Edital, Manual de Avaliação, Guia/FAQ) foram lidos
na íntegra e sintetizados em [REGRAS_DESAFIO.md](REGRAS_DESAFIO.md). Balanço:

**Onde o projeto já atende bem (critérios 2, 3, 4 — 55% do peso):**
- Hipótese explícita (fenômeno + justificativa + forma de teste) — exigência
  literal do critério 2 (20%); coberta pela tese e pelos ADRs.
- Modelagem com inputs/processamento/saída claros e replicáveis (critério 3,
  20%) — pipeline por etapas, config única, seeds, docstrings PT-BR.
- Backtest próprio com mitigação de vieses NOMEADA pelo edital (critério 4,
  15%): sem cherry-picking de período (amostra completa 2005–2025), guarda
  anti-look-ahead de timestamps completos, teste de survivorship, placebo
  intra-data, DSR com nº real de trials.

**Gaps identificados (a resolver):**
1. **Nome e identidade do robô (critério 1, 5%)** — ainda não temos marca/
   branding da estratégia. Decidir em equipe.
2. ~~Documentação do uso de GenAI (critério 7, 15%)~~ — **RESOLVIDO**:
   [GENAI_USAGE.md](GENAI_USAGE.md) registra etapa × contribuição (incl. as
   revisões multi-agente), com a distinção FinBERT (ML no modelo) × Claude
   (GenAI no processo). Manter atualizado a cada fase.
3. **Pré-relatório em 31/07/2026** — entra no cronograma antes do fim das
   fases; reservar tempo para redigi-lo a partir de DECISIONS/PROGRESS.
4. **Análise crítica (critérios 5 e 6, 25%)** — o relatório final não pode ser
   só métricas: regimes, cenários desfavoráveis, limitações declaradas,
   evolução realista. São nossos ativos aqui: o **pivô honesto** de Brockman
   para Angelo, o contraste **sinal cru versus sinal limpo** (o cru vira
   negativo no holdout), a **estabilidade pré/pós-2015** e as limitações
   declaradas acima.

## Como reproduzir

Ver [README.md](../README.md): `make install`, depois um alvo por etapa
(`make download`, `make diagnostics`, ...). Em Windows sem `make`, os comandos
`python scripts/XX_*.py --config config.yaml` equivalentes estão documentados.
