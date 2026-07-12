# strategy_spec.md — A lâmina da estratégia "Tone Distance"

> Documento legível por um gestor (PM) que nunca viu o código. Descreve a
> estratégia como um produto: tese, regras 100% mecânicas, universo, custos,
> capacidade e riscos. A METODOLOGIA é definitiva; os números de DESEMPENHO
> são os da rodada real sobre os dados abertos (transcrições + preços,
> 2005–2025) e derivam todos de `config.yaml` (fonte única de verdade).

## 1. Tese em uma frase

Em uma earnings call, quando os **gestores discordam de tom entre si** — uns
soam mais otimistas, outros mais cautelosos sobre a mesma empresa no mesmo
trimestre —, essa **distância de tom entre os executivos** carrega informação
sobre o retorno futuro da ação: compramos onde o desacordo gerencial é alto,
porque mais desacordo significa mais incerteza e, portanto, maior retorno
exigido nos meses seguintes.

- **Fenômeno capturado:** discordância de tom **entre os gestores** dentro da
  MESMA call (não gestão versus analistas). É o "Tone Distance" de **Angelo,
  Reck, Wu & Zhu (2025, Financial Review), "Tone Distance: Managerial Tone
  Divergence and Market Reaction to Earnings Announcements"**.
- **Por que existiria:** quando os executivos de uma empresa transmitem sinais
  de tom conflitantes, o mercado enfrenta maior incerteza sobre o valor
  fundamental. O preço reage no anúncio, mas o **prêmio de risco** por essa
  incerteza é pago ao longo dos ~1–3 meses seguintes — mais desacordo, maior
  retorno exigido.
- **Nossa contribuição (modernização):** medimos o tom de cada fala com
  **FinBERT** (`yiyanghkust/finbert-tone`, modelo contextual) no lugar do
  dicionário de palavras Loughran-McDonald que Angelo usa. A tese e a direção
  do sinal são do paper; a medição do tom é nossa atualização de método.
- **Como é testado:** estudo de evento (retorno anormal acumulado por tercil) +
  backtest calendar-time (o entregável), ambos com controles, guardas
  anti-look-ahead e testes de significância.

## 2. Sinal

- **Métrica bruta:** `tone_distance` — dentro de cada call, o FinBERT dá a cada
  fala uma distribuição de tom [negativo, neutro, positivo]. Por gestor,
  agregam-se (ponderado por tokens) as coordenadas (probabilidade positiva,
  probabilidade negativa). Calcula-se a **distância euclidiana** de cada gestor
  até a média dos gestores da call; a Tone Distance é a **média** dessas
  distâncias. Exige **≥ 2 gestores** (cada um com ≥ 25 tokens pontuados) — do
  contrário a call não tem sinal. Cobertura: **97,5% das calls** (mediana de 4
  gestores por call).

- **Insight central — por que o sinal precisa ser LIMPO:** a distância de tom
  **crua não prevê nada**. Ela é confundida por nível de tom da call, tom dos
  analistas, tamanho e comprimento. No teste sem controles, o efeito é nulo
  (estatística t < 0,4). O efeito só aparece com o método do Angelo:
  **controlar os confundidores + efeitos fixos de empresa e de trimestre +
  erros agrupados por empresa**.

- **Sinal operável (`tone_distance_clean`):** o **resíduo estritamente-passado**
  da distância de tom sobre os confundidores — regredimos a distância de tom
  contra `[nível de tom da call, tom dos analistas, distância de tom dos
  analistas, tamanho, comprimento]` usando **apenas dados anteriores à
  decisão** (residualização point-in-time). O que sobra — a parcela de
  desacordo gerencial não explicada pelos confundidores — é o sinal. É essa
  limpeza que dá robustez fora da amostra (§4).

- **Direção do gatilho:** percentil do sinal limpo do evento contra os eventos
  dos **90 dias corridos estritamente anteriores** (point-in-time; ~um
  trimestre de calls). Percentil **alto → comprar** (distância de tom alta);
  baixo → vender. Direção **fixada a priori pelo paper** (Angelo, Tabela 6),
  não escolhida por nós para maximizar retorno. Empates resolvidos por midrank
  (sinal sem informação → sem posição).

- **Cobertura do sinal limpo:** ~**63% das calls** — exige, além dos ≥ 2
  gestores, falas de analista na call (para os confundidores) e histórico
  passado suficiente para a residualização.

## 3. Regras da carteira (100% mecânicas — nenhuma decisão discricionária)

| Regra | Especificação |
|---|---|
| **Universo elegível** | Tech dos EUA (lista explícita em camadas + rede de setor; ver ADR-001), com filtro de liquidez point-in-time. |
| **Liquidez** | Volume médio diário em dólar (ADV) dos últimos 60 pregões ≥ US$ 5 MM, medido estritamente antes da decisão. |
| **Sinal-gatilho** | `tone_distance_clean` — resíduo estritamente-passado da distância de tom entre gestores (FinBERT). Comprar percentil alto (`top_fraction = 0,2`). |
| **Entrada** | No **open** do pregão de decisão (regra T+1: call que não termina antes das 09:30 ET decide no dia seguinte; ver ADR-002). |
| **Holding** | **H = 63 pregões (~3 meses)** — o horizonte do prêmio pós-anúncio da tese (Angelo, Tabela 6). Saída mecânica no close de d_{H−1}. |
| **Sobreposição** | Tranches estilo Jegadeesh-Titman: a carteira de cada dia é a média das H tranches abertas nos últimos H pregões. |
| **Pesos** | Iguais dentro do cohort, teto de **10%** por nome (excesso vira caixa). |
| **Variantes** | (i) **long-short market-neutral** (exposição líquida residual reportada e também hedgeada); (ii) **long-only vs. benchmark**. |
| **Caixa** | Capital não alocado rende **0%** (conservador). |
| **Custos** | **5 bps por perna** sobre o turnover diário (entradas + desmontes ao notional corrente). |

## 4. Como lemos o resultado

Comparações obrigatórias, todas líquidas de custos:

- **vs. QQQ e SPY** buy-and-hold no mesmo período (benchmarks sem custo — o pior
  caso para nós);
- **vs. placebo de sinal aleatório** (20 permutações do sinal — sanity check);
- **vs. o sinal CRU** (sem controles): serve de contraprova do valor da limpeza;
- **Significância:** t-stat Newey-West das séries; regressão controlada (efeitos
  fixos de empresa+trimestre, erros agrupados); placebo de datas falsas no
  estudo de evento; Deflated Sharpe (Bailey-LdP com nº real de trials); PBO via
  CSCV; holdout temporal de 18 meses intocado durante o desenvolvimento.

## 5. Desempenho (rodada real, sinal limpo, horizonte de 3 meses)

**Evidência estatística.**

- **Estudo de evento:** retorno anormal acumulado do tercil de distância alta
  menos o do tercil baixo = **+1,5%**, significância forte (p ≈ 0,000); placebo
  de datas falsas p = 0,001.
- **Regressão controlada** (efeitos fixos de empresa+trimestre, erros agrupados,
  distância de tom padronizada): retorno de 1 mês **+0,41% por desvio-padrão**
  (t = 2,39; p = 0,017); retorno de 3 meses **+1,01% por desvio-padrão**
  (t = 2,50; p = 0,013). n = 3.584. No teste CRU, nada (t < 0,4).
- Uma verificação independente reproduziu a regressão principal do zero e bateu
  exatamente; sem vazamento de futuro nos controles.

**Backtest — período completo, incluindo o holdout, líquido de custos.**

| Série | Sharpe líq. | Retorno a.a. | Max DD | Retorno total |
|---|---|---|---|---|
| **Long-short (market-neutral)** | **0,47** | +0,9% | **−7%** | +16% |
| Long-only | 0,94 | +3,2% | −6% | — |
| QQQ (buy-and-hold) | 0,93 | +19,5% | — | — |
| SPY (buy-and-hold) | 0,78 | — | — | — |

- A exposição líquida do long-short é ~**3%** (de fato neutra).
- **Holdout (18 meses lacrados, fora da amostra):** a estratégia
  market-neutral fica **positiva** (~+0,3 em 3 meses; +0,45 em 1 mês). O sinal
  **cru**, ao contrário, **vira negativo** no holdout — a limpeza dos
  confundidores é o que dá robustez.
- **Bloco de desenvolvimento** (antes do holdout): Sharpe líquido 0,49; o sinal
  limpo **bate os 20 placebos aleatórios** (melhor acaso 0,36).

**Robustez.** Probabilidade de sobreajuste (PBO) 0,26 (aceitável). Sinal limpo é
a **campeã de uma grade de 60 combinações** (feature × horizonte); Sharpe
deflacionado (best-de-60) 0,18. Placebo do próprio sinal p = 0,069 (marginal).

**Estabilidade no tempo (pré vs. pós-2015).** O efeito é **concentrado no
período recente**: o poder preditivo vira de negativo para positivo (mudança
significativa, p = 0,015); Sharpe de −0,16 para +0,78 (p = 0,054). É um
fenômeno da era moderna, não dos 20 anos inteiros — declarado como limitação
(§7).

## 6. Capacidade estimada

Piso de liquidez de US$ 5 MM/dia de ADV e teto de 10% por nome limitam o
tamanho negociável. Universo de 116 empresas de tecnologia dos EUA; a carteira
market-neutral opera cerca de 40% do universo elegível por lado a cada momento
(`top_fraction = 0,2` em long e short). O holding de 3 meses mantém o turnover
baixo, o que ajuda a capacidade. **Limitação declarada:** não modelamos impacto
de mercado nem custos fixos; os 5 bps/perna são proporcionais.

## 7. Riscos conhecidos (declarados, não escondidos)

1. **Efeito concentrado pós-2015.** O poder preditivo é um fenômeno da era
   moderna (vira de negativo para positivo, p = 0,015); nos 20 anos inteiros o
   sinal é mais fraco. É a nossa limitação mais material.
2. **Cobertura do sinal limpo (~63% das calls).** A residualização exige falas
   de analista e histórico passado; calls sem esses insumos ficam de fora — o
   sinal opera em um subconjunto, não em todas as calls.
3. **Magnitude modesta.** Sharpe ~0,47 na versão market-neutral — não bate o
   índice em termos absolutos (QQQ ~0,93). O valor é ser **alpha
   descorrelacionado de baixo risco** (drawdown −7%, exposição líquida ~3%), não
   retorno absoluto.
4. **Placebo do próprio sinal marginal (p = 0,069)** e Sharpe deflacionado
   modesto (0,18): sinal genuíno, porém não folgado — a ser lido com o event
   study e a regressão controlada em conjunto, não isoladamente.
5. **Tamanho é proxy de liquidez, não capitalização** (ADV, por falta de dados
   de balanço abertos); é um controle aproximado do confundidor de tamanho.
6. **Efeito de curtíssimo prazo não observável.** O impacto de anúncio do dia do
   evento (Angelo) não é capturado aqui: entramos no dia seguinte (T+1
   conservador). Operamos o prêmio dos meses seguintes, não o salto do anúncio.

## 8. Reprodutibilidade

Tudo deriva de `config.yaml` (seed fixa) e roda por `make download → … →
report`. Nenhum parâmetro fora do config; nenhuma decisão discricionária no
caminho do sinal ao P&L. 159 testes automáticos, guardas anti-look-ahead,
revisões adversariais multi-agente e reprodução independente da regressão
principal. Ver README e docs/DECISIONS.md (ADR-023 registra o pivô Brockman →
Angelo). IA generativa: desenvolvido em pair-programming com Claude Code; o
FinBERT é classificador (não generativo), então a exigência de GenAI é cumprida
pelo PROCESSO.
