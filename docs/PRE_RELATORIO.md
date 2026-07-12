# Pré-relatório — Desafio Quant AI 2026 (Itaú Asset)

> **RASCUNHO** (v2, 2026-07-08). O formato oficial do pré-relatório será
> divulgado no canal do desafio; este documento organiza o conteúdo a partir da
> metodologia já implementada e será adaptado ao template oficial. Todos os
> números empíricos abaixo vêm da rodada concluída do pipeline (pontuação
> FinBERT das ~408 mil falas + regressões, event study e backtests). Entrega do
> pré-relatório: **31/07/2026**.
>
> **Nota de honestidade — a virada de tese.** Este projeto começou na tese de
> Brockman, Li & Price (2015) — divergência de tom entre analistas e gestão — e
> **pivotou** para a tese de Angelo et al. (2025), a **distância de tom entre os
> próprios gestores** ("Tone Distance"). O motivo do pivô: a tese do Angelo é
> mais recente, mais inovadora e mais alinhada à orientação recebida. As
> decisões antigas que sustentavam a versão Brockman (ADR-021 e ADR-022) ficam
> **superadas** por este pivô e estão registradas como tal em `docs/DECISIONS.md`.

## 1. Identificação

- **Robô / estratégia:** "Tone Distance" — sinal de **distância de tom entre
  gestores** dentro da mesma earnings call.
- **Equipe:** 2 integrantes.
- **Tese em uma frase:** quando os executivos de uma mesma empresa divergem no
  tom (uns mais otimistas, outros mais cautelosos) durante a mesma earnings
  call, esse **desacordo interno de tom** carrega informação sobre o retorno
  futuro da ação de tecnologia — depois de removidos os fatores que o confundem.

## 2. Resumo executivo

Construímos, com rigor institucional, um sinal quantitativo a partir de
linguagem. Medimos por FinBERT (`yiyanghkust/finbert-tone`) o tom
[negativo, neutro, positivo] de cada fala em ~5,5 mil earnings calls de empresas
de tecnologia dos EUA (2005–2025) e quantificamos a **distância de tom entre os
gestores** da mesma call: para cada gestor calculamos uma coordenada de tom e
medimos o quanto cada um se afasta da média dos gestores da call; a **Tone
Distance** é a média dessas distâncias.

O achado central é metodológico. A distância de tom **crua** não prevê nada — é
confundida por nível de tom, tamanho, setor e pelo tom dos analistas (no teste
sem controles, t < 0,4). O efeito só aparece com o método de Angelo et al.
(2025): **regressão com controles + efeitos fixos de empresa e de trimestre +
erros agrupados por empresa**. O sinal operável é o **resíduo estritamente
passado** da distância de tom sobre os confundidores — que chamamos de **"sinal
limpo"** (`tone_distance_clean`). Compramos distância de tom alta: mais
desacordo gerencial significa mais incerteza, logo maior retorno exigido — e
esse prêmio se realiza nos ~1 a 3 meses seguintes ao anúncio.

Validamos por (i) um **event study** (evidência científica), (ii) uma
**regressão controlada** e (iii) um **backtest market-neutral** líquido de
custos (o entregável), todos com guardas anti-look-ahead. Um agente independente
reproduziu a regressão principal do zero e bateu exatamente. Nossa modernização
sobre Angelo: medimos o tom com **FinBERT** no lugar do dicionário de palavras
Loughran-McDonald usado no paper. Todo o pipeline é reprodutível, versionado e
testado (159 testes automatizados); todas as decisões estão registradas em ADRs.

## 3. Conceito da estratégia — hipótese de pesquisa (critério 2)

- **Fenômeno capturado:** o **desacordo de tom entre os executivos** de uma
  mesma empresa na mesma earnings call. Quando um diretor soa confiante e outro
  soa cauteloso na mesma conversa, essa dispersão de tom sinaliza incerteza
  sobre o rumo do negócio que o preço ainda não incorporou por completo.
- **Justificativa econômica e referência exata.** A tese é de **Angelo et al.
  (2025, *Financial Review*), "Tone Distance: Managerial Tone Divergence and
  Market Reaction to Earnings Announcements"**. O mecanismo é um **prêmio de
  risco por desacordo**: mais divergência interna de tom = mais incerteza
  percebida = maior retorno exigido pelos investidores. Por isso a ação tende a
  **cair no anúncio** (o mercado desconta a incerteza) e a **render mais nos ~1
  a 3 meses seguintes** (o retorno exigido se materializa). A direção do sinal —
  **comprar distância de tom alta** — foi fixada *a priori* pelo paper (Angelo,
  Tabela 6), não ajustada aos nossos dados.

  Distinção importante — três conceitos de "tom" na literatura, para não
  confundir:
  1. **Tom da gestão sozinha → retorno** (Price, Doran, Peterson & Bliss 2012).
  2. **Diferença de tom analistas × gestão → retorno** (Brockman, Li & Price
     2015). **Era a nossa tese original**; foi de onde partimos e de onde
     pivotamos.
  3. **Distância de tom ENTRE os gestores na mesma call ("Tone Distance")** —
     Angelo et al. (2025). **É a nossa tese atual e definitiva.**

- **Nossa contribuição / modernização.** Angelo mede o tom com o dicionário de
  palavras Loughran-McDonald (contagem de termos). Nós medimos com **FinBERT**,
  um classificador contextual que lê a frase inteira e devolve a distribuição
  [negativo, neutro, positivo] — capturando negação, ironia e contexto que a
  contagem de palavras perde. Mantemos a arquitetura de identificação do efeito
  do Angelo (controles + efeitos fixos + erros agrupados) e a testamos em
  amostra própria (tecnologia dos EUA, 2005–2025).
- **Como é testado (definido ANTES do holdout):** feature-gatilho
  `tone_distance_clean` (resíduo estritamente passado); entrada no pregão
  seguinte ao anúncio; horizontes de 1 e 3 meses; validado por event study +
  regressão controlada + backtest, com testes de significância e placebos.

## 4. Dados

- **Transcrições:** `kurry/sp500_earnings_transcripts` (HuggingFace), com
  segmentação por orador. Aberto.
- **Preços:** `yfinance` (ações + índice de mercado; `QQQ` e `SPY` como
  benchmarks). 100% aberto.
- **Universo:** tecnologia dos EUA por **lista curada** (não por classificação
  setorial GICS, instável no período). Resultado real: **116 empresas**,
  **5.452 calls** aprovadas na qualidade, **~408 mil falas**, 2005–2025.
- **Cobertura do sinal.** A Tone Distance exige **≥ 2 gestores** na call
  (mediana de **4 gestores/call**) e está disponível em **97,5% das calls**. O
  **sinal limpo** (`tone_distance_clean`), que exige também fala de analista +
  histórico point-in-time para o resíduo, cobre **~63% das calls**.
- **Survivorship (limitação declarada):** o dataset traz poucas empresas
  deslistadas; o viés é reportado, não escondido, e afeta mais o *nível* de
  retorno do que o sinal *relativo* no corte transversal.

## 5. Modelagem (critério 3)

Pipeline por etapas, cada entrada/saída documentada:

1. **Tom por fala (FinBERT).** `yiyanghkust/finbert-tone` devolve, para cada
   fala, a distribuição [negativo, neutro, positivo]. Mapeamento de labels lido
   em RUNTIME (evita inversão silenciosa positivo↔negativo); chunking por
   sentença ≤ 512 tokens; agregação ponderada por tokens.
2. **Coordenada de tom por gestor.** Para cada gestor da call, agregam-se
   (ponderadas por tokens) as coordenadas (probabilidade positiva, probabilidade
   negativa) de suas falas.
3. **Tone Distance.** Distância euclidiana de cada gestor até a **média dos
   gestores da call**; a Tone Distance da call é a **média dessas distâncias**.
   Exige ≥ 2 gestores.
4. **Sinal limpo (`tone_distance_clean`) — o coração do método.** A distância de
   tom crua é regredida sobre os **confundidores estritamente passados** — nível
   de tom da call, tom dos analistas, distância de tom dos analistas, tamanho e
   comprimento da call — e ficamos com o **resíduo**. É esse resíduo, e não a
   distância crua, que é o sinal operável.
5. **Regressão de identificação do efeito (método Angelo).** Retorno futuro
   contra a distância de tom padronizada, com **controles + efeitos fixos de
   empresa e de trimestre + erros agrupados por empresa**.
6. **Alinhamento point-in-time.** Entramos no pregão seguinte ao anúncio;
   guarda forte `assert_no_lookahead` compara timestamps completos; o resíduo
   usa apenas informação anterior à decisão.

## 6. Backtest e mitigação de vieses (critério 4)

- **Estratégia:** portfólio **market-neutral** (long-short com exposição líquida
  residual reportada) e, como comparação, uma variante **long-only**. Compra-se
  distância de tom **alta** (sinal limpo), horizonte de referência de **3
  meses**. Pesos iguais com teto por nome; filtro de liquidez point-in-time;
  saída mecânica.
- **Custos:** aplicados por perna sobre o turnover.
- **Anti-vieses (nomeados pelo edital):** amostra completa 2005–2025 (sem
  cherry-picking de período); guarda anti-look-ahead de timestamps completos;
  resíduo estritamente passado (sem vazamento de futuro nos controles —
  verificado por agente independente); **placebo de datas falsas** no event
  study; **placebo do próprio sinal**; **Sharpe deflacionado** (best-de-60 numa
  grade de 60 combinações); **probabilidade de sobreajuste (PBO)**; **holdout
  temporal de 18 meses** lacrado durante todo o desenvolvimento.
- **Validação independente.** Um agente reproduziu a regressão principal do zero
  e bateu **exatamente**; confirmou ausência de vazamento de futuro nos
  controles.

## 7. Resultados preliminares (critério 5)

Todos os números abaixo referem-se ao **sinal limpo** (`tone_distance_clean`),
n = 3.584 observações com sinal limpo definido.

**Event study.** Retorno anormal acumulado do **tercil alto menos o tercil
baixo** = **+1,5%**, significância forte (p ≈ 0,000); placebo de datas falsas
p = 0,001.

**Regressão controlada** (efeitos fixos de empresa + trimestre, erros agrupados,
distância de tom padronizada):

| Horizonte | Coeficiente por desvio-padrão | t | p |
|---|---|---|---|
| Retorno de **1 mês** | **+0,39%** | 2,62 | 0,009 |
| Retorno de **3 meses** | **+0,76%** | 2,23 | 0,026 |

Variáveis winsorizadas nos percentis 5 e 95, como o Angelo (Seção 4.1); sem
winsorizar, o efeito é semelhante e também significativo (1 mês t = 2,35; 3 meses
t = 2,56). No teste **cru** (sem controles) não há nada: **t < 0,4** — a evidência
de que o efeito vive no resíduo, não na distância bruta.

**Backtest — período completo (inclui o holdout):**

| Série | Sharpe líq. | Retorno anualizado | Retorno total | Max DD |
|---|---|---|---|---|
| **Long-short (market-neutral)** | **0,47** | +0,9% | +16% | −7% |
| **Long-only** | **0,94** | +3,2% | — | −6% |
| QQQ (buy-and-hold) | 0,93 | 19,5% | — | — |
| SPY (buy-and-hold) | 0,78 | — | — | — |

O long-short é **de fato neutro** (exposição líquida ~3%). Não bate o índice em
retorno absoluto — nem pretende: é **alpha descorrelacionado de baixo risco**.

**Bloco de desenvolvimento (sinal limpo, horizonte 3 meses).** Sharpe líquido
**0,49**; exposição líquida ~3%; **bate os 20 placebos aleatórios** (melhor
acaso 0,36).

## 8. Robustez, holdout e estabilidade no tempo (critério 5)

- **Holdout (18 meses lacrados, fora da amostra).** A estratégia market-neutral
  fica **positiva** no holdout (~+0,3 em 3 meses; +0,45 em 1 mês). O sinal
  **cru**, ao contrário, **vira negativo** no holdout — evidência direta de que
  limpar os confundidores é o que dá robustez fora da amostra.
- **Robustez.** Probabilidade de sobreajuste **0,26** (aceitável); placebo do
  próprio sinal p = **0,069** (marginal, declarado); o sinal limpo é a
  **campeã** de uma grade de **60 combinações**; Sharpe deflacionado
  (best-de-60) **0,18**.
- **Estabilidade no tempo (pré vs. pós-2015).** O efeito é **concentrado no
  período recente**: o poder preditivo vira de negativo para positivo (mudança
  significativa, p = 0,015); o Sharpe passa de **−0,16 para +0,78** (p = 0,054).
  **Limitação declarada:** é um fenômeno da era moderna, não dos 20 anos
  inteiros.

## 9. Uso de IA Generativa (critério 7)

O projeto foi desenvolvido em pair-programming com **Claude Code** (registro
detalhado em `docs/GENAI_USAGE.md`): estruturação da hipótese, arquitetura,
implementação, **revisões adversariais multi-agente** que encontraram e
corrigiram bugs reais em cada fase, **reprodução independente da regressão
principal**, análise do regulamento e documentação. **Distinção importante:** o
FinBERT é um classificador (NLP/ML no núcleo do modelo, permitido; ML é
opcional) e **não** é IA generativa — a exigência de GenAI é cumprida pelo uso
no **PROCESSO**, não pelo FinBERT.

## 10. Limitações declaradas

- Efeito **concentrado pós-2015** — não vale para os 20 anos inteiros.
- Placebo do próprio sinal **marginal** (p = 0,069).
- Sinal limpo cobre **~63% das calls** (o resíduo exige analista + histórico).
- **Tamanho** é proxy de liquidez (não de capitalização), por falta de dados de
  balanço abertos.
- **Magnitude modesta:** Sharpe ~0,47 no market-neutral; não bate o índice em
  termos absolutos — o valor é ser **alpha descorrelacionado de baixo risco**.
- O **efeito de anúncio de curtíssimo prazo** do Angelo não é observável aqui:
  entramos no dia seguinte ao anúncio.
- **Survivorship parcial** do dataset; **exposição líquida residual** (~3%) do
  long-short; custos e horário assumidos. Nenhuma limitação escondida.

## 11. Conclusão e próximos passos (critério 6)

A tese está validada em três frentes independentes e consistentes — event study
(+1,5% alto−baixo, p ≈ 0,000), regressão controlada (+0,39%/1 mês e +0,76%/3
meses por desvio-padrão, significativos) e backtest market-neutral (Sharpe
líquido 0,47, positivo também no holdout) — sempre com o **sinal limpo**, e com
reprodução independente da regressão. O contraste com o sinal **cru** (que não
prevê nada e vira negativo no holdout) é a nossa evidência mais forte de que o
método de controles + efeitos fixos + erros agrupados é o que faz o sinal
existir.

**Próximos passos:** aprofundar a análise de regimes (por que pós-2015),
estressar cenários desfavoráveis, refinar a estimativa de capacidade com ADV
real por nome, e consolidar a lâmina final da estratégia
(`docs/strategy_spec.md`). Reprodutível de ponta a ponta pelo README, com
`config.yaml` como fonte única de verdade.

### Mapeamento aos critérios do edital

| # | Critério | Peso | Onde neste documento |
|---|---|---|---|
| 1 | Robô (nome e identidade) | 5% | §1 (Tone Distance) |
| 2 | Conceito (criatividade/inovação) | 20% | §3 (tese Angelo + FinBERT) |
| 3 | Modelagem | 20% | §5 (pipeline, sinal limpo) |
| 4 | Backtest (rigor e vieses) | 15% | §6 (anti-vieses, holdout, reprodução) |
| 5 | Análise dos resultados | 15% | §7–§8 (retorno, risco, estabilidade) |
| 6 | Conclusão e próximos passos | 10% | §11 |
| 7 | Uso de IA generativa | 15% | §9 |
