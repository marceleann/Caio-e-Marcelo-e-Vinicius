# Prompt para o novo chat — Projeto Tone Distance

> Cole tudo abaixo da linha como primeira mensagem do chat novo, na pasta
> `Desafio Itaú QAI 26`.

---

# Contexto

Você é meu parceiro de pesquisa quantitativa no **Desafio Itaú Asset Quant AI 2026**.
Somos um time de 2 pessoas. Eu não sou quant profissional — vou ter que **defender
cada escolha deste projeto ao vivo, diante de uma banca de gestores do Itaú Asset e
do Itaú BBA**. Isso significa que eu preciso entender de verdade o que estamos
fazendo, não apenas ter um repositório que roda.

O repositório já existe, tem dados baixados e código, mas **foi construído sobre a
tese errada** e precisa de uma reformulação conceitual. Este prompt explica o que é
a tese certa, o que já existe, o que está quebrado, e como quero trabalhar.

---

# 1. Regras de trabalho — não negociáveis

1. **Explique antes de fazer.** Antes de qualquer script, diga em português claro: o
   que vai fazer, por quê, o que espera ver, e como saberemos se deu errado. Só
   execute depois do meu "ok".

2. **Pare em checkpoints.** Divida o trabalho em etapas curtas. Ao fim de cada uma,
   pare, mostre os números que saíram, e espere minha validação. Não encadeie cinco
   etapas antes de falar comigo.

3. **Ensine a estatística.** Sempre que usar um conceito — CAR, market model,
   efeito fixo, erro-padrão agrupado, winsorização, Deflated Sharpe Ratio,
   point-in-time — explique em português simples **antes** de usar. Assuma que eu
   não sei. Eu preciso conseguir explicar isso a um gestor.

4. **Nunca invente números.** Nenhum resultado, tabela, correlação, t-stat ou
   conclusão pode entrar em documento algum sem ter saído de um script que rodou de
   verdade nos dados reais. Se não rodou, escreva "não rodado". Se um número é uma
   estimativa, rotule como estimativa. Placeholders são proibidos, mesmo temporários.

5. **Não peça permissões em lote.** Um comando por vez, com a justificativa junto.
   Se você não consegue explicar por que precisa rodar algo, não rode.

6. **Discorde de mim.** Se eu pedir algo metodologicamente errado, diga que está
   errado e por quê. Um "sim" complacente que quebra na banca vale menos que um
   "não" hoje.

7. **Português**, em todo código, comentário, commit e documento.

---

# 2. A tese

## O paper

**Angelo, B., Johnston, M., Singh, A., & Wan, Y. Q. (2025).** "Tone Distance:
Managerial Tone Divergence and Market Reaction to Earnings Announcements."
*The Financial Review*, 60(4), 1415–1435. O PDF está na pasta do projeto. **Leia-o
inteiro antes de escrever qualquer linha de código.**

## O que é Tone Distance

É a **divergência de tom entre os gestores que falam na MESMA earnings call**.
Mede o quanto os executivos de uma empresa deixam de apresentar uma frente unida.

Construção original (Equação 1 do paper):

1. Para cada **gestor** `m` que fala no transcript `t`, conte: total de palavras,
   palavras positivas, palavras negativas (dicionário Loughran-McDonald 2011).
2. `Pos(m,t)` = palavras positivas / total de palavras do gestor `m`.
   `Neg(m,t)` = palavras negativas / total de palavras do gestor `m`.
3. Cada gestor vira um ponto no plano cartesiano `(Pos, Neg)`.
4. `Avg.Pos(t)` e `Avg.Neg(t)` = média entre **todos os gestores** da call.
5. `ManagerDistance(m,t) = sqrt( (Pos(m,t) − Avg.Pos(t))² + (Neg(m,t) − Avg.Neg(t))² )`
6. **`Tone Distance(t)` = média de `ManagerDistance` entre todos os gestores da call.**

Detalhes de fidelidade que importam:

- Positivo e negativo entram como **dois eixos separados**, nunca como tom líquido.
  O paper é explícito: gestores podem divergir no eixo positivo ou no negativo, e
  isso carrega informação distinta.
- O paper usa o **transcript inteiro** (prepared remarks + Q&A), não só o Q&A.
- Comentários da **operadora** são excluídos.
- Amostra do paper: 188.466 transcripts, 7.526 empresas, 2006–2022, Capital IQ.
  Tone Distance: média 0,0079, mediana 0,0074, desvio 0,0048 (CV = 0,61).
- Winsorização em 5% e 95%.

## O que NÃO é

**Tone Distance não é a divergência entre a gestão e os analistas.** Essa é outra
tese — Brockman, Li & Price (2015) — e é o erro que contaminou o repositório atual.
No Angelo, analistas aparecem **apenas como variáveis de controle** (`Analyst Tone`
e `Analyst Tone Dispersion`), justamente porque o tom das perguntas dos analistas
pode influenciar o tom das respostas dos gestores. Nunca como sinal.

## As hipóteses do paper

- **H1**: Tone Distance maior → retornos anormais **menores** no período do evento
  (uma variação interquartil ⇒ CAR de 1 dia **0,21% menor**).
- **H2**: Tone Distance maior → **mais risco de mercado** depois da call (desvio-padrão
  dos retornos, average range, volatilidade implícita).
- **H3**: Tone Distance maior → **mais risco operacional** (volatilidade do ROA,
  volatilidade do fluxo de caixa, SUE do trimestre seguinte, Tobin's Q menor).
- E: Tone Distance prevê **positivamente** os retornos mensais nos ~3 meses seguintes
  ao anúncio — o mercado exige prêmio de risco. **Esta é a direção operacional da
  nossa estratégia: comprar distância de tom alta.** A direção é fixada *a priori*
  pelo paper, não escolhida por nós para maximizar retorno.

## A nossa tese

> **Em empresas de tecnologia dos EUA, o desacordo de tom entre os executivos dentro
> de uma mesma earnings call é informação sobre o risco futuro da empresa — e essa
> informação é precificada com atraso, gerando um prêmio nos ~3 meses seguintes.**

Nossa **contribuição sobre o Angelo** é dupla:

1. **Modernizar o sensor de tom**: trocar o dicionário Loughran-McDonald por
   **FinBERT** (`yiyanghkust/finbert-tone`), um classificador contextual que entende
   negação, ironia e contexto — coisas que um dicionário de palavras não vê.
2. **Corrigir um viés mecânico da métrica** que o paper não trata (ver seção 4).

---

# 3. O que já existe no repositório (inventário verificado)

Isto foi auditado. Confie, mas confirme rodando.

## Dados

| Arquivo | Conteúdo |
|---|---|
| `data/raw/calls_all.parquet` | Transcripts brutos, HuggingFace `kurry/sp500_earnings_transcripts` (licença MIT) |
| `data/raw/prices.parquet` | Preços via `yfinance`; proxy de mercado `^GSPC` |
| `data/interim/calls.parquet` | **5.452 calls**, **116 tickers** com calls, de **2005-11-16 a 2025-05-15** |
| `data/interim/universe.parquet` | 150 tickers listados (eu falo em 166 — **confira e reconcilie**) |
| `data/interim/utterances.parquet` | Falas segmentadas por orador |
| `data/interim/utterances_roles.parquet` | Falas + papel inferido por **heurística de regex** |
| `data/interim/utterance_scores.parquet` | **307.441 falas** pontuadas com FinBERT |

Cobertura de papéis atual (heurística): 95,4% das calls com Q&A detectado;
180.326 falas de gestão, 156.805 de analistas, 70.963 da operadora, 154 desconhecidas.

## Código

Repositório maduro: `src/tonediv/` (biblioteca pura), `scripts/00`–`11` (orquestração),
`tests/` (159 testes), `config.yaml` (fonte única de parâmetros),
`docs/DECISIONS.md` (25 ADRs). Vale a pena preservar a **infraestrutura**:
alinhamento point-in-time (`align/pit.py`), event study, portfólio calendar-time,
métricas, custos, e a suíte de testes anti-vazamento.

O que **não** vale preservar é o **núcleo conceitual**.

---

# 4. Diagnóstico — os quatro problemas

Estes números foram calculados diretamente nos parquets do repositório. Reproduza-os
como primeira tarefa de sanidade.

## P1 — O FinBERT está sendo aplicado na unidade errada

`yiyanghkust/finbert-tone` foi *fine-tuned* em **10.000 sentenças** anotadas de
relatórios financeiros (Huang, Wang & Yang, 2022, *Contemporary Accounting Research*).
O pipeline atual o alimenta com **falas inteiras** (mediana de 95 tokens, média de
1,13 chunks de 512 tokens).

Resultado: as probabilidades saturam. Nos scores atuais —

- `p_pos < 0,01` em **53,5%** das falas
- `p_pos > 0,99` em **21,4%** das falas
- apenas **16,5%** das falas caem na faixa intermediária `(0,05 ; 0,95)`

A "probabilidade" virou um voto quase binário. A coordenada do gestor deixou de ser
"quão positivo ele foi" e virou "que fração dos tokens dele caiu em falas rotuladas
como positivas".

**Isso tem conserto e o conserto é elegante.** Pontue **sentença por sentença** e
agregue por gestor ponderando por tokens. Aí `Pos(m)` vira a **fração de sentenças
positivas** do gestor — o análogo direto da **fração de palavras positivas** do
Loughran-McDonald. A ponte conceitual com o Angelo fica perfeita, e é assim que a
troca de sensor deve ser defendida.

Custo: mais forward passes (estime antes de rodar; use GPU se houver, e mantenha o
checkpointing por shards que já existe).

## P2 — A Tone Distance atual mede volume de fala, não desacordo

Este é o problema grave. Calculei a Tone Distance pelo método do Angelo em cima dos
scores existentes:

| Correlação com Tone Distance | valor |
|---|---|
| nº de tokens do gestor que **menos falou** | **−0,545** |
| nº de gestores na call | +0,157 |
| tom positivo médio da call | +0,017 |

O determinante dominante da métrica é **quão pouco o gestor mais quieto falou.**

A mecânica: um gestor que responde uma única pergunta produz uma coordenada estimada
sobre pouquíssimas sentenças. `Var(Pos) ≈ p(1−p)/n_sentenças`. Com 8 sentenças, a
variância é ~25× a de um gestor com 200. Ele cai longe da média por **ruído amostral**,
não por discordar. A call inteira é rotulada como "alto desacordo".

Confirmação: a Tone Distance média sobe de **0,224** (2 gestores) para **0,307**
(3 gestores) e **0,314** (4 gestores) — monotônica no número de falantes.

E `config.yaml` hoje admite um gestor na conta com apenas **25 tokens**.

Isso também explica por que a dispersão não bate com o paper:

| | média | desvio | CV |
|---|---|---|---|
| Angelo (LM, 188k calls) | 0,0079 | 0,0048 | **0,61** |
| Nosso FinBERT atual (4.620 calls) | 0,299 | 0,088 | **0,29** |

### Como corrigir

**Passo A (obrigatório):** re-pontuar por sentença (P1). Isso aumenta o `n` efetivo
por gestor e encolhe o viés. **Mas não o elimina** — o ruído continua proporcional a
`1/√n` do gestor mais quieto.

**Passo B (medir antes de decidir):** depois do re-scoring, recalcule
`corr(Tone Distance, nº de sentenças do gestor mais quieto)`.

> **Regra de decisão, fixada agora para não virar garimpo depois:**
> - Se `|corr| < 0,10` → basta incluir controles na regressão (nº de gestores,
>   `log` do nº de sentenças do gestor mais quieto, concentração de fala tipo
>   Herfindahl). O nulo de permutação vira robustez.
> - Se `|corr| > 0,25` → o **nulo de permutação vira o sinal principal**.
> - Entre 0,10 e 0,25 → rode os dois e reporte lado a lado.

**Passo C — o nulo de permutação.** Para cada call:

1. Junte todas as sentenças dos gestores daquela call num único conjunto.
2. Redistribua ao acaso entre os mesmos gestores, **preservando exatamente quantas
   sentenças cada um falou**.
3. Nesse mundo, por construção, **ninguém discorda** — todos amostram da mesma
   distribuição. Recalcule a Tone Distance.
4. Repita 200 vezes (semente fixa, do `config.yaml`). A média é `E[TD_nulo]`: a
   distância que **esta call específica** produziria só por ruído, dado o seu padrão
   de fala.
5. Sinal de-enviesado: `TD_ajustada = TD_real − E[TD_nulo]`.
   Versão padronizada: `z = (TD_real − E[TD_nulo]) / sd[TD_nulo]`. Reporte as duas.

Custo computacional: desprezível. Não roda FinBERT de novo — embaralha sentenças já
pontuadas. É numpy puro.

Fundamentação para a banca: é um **teste de aleatorização condicional** (Fisher, 1935),
sem hipótese de forma funcional. E tem precedente direto na literatura contábil:
**Barron, Kim, Lim & Stevens (1998, *The Accounting Review*)** mostram que a dispersão
de previsões de analistas é contaminada pelo número de analistas e pelo ruído
individual, e deve ser decomposta antes de ser lida como desacordo. Mesmo problema
estrutural, gestores no lugar de analistas.

**Sanidade obrigatória:** depois de aplicar, `corr(TD_ajustada, nº de sentenças do
gestor mais quieto)` tem que cair para perto de zero. Se não cair, o nulo está errado
— investigue, não maquie.

## P3 — O legado da tese errada contamina o código

O repositório nasceu sobre Brockman, Li & Price (2015) e o Angelo foi enxertado por
cima (ADR-023/024/025). Resquícios:

- O pacote se chama `tonediv`; a feature `mgmt_analyst_divergence` ainda existe.
- `tone_distance_per_call()` **descarta calls sem Q&A detectado**, porque a heurística
  de papéis não é confiável sem Q&A. O Angelo usa o **transcript inteiro**. Perdemos
  amostra por uma limitação que não é da tese, é da heurística.
- `min_manager_tokens: 25` — baixo demais, alimenta direto o viés do P2.
- O sinal operado, `tone_distance_clean`, é um resíduo estritamente-passado sobre
  confundidores (`residualize_pit`). É uma construção *ad hoc*, não o que o paper faz
  (regressão com efeitos fixos de empresa e trimestre + erros agrupados por empresa).
  **Faça a replicação do paper primeiro. Depois, se quiser, proponha a variante — e
  justifique.**

**Tarefa:** limpe. Renomeie o pacote para algo coerente com a tese
(`tonedist` ou similar). Apague `mgmt_analyst_divergence` como sinal — mantenha
`analyst_tone` e `analyst_tone_dispersion` **apenas como controles**, exatamente como
o Angelo. Reescreva `docs/DECISIONS.md` com os ADRs superados marcados como tal.

## P4 — Não há IA generativa no núcleo do projeto

O edital **exige** uso de IA generativa em pelo menos uma etapa (15% da nota). FinBERT
**não conta**: é um encoder classificador, não é generativo.

E, por sorte, o melhor uso de GenAI é exatamente o conserto do bug que quebrou o
projeto — veja a Etapa 1.

---

# 5. O plano, em etapas com checkpoint

## Etapa 0 — Ler e devolver um plano

**Não escreva código nesta etapa.** Leia o paper do Angelo. Leia `config.yaml`,
`docs/DECISIONS.md`, `src/tonediv/nlp/features.py`, `src/tonediv/nlp/roles.py`,
`src/tonediv/nlp/scorer.py`. Reproduza os números do diagnóstico da seção 4. Depois,
me devolva:

- Onde você concorda e onde discorda deste prompt.
- Um plano com estimativa de tempo de cada etapa.
- As três decisões de maior risco do projeto.

Espere meu ok.

## Etapa 1 — Classificar os oradores com um LLM ← *prerequisito de tudo, e a nossa GenAI*

Nada do Angelo existe sem saber, para cada fala: **quem falou, e essa pessoa é gestor
da empresa, analista de sell-side, ou a operadora.**

Hoje isso é regex frágil. Substitua por um **LLM lendo os cabeçalhos de fala**
("Timothy D. Cook, Chief Executive Officer", "Katy Huberty, Morgan Stanley").

Requisitos:

- Rode sobre os **cabeçalhos únicos** de todo o corpus (são poucos milhares, não 300k),
  com cache. Barato.
- Saída estruturada: `{nome_canônico, papel ∈ {gestor, analista, operadora},
  cargo, empresa}`.
- **Resolução de identidade**: "Mike Smith", "Michael Smith" e "Michael Smith, CFO"
  são a mesma pessoa. Isso importa — sem isso, um gestor vira dois pontos no plano.
- **Validação obrigatória**: amostre 300 cabeçalhos, classifique à mão, e reporte a
  matriz de confusão LLM × humano. Sem essa tabela, a etapa não está concluída.
- Documente prompt, modelo, versão, custo e taxa de acerto em `docs/GENAI_USAGE.md`.

Com papéis confiáveis, **remova o filtro de "só calls com Q&A"** e volte a usar o
transcript inteiro, como o paper.

## Etapa 2 — Re-pontuar o tom por sentença

FinBERT sobre sentenças. Agregue ao gestor ponderando por tokens. Guarde a distribuição
`[p_neg, p_neu, p_pos]` por sentença — o nulo de permutação vai precisar dela.

Antes de rodar: **estime o tempo** e me diga. Mantenha o checkpointing por shards.

Verificação: refaça o histograma de `p_pos`. A fração na faixa `(0,05 ; 0,95)` deve
subir bem acima dos 16,5% atuais.

## Etapa 3 — Tone Distance e o viés

Implemente a Equação (1) fielmente. Depois:

1. Reporte média, mediana, desvio, CV. Compare com o paper (0,0079 / 0,0048 / 0,61).
2. Reporte `corr(TD, nº de sentenças do gestor mais quieto)` e `corr(TD, nº de gestores)`.
3. Aplique a **regra de decisão** do P2.
4. Suba `min_manager_tokens` para um patamar defensável e mostre a sensibilidade da
   cobertura (quantas calls sobrevivem a 25 / 100 / 250 / 500 tokens).

Checkpoint. Não siga sem discutir comigo.

## Etapa 4 — Replicar o Angelo

Antes de qualquer backtest, mostre que a métrica se comporta como no paper.

- **Determinantes** (Tabela 2 do paper): regrida Tone Distance nas variáveis de
  disclosure e financeiras que conseguirmos construir com dados abertos. Seja explícito
  sobre o que **não** temos (I/B/E/S, Compustat, volatilidade implícita) e o que usamos
  como substituto.
- **H1**: `CAR[-1,+1] ~ Tone Distance + controles`, com **efeitos fixos de empresa e de
  ano-trimestre** e **erros-padrão agrupados por empresa**. Explique-me o que cada um
  desses três elementos faz antes de rodar.
- Controles do paper que conseguimos: `Disclosure Tone`, `Analyst Tone`,
  `Analyst Tone Dispersion`, `Industry Tone`, `Length`, `Standardized ME`,
  `Lagged Avg Tone Distance`. Documente o que faltou.
- Winsorize a 5% / 95%, como o paper.
- **H2** (risco de mercado pós-call) é testável com preços: desvio-padrão dos retornos
  e average range em 20 e 120 pregões. Faça.
- **H3** exige fundamentos que não temos. Diga isso, não finja.

Sinal esperado do coeficiente de H1: **negativo**. Se der positivo, **não conserte o
sinal** — investigue e reporte.

## Etapa 5 — Event study

Market model estimado em janela pré-evento, CAR em torno de t=0. Placebo com datas
falsas. A infraestrutura já existe; audite antes de reusar.

## Etapa 6 — Backtest

- Portfólio calendar-time, posições sobrepostas, horizonte de ~63 pregões (3 meses).
- Direção **fixada a priori pelo paper**: comprar Tone Distance alta.
- Long-short (top vs bottom quintil) e long-only vs benchmark.
- **T+1 rigoroso**: call às 17h ⇒ decisão no próximo pregão. Existe teste para isso;
  mantenha-o.
- **Point-in-time**: nenhum percentil, média de pares ou coeficiente pode enxergar
  dados do mesmo instante ou do futuro.
- Custos de transação declarados. Filtro de liquidez.
- Benchmarks: QQQ e SPY.

## Etapa 7 — Robustez

- Sensibilidade ao `min_manager_tokens` e ao nº mínimo de gestores.
- Subperíodos (pré/pós-2015). Decay do sinal.
- Placebo de sinal aleatório (várias permutações, não uma).
- **Deflated Sharpe Ratio** com `n_trials` honesto — conte *todas* as variantes que
  testamos, não só as que reportamos. Explique-me o que o DSR faz.
- **Survivorship bias**: a base é de constituintes atuais do S&P 500. Isso enviesa
  para cima. Meça o que der para medir e **declare o resto como limitação**.
- Poder estatístico: temos ~4.600 calls contra as 188.466 do paper. O efeito do Angelo
  (0,21% de CAR por variação interquartil) pode ser pequeno demais para detectarmos.
  **Calcule o poder e me diga**, antes de interpretar um resultado nulo como fracasso
  da tese.

## Etapa 8 — Relatório

Guie-se pelos pesos da banca (seção 7). Escreva honesto. Um resultado ruim bem
explicado vale mais que um resultado bonito e frágil — o próprio FAQ do desafio diz
isso, na pergunta 22.

---

# 6. Regras metodológicas inegociáveis

- **Sem look-ahead.** Toda média, percentil, coeficiente ou referência usa **apenas**
  dados estritamente anteriores ao instante da decisão. Calls simultâneas de outras
  empresas não entram.
- **T+1.** A decisão nunca acontece no mesmo pregão da call se a call foi depois do
  fechamento.
- **A direção do sinal é do paper, não nossa.** Nunca inverta o sinal porque o retorno
  melhorou.
- **Todo parâmetro mora no `config.yaml`.** Nenhum número mágico no código.
- **Determinismo.** Toda aleatoriedade deriva de uma semente do config.
- **Cada variante testada é registrada**, mesmo as que falharam. Isso alimenta o
  `n_trials` do DSR e é a nossa defesa contra p-hacking.
- **A suíte de testes tem que continuar verde**, e os testes de vazamento não podem ser
  afrouxados para acomodar código novo.

---

# 7. Como a banca avalia (Manual Oficial)

| Critério | Peso |
|---|---|
| Apresentação do robô (nome e identidade) | 5% |
| **Conceito da estratégia (criatividade e inovação)** | **20%** |
| **Modelagem (estrutura e lógica do modelo)** | **20%** |
| Backtest (rigor e mitigação de vieses) | 15% |
| Análise dos resultados (clareza e profundidade) | 15% |
| Conclusão e próximos passos | 10% |
| Uso de IA generativa no processo | 15% |

Pontos negativos explícitos do manual: ausência de hipótese definida; **uso de
complexidade sem fundamentação**; processos não replicáveis; **uso de ferramentas
externas sem compreensão adequada**; apresentação exclusivamente descritiva de
métricas; omissão de fragilidades do modelo; uso de IA "superficial ou meramente
declaratório".

Leia isso de novo. Cada uma dessas linhas descreve um jeito de o projeto atual perder
pontos.

## Prazos

- **Pré-relatório: 31/07/2026**
- Entrega final: 17/08/2026
- Quartas de final (online): 31/08/2026
- Semifinal (online, com Q&A ao vivo): 09/09/2026
- Final presencial (Faria Lima): 26/09/2026

Hoje é 09/07/2026. Restam **22 dias** até o pré-relatório.

---

# 8. Sua primeira resposta

Não escreva código. Faça a Etapa 0 e me responda com:

1. **O que você entendeu que é Tone Distance** — com suas palavras, em 5 linhas. Se
   você escrever qualquer coisa envolvendo analistas como parte do sinal, pare e releia.
2. **Os números do diagnóstico, reproduzidos por você** nos parquets do repositório.
3. **Onde você discorda deste prompt.**
4. **Um plano de 22 dias** até o pré-relatório, com estimativa de tempo por etapa.
5. **Os três maiores riscos** do projeto, e o que faríamos se cada um se materializasse.

Depois disso, pare e espere.
