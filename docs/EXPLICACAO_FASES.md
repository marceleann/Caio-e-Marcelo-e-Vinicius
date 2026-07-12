# O projeto Tone Distance, fase a fase

> **AVISO DE VERSÃO (leia antes).** Este documento foi escrito na tese **inicial**
> — a "Tone Divergence" de Brockman, Li & Price (2015), divergência de tom
> analistas×gestão medida por distância de Jensen-Shannon. O projeto depois
> **pivotou** para **Angelo et al. (2025, *Financial Review*), "Tone Distance"**:
> o desacordo de tom **entre os próprios gestores** da mesma call, e o sinal
> operável é o **resíduo estritamente-passado** dessa distância sobre os
> confundidores (o **sinal limpo**, `tone_distance_clean`). As decisões antigas
> ADR-021/022 ficam **superadas** pelo pivô. Onde este texto trata a divergência
> analistas×gestão como "a tese", leia-a como a **feature de comparação/legado**
> que hoje serve de robustez (segue calculada, não é o produto). A tese atual, os
> números finais e o método de provas estão em [README.md](../README.md) e
> [PROGRESS.md](PROGRESS.md); mantemos este arquivo como registro didático do
> percurso e da infraestrutura (fases 0–2) reaproveitada.
>
> Documento didático para explicar o que **foi feito**, o que **está sendo feito** e o
> que **será feito** em cada uma das seis fases do projeto. Escrito para ser lido por
> quem não acompanhou o código de perto (a orientadora) e para servir de material de
> estudo da própria equipe. Números empíricos vêm da execução real do pipeline;
> conceitos técnicos aparecem em caixas **"Conceito-chave"**.

---

## Visão geral: a ideia em uma frase e por que dividir em fases

**A tese (atual, após o pivô).** Numa *earnings call* (teleconferência de resultados),
vários executivos da mesma empresa falam — o CEO, o CFO e outros. Nossa hipótese é que,
quando esses **gestores divergem de tom entre si** na mesma call, esse desacordo carrega
informação sobre o retorno futuro da ação. É a tese de **Angelo et al. (2025,
*Financial Review*), "Tone Distance"**, medida no paper com o dicionário de palavras
Loughran-McDonald; **nossa contribuição** é modernizar o sensor de tom para um modelo de
linguagem (**FinBERT**). O sinal operável não é a distância crua (confundida por nível
de tom, tamanho e setor), e sim o **resíduo estritamente-passado** dela sobre os
confundidores — o **sinal limpo** (`tone_distance_clean`). A direção é fixada a priori
pelo paper: comprar (long) distância de tom **alta** (prêmio de risco no horizonte de
~1–3 meses).

> **Nota do pivô.** O projeto **começou** na tese de **Brockman, Li & Price (2015)** —
> divergência de tom *analistas×gestão* medida por distância de Jensen-Shannon, que
> chamávamos de "Tone Divergence" — e **pivotou** para Angelo por ser mais nova,
> inovadora e alinhada à orientadora. A infraestrutura das fases 0–2 (dados, papéis,
> scoring FinBERT) foi reaproveitada; a divergência analistas×gestão **segue sendo
> calculada como feature de comparação/robustez**, não como o produto. Este documento
> ainda usa o vocabulário da tese antiga em vários pontos — veja o aviso no topo.

**Duas perguntas de pesquisa:**
1. **Principal:** a distância de tom **entre os gestores** de uma mesma call antecipa
   retornos em ações de tecnologia dos EUA (2005–2025)?
2. **Secundária (decay):** esse tipo de efeito de sinais de tom, uma vez *publicado* na
   literatura, continuou funcionando — ou o mercado aprendeu e ele decaiu? (referência:
   McLean & Pontiff, 2016).

**O entregável** não é um artigo: é uma **estratégia operável**, com regras mecânicas e
retorno líquido de custos.

**Por que fases.** Cada fase é uma camada que só faz sentido sobre a anterior — como
montar um sensor, calibrá-lo e só então usá-lo para medir. Pedimos à IA que
trabalhasse assim para conseguirmos **entender e perguntar ao longo do caminho**, em vez
de receber tudo pronto no fim. O fluxo:

| Fase | Nome | O que produz | Scripts |
|---|---|---|---|
| 0 | Esqueleto | Fundação: estrutura, `config.yaml`, registro de decisões | — |
| 1 | Dados e diagnóstico | Transcrições + preços + universo, tabelas auditáveis | 00–01 |
| 2 | Papéis e NLP | *Quem* fala (analista/gestão) + *tom* de cada fala | 02–03 |
| 3 | Features e alinhamento | As 7 variáveis por call + alinhamento sem "olhar o futuro" | 04 |
| 4 | Provas | Evidência científica (event study) + estratégia (backtest) | 05–07 |
| 5 | Decay e fechamento | Teste do decaimento pós-2015 + a lâmina da estratégia | 08–09 |

**Onde estamos hoje (resumo honesto):** toda a "máquina" (o código das seis fases) está
**construída e testada** — 156 testes automatizados verdes. A ingestão de dados (Fase 1)
e a inferência de papéis (Fase 2, script 02) **já rodaram na base real**. O único passo
**em execução agora** é a pontuação de tom pelo FinBERT (Fase 2, script 03), que é a
etapa mais cara (~1 dia de processamento, feita de forma retomável). Quando ela
terminar, rodamos os scripts 04–09 sobre os dados reais e preenchemos os números finais
da estratégia. Detalhe por fase abaixo.

---

## Fase 0 — Esqueleto (a fundação de engenharia)

**O que resolve.** Antes de qualquer análise, montar um alicerce que torne o projeto
**reprodutível, auditável e defensável** perante a banca. Um resultado bonito que não
pode ser reproduzido, ou cuja premissa está escondida no código, não vale nada numa
competição julgada por quants.

**O que foi feito.**
- **Estrutura de pastas** que separa a *biblioteca* (`src/tonediv/`, lógica pura e
  testável) da *orquestração* (`scripts/`, que só carregam config, chamam a biblioteca e
  salvam resultado). Isso permite testar a matemática do sinal sem baixar dados nem
  carregar o FinBERT.
- **`config.yaml` — a fonte única de verdade** (352 linhas, 18 seções): *todos* os
  parâmetros do projeto (datas, limiares, lista de empresas, custos) vivem num arquivo
  só, cada um comentado com a justificativa. Nada de "número mágico" escondido no código.
- **`docs/DECISIONS.md` — o registro de decisões (ADRs):** cada escolha metodológica
  não-óbvia é congelada com contexto, alternativas e justificativa. Hoje são 22 ADRs.
- **Documentos vivos:** regras do desafio, esquemas de dados, progresso e uso de IA
  generativa (exigência do critério 7). Ferramentas de qualidade: Makefile, versões de
  bibliotecas fixadas, formatação/testes automáticos.

> **Conceito-chave — Fonte única de verdade e ADR.** Um "número mágico" é uma constante
> jogada no meio do código sem nome nem explicação; ninguém sabe de onde veio. A regra
> aqui é o oposto: cada parâmetro mora no `config.yaml` com um comentário do porquê. E um
> **ADR** (*Architecture Decision Record*) é uma nota curta que registra **por que** uma
> decisão foi tomada. Juntos, eles fazem com que qualquer avaliador consiga auditar cada
> premissa — e são, literalmente, a base do capítulo de metodologia e da defesa oral.

**Rigor na prática.** Já nesta fase um teste automatizado pegou um bug real: o ticker
`ON` (ON Semiconductor) era interpretado pelo formato YAML como o valor lógico
"verdadeiro" e sumia da lista de empresas. Foi corrigido e blindado com uma verificação
que **falha na hora** se algum ticker não for texto.

**Status:** ✅ feito.

---

## Fase 1 — Dados e diagnóstico

**O que resolve.** Transformar duas fontes 100% open-source em tabelas limpas e
confiáveis, e **medir a realidade da base** antes de gastar processamento com ela.

**O que foi feito.**
- **Ingestão:** transcrições do dataset `kurry/sp500_earnings_transcripts` (HuggingFace) e
  preços do `yfinance` (mais o índice de mercado e os benchmarks QQQ/SPY), viradas em
  tabelas tipadas. Uma tabela com uma linha por *call* e outra "longa" com uma linha por
  *fala*, preservando a ordem (essencial para a próxima fase saber quem falou antes de
  quem).
- **Definição do universo tech por lista explícita**, não pelo setor oficial (GICS) — ver
  conceito abaixo. Casos fronteiriços viram "blocos de sensibilidade" (pagamentos, folha,
  solar, etc.) que a análise roda **com e sem**, transformando a discussão subjetiva
  "isso é tech?" em número.
- **Diagnóstico rodado na base real**, sem maquiar: **33.362** calls no dataset, **116**
  empresas do nosso universo presentes, **5.452** calls aprovadas na qualidade,
  **408.248** falas (mediana de 70 falas/call). Distribuição de horários das calls:
  26,3% após o fechamento, 36,2% pré-abertura, 36,9% durante o pregão, 0,6% sem horário.

> **Conceito-chave — Universo e survivorship bias.** *Universo* é o conjunto de empresas
> que a estratégia pode negociar. Definir "quem é tech" parece trivial, mas a
> classificação oficial GICS mudou **duas vezes** no período (2018 tirou Google/Meta/
> Netflix do setor de TI; 2023 moveu pagamentos para Financeiro) — logo "setor = TI" não
> é estável. Por isso usamos lista curada + varredura automática. Já o **viés de
> sobrevivência** é o erro de estudar só quem sobreviveu: se você só olha as empresas que
> existem hoje, seus resultados parecem melhores do que a realidade, porque as que
> quebraram sumiram da amostra. Medimos isso: das 37 empresas deslistadas que
> procuramos, só **5** têm transcrição na base — uma limitação que **declaramos**, em vez
> de esconder.

**Rigor na prática.** A primeira execução real tropeçou numa quebra de versão do
`yfinance` (retornava zero preços) e numa cascata de bloqueios do Yahoo; ambos foram
corrigidos (atualização de versão + espera progressiva entre requisições). 13 testes que
rodam **sem internet** confirmam cada transformação.

**Status:** ✅ feito e executado na base real.

---

## Fase 2 — Papéis e NLP (os dois "sensores")

**O que resolve.** A tese precisa de duas informações que o dado bruto não entrega
prontas: **quem** está falando (analista ou gestão) e **qual o tom** de cada fala.

**O que foi feito.**
- **Inferência de papéis** (`roles.py`): como o dataset traz o *nome* do orador mas não o
  *papel*, usamos uma heurística baseada na estrutura da call — detecta-se onde começa o
  Q&A (pela fala do moderador), assume-se que quem falou antes é gestão e quem estreia no
  Q&A é analista. **Já rodou na base real:** Q&A detectado em **80%** das calls,
  **99,95%** das falas não-moderador classificadas (204.476 de gestão, 132.655 de
  analista).
- **Pontuação de tom** (`scorer.py`): o **FinBERT** lê cada fala e devolve três
  probabilidades — quão negativa, neutra e positiva ela soa. **Está em execução agora**,
  de forma retomável (ver "checkpoint" abaixo).

> **Conceito-chave — FinBERT e a inversão silenciosa de labels.** FinBERT é um modelo de
> linguagem treinado em textos financeiros; é o "termômetro de humor" que transforma
> texto em número. Uma armadilha sutil: o modelo devolve três números, mas **a ordem
> deles** (qual é o negativo, qual é o positivo) depende de como foi treinado e não é
> padronizada. Se a gente adivinhar errado, troca positivo por negativo e o projeto
> inteiro fica com o **sinal ao contrário — sem dar erro nenhum**. A defesa é perguntar
> ao próprio modelo, em tempo de execução, o nome de cada posição, e nunca confiar na
> ordem por acaso.

> **Conceito-chave — Checkpoint (retomada em fatias).** Pontuar ~307 mil falas leva
> horas em CPU; se a máquina desligar no meio, recomeçar do zero seria terrível. O
> *checkpoint* salva o trabalho em fatias de 4.000 falas em disco assim que cada uma
> termina. Se o processo cair e for religado, ele lê do disco o que já foi feito e
> continua de onde parou — como um videogame que salva o progresso.

**Rigor na prática (o ponto alto do projeto).** Uma **revisão adversarial com múltiplos
agentes de IA** reproduziu e corrigiu dois bugs críticos e **silenciosos**:
1. A frase-padrão de abertura do moderador ("*mais tarde faremos uma sessão de perguntas
   e respostas...*") disparava a detecção do Q&A cedo demais e rotulava **toda a gestão
   como analista** — sem dar erro, a cobertura continuava 100%, mascarando a falha.
2. A grafia inconsistente do mesmo executivo entre as seções (nomes colados como
   "MarcBenioff", apelidos, cargo grudado) fazia o gestor não "casar" com a própria fala
   e virar analista — contaminando ~130 calls.

Cada correção está ancorada num ADR e travada por testes de regressão, para os erros não
poderem voltar sem quebrar a suíte.

**Status:** 🟡 papéis prontos e rodados; **scoring de tom em execução** (checkpointado).

---

## Fase 3 — Features e alinhamento point-in-time

**O que resolve.** Condensar as notas de tom (por fala) em **7 indicadores por call** e
alinhar cada call ao histórico de preços com um cuidado obsessivo para **não usar
nenhuma informação do futuro**.

**O que foi feito.**
- **As 7 features** por call. O sinal da tese **atual** (Angelo) é a **distância de tom
  entre os gestores** da própria call — cuja versão operável é o resíduo sobre os
  confundidores (o **sinal limpo**). Entre as 7 features também está a
  `mgmt_analyst_divergence` — a distância de Jensen-Shannon entre o perfil de tom dos
  analistas e o da gestão no Q&A, que **era a tese antiga (Brockman) e hoje é mantida
  como feature de comparação/robustez** (segue calculada). Outras medem o tom líquido, a
  dispersão, a variação em relação à call anterior da mesma empresa, e o "gap" entre o
  discurso preparado e as respostas espontâneas.
- **Ajuste idiossincrático:** parte do tom é só "clima do setor" naquele momento; para
  isolar o que é específico da empresa, subtrai-se a média das *outras* empresas nos 90
  dias anteriores.
- **Alinhamento anti-look-ahead** (`pit.py`): a estratégia só entra na abertura do
  primeiro pregão em que a informação já era pública, e uma **guarda automática dispara
  erro** se alguma decisão não for estritamente posterior à call.

> **Conceito-chave — Look-ahead bias e point-in-time.** O *look-ahead* é o pecado capital
> de um backtest: usar, sem querer, uma informação que só existiria no futuro — como
> apostar no cavalo já sabendo o resultado da corrida. *Point-in-time* é a disciplina
> oposta: a cada instante, o modelo só pode enxergar o que era genuinamente conhecido
> naquele momento. Exemplo concreto do projeto: se a call termina às 19h (após o
> fechamento), não dava para negociar naquele dia, então a regra manda esperar a abertura
> do dia seguinte (**T+1**).

> **Conceito-chave — Distância de Jensen-Shannon.** Uma régua que mede o quão diferentes
> são duas distribuições de probabilidade, num valor entre 0 (idênticas) e 1 (sem nada em
> comum). Na feature de comparação/legado (a antiga tese de Brockman), mede o quanto o
> "perfil de tom" dos analistas difere do da gestão. Foi escolhida por ser simétrica e
> por funcionar mesmo quando uma das classes tem probabilidade zero — ao contrário de
> medidas mais ingênuas. (A tese **atual**, de Angelo, mede a distância de tom **entre
> gestores** por distância euclidiana no plano de coordenadas de tom — ver README.)

**Rigor na prática.** Uma auditoria da fase encontrou quatro armadilhas sutis e graves —
entre elas, uma janela de "90 dias" que, por um detalhe de unidade de tempo, virava
"90.000 dias" em certas versões do Python, e uma junção com preços que vazava o
fechamento do próprio dia. Todas corrigidas e travadas por testes.

**Status:** ✅ código + 35 testes prontos; ⏳ execução na base real pendente (depende do
scoring da Fase 2).

---

## Fase 4 — Provas (a espinha probatória)

**O que resolve.** Provar que o sinal contém informação **e** empacotá-lo numa estratégia
que sobrevive a custos e às armadilhas clássicas de quant. É a fase que transforma "temos
uma variável" em "temos evidência + um produto".

**O que foi feito.**
- **Event study** (a evidência científica): mede se, ao redor da call, a ação teve
  retorno *anormal* (acima do que o mercado explicaria), comparando o grupo de sinal alto
  com o de sinal baixo — e testa isso contra uma distribuição nula honesta de "datas
  falsas".
- **Backtest calendar-time** (o entregável): a estratégia operável, com regras mecânicas
  (universo tech, filtro de liquidez, entrada no open T+1, holding de 5 pregões, teto de
  10% por nome, 5 basis points de custo por perna), nas variantes long-short e long-only.
- **Arsenal anti-viés:** ranking point-in-time, Deflated Sharpe Ratio, PBO/CSCV,
  validação cruzada com purga e embargo, testes de placebo, e um **holdout de 18 meses**
  que fica lacrado durante todo o desenvolvimento.

> **Conceito-chave — Event study vs. backtest.** São duas perguntas diferentes. O *event
> study* pergunta "ao redor do dia do evento, a ação teve retorno anormal?" — é um teste
> **científico** da existência de informação. O *backtest* pergunta "se eu operasse esse
> sinal todo dia por 20 anos, ganharia dinheiro depois de custos?" — é um teste de
> **engenharia** da estratégia. Fazemos os dois: primeiro provamos o efeito, depois o
> transformamos em produto.

> **Conceito-chave — Deflated Sharpe e overfitting.** O Sharpe mede retorno por unidade de
> risco. Mas se você testa 100 combinações e escolhe a melhor, o melhor Sharpe está
> inflado por **sorte** — alguém sempre parece bom em 100 tentativas. O *Deflated Sharpe*
> desconta esse viés usando o número **real** de tentativas. E o *PBO* (Probability of
> Backtest Overfitting) responde diretamente: "a estratégia que pareceu melhor no passado
> costuma continuar boa no futuro, ou é só overfitting?".

> **Conceito-chave — Portfólio calendar-time.** Earnings calls acontecem em dias esparsos.
> Se você exigisse "compre 20 empresas que tiveram call hoje", quase todo dia a carteira
> ficaria vazia. A solução (de Jegadeesh-Titman): cada evento abre uma pequena posição
> que dura alguns pregões, e a carteira de qualquer dia é a **média** de todas as posições
> ainda abertas. Assim, mesmo com eventos raros, há uma carteira cheia e uma série de
> retornos contínua.

**Rigor na prática.** Uma auditoria multi-agente confirmou que o núcleo matemático estava
correto e corrigiu quatro detalhes de implementação (a exposição líquida residual do
long-short passou a ser reportada e neutralizada; empates no ranking tratados
corretamente; calendário real de pregões em vez de dias fabricados; custo de saída
cobrado sobre o valor atual da posição). Antes de confiar na máquina, ela foi testada num
**mundo sintético com "alpha plantado"**: ela precisa recuperar o sinal quando ele existe
**e** não achar nada quando só há ruído — os dois lados são eliminatórios.

**Status:** ✅ código + testes prontos e verdes (validados em dados sintéticos); ⏳
execução na base real pendente (scripts 05–07, dependem do scoring).

---

## Fase 5 — Decay e fechamento

**O que resolve.** Responder à pergunta secundária — **o sinal decaiu depois de publicado
em 2015?** — e fechar o entregável com a lâmina da estratégia.

**O que foi feito.**
- **Análise de decay** (`decay.py`): compara três métricas (IC, CAR e Sharpe) entre o
  regime pré-2015 e o pós-2015, na amostra **completa** (o corte em 2015 é só a fronteira
  dos regimes, não o fim dos dados), cada uma com o teste de diferença estatística
  **correto para a sua natureza**.
- **A lâmina** (`strategy_spec.md`): documento para um gestor que nunca viu o código —
  tese, definição do sinal, regras 100% mecânicas da carteira, comparações obrigatórias e
  riscos declarados. Os números de desempenho ficam como marcadores até a rodada real.

> **Conceito-chave — Decay (McLean & Pontiff).** Quando um padrão lucrativo é publicado
> num paper, muita gente passa a explorá-lo — e, ao fazê-lo, o próprio padrão enfraquece
> ou some. McLean & Pontiff (2016) mostraram que, em média, o retorno de anomalias cai
> cerca de metade após a publicação. Nossa pergunta testa exatamente isso para o sinal de
> 2015. **Os dois desfechos são resultado válido:** se sobreviveu, é sinal robusto; se
> decaiu, é evidência honesta de que o mercado aprendeu.

> **Conceito-chave — Por que um teste diferente para cada métrica.** IC e CAR usam
> *permutação* (embaralhar os rótulos "pré/pós" e ver com que frequência o acaso produz
> uma diferença tão grande), porque suas unidades — trimestre e evento — são
> aproximadamente independentes. Mas o Sharpe é calculado sobre a série **diária** de
> retornos da estratégia, que é **autocorrelacionada** (cada posição fica aberta 5 dias,
> então o retorno de hoje compartilha ações com o de ontem). Para o Sharpe usamos um
> *bootstrap de blocos móveis*: reamostrar em blocos contíguos de 5 dias preserva essa
> dependência; reamostrar dia a dia a quebraria e declararia **significância falsa**.

**Rigor na prática.** A auditoria da fase achou uma armadilha séria — não no código, mas
nos **testes**: eles não travavam o uso do bootstrap de blocos, e trocá-lo por um
bootstrap ingênuo teria passado despercebido. Numa série autocorrelacionada, essa troca
**inverte a conclusão** sobre o decay (num teste construído, o método correto dá p≈0,18 —
não significativo — e o ingênuo dá p≈0,025 — falso positivo). Adicionamos um teste
dedicado que prova numericamente que o método importa.

**Status:** ✅ código + testes + lâmina prontos; ⏳ execução na base real pendente
(scripts 08–09).

---

## Fecho: o que falta e o critério de IA

**O caminho até os números finais.** Assim que o scoring de tom (Fase 2) concluir —
único passo em execução, retomável, ~1 dia — basta rodar os scripts 04→09 sobre a base
real para preencher: as 7 features, o event study, o backtest líquido nas duas variantes,
a grade de robustez, o teste de decay e a lâmina final. A **metodologia já está
congelada**; falta só virar a chave.

**Uso de IA generativa (critério 7).** O projeto foi desenvolvido em *pair-programming*
com IA (registro em `docs/GENAI_USAGE.md`): estruturação da hipótese, arquitetura,
implementação e, sobretudo, as **revisões adversariais multi-agente** que encontraram
bugs reais em cada fase. Distinção importante para a banca: o FinBERT é um classificador
(NLP/ML) no núcleo do modelo — a exigência de IA generativa é cumprida pelo uso de IA no
**processo de engenharia**, não pelo FinBERT.

**A promessa central do projeto** é que cada afirmação seja rastreável: todo parâmetro
tem uma linha no `config.yaml`, toda decisão tem um ADR, todo cuidado anti-viés tem um
teste que quebra se alguém o remover. É isso que separa "um backtest que deu certo" de
"um trabalho sólido, bem pensado e tecnicamente consistente" — que é exatamente o que o
edital pede.
