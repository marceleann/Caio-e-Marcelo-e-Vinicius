# CORO — Divergência de tom entre gestores como sinal quantitativo

**Pré-relatório · Desafio Quant AI 2026 (Itaú Asset)**
Equipe: Marcelo e Caio · Orientação: Profa. Nadia Cardoso Moreira
Repositório (código + dados derivados + replicação em 4 comandos):
https://github.com/marceleann/Caio-e-Marcelo-e-Vinicius

> **[NOME A CONFIRMAR PELO TIME]** Em avaliação: **DESAFINADO** ("quando a
> diretoria desafina, o mercado ouve"), **DISSONA**, **OUVIDOR** ou **CORO**.
> Qualquer que seja a escolha, o nome resume a tese em uma imagem: não
> medimos o tom médio da empresa, e sim o desalinhamento entre as vozes que
> a representam.

---

## Resumo executivo — a história em uma página

**O ponto de partida.** Numa teleconferência de resultados, a empresa ensaia
uma mensagem única — mas várias vozes a entregam: CEO, CFO, diretores. A
tese deste trabalho é que, quando essas vozes desafinam entre si, isso é
informação: desacordo interno e incerteza que a mensagem oficial tenta
uniformizar, mas que o Q&A ao vivo não deixa esconder. Essa divergência tem
medida (a **Tone Distance** de Angelo et al., 2025) e, mostramos, tem preço.
Replicamos o artigo do zero no S&P 500 — 33.362 calls, 685 empresas,
2005–2025, dados 100% públicos — com uma única inovação deliberada: um
segundo medidor de tom, o modelo neural FinBERT, ao lado do dicionário
Loughran-McDonald do artigo.

**O que encontramos, em três atos.** *Primeiro:* a hipótese central replica
— o mercado penaliza no anúncio a empresa cujos gestores divergem (t até
−2,9, e o resultado sobrevive à troca completa do sensor de tom). No
caminho, um achado próprio: a fórmula do artigo admite duas leituras, e
diagnosticar qual delas separa sinal de ruído (o peso de fala de cada
gestor) virou contribuição metodológica. *Segundo:* a tese vira carteira —
comprar todo mês as 10 empresas de maior Tone Distance rendeu **+20,4% ao
ano contra +13,2% do S&P 500** em 16 anos e meio (excesso com t=2,60), com
a leitura honesta de que parte disso é o equal-weight e o tilt próprio da
seleção (+3,6% a.a. sobre o universo equivalente) é consistente em todos os
subperíodos, mas não significativo com só 10 nomes. *Terceiro:* o sinal
prevê a **incerteza do próximo balanço** (t=+3,58, nosso resultado
prospectivo mais forte) — divergência hoje antecipa surpresa de lucro
grande no trimestre seguinte.

**O que o trabalho descarta — e por que isso vale tanto quanto o que
valida.** Das três hipóteses do artigo, a de risco (H2) morre na validação
temporal: verdadeira como descrição, vazia como sinal. O detalhe central: em
amostra cheia, as **três** hipóteses pareciam confirmadas em 48 de 48
momentos testados — só separar passado de futuro distingue achado real de
miragem estatística. Esse filtro, aplicado sem exceção, é o que sustenta
cada número deste relatório; todos saem de scripts versionados num
repositório público e se reproduzem em 4 comandos.

*(As seções seguem os critérios do edital: identidade §1, conceito e
hipóteses §2, dados e modelagem §3, resultados e validação §4, backtest e
estratégia §5, análise crítica §6, conclusão §7, uso de GenAI §8.)*

---

## 1. Identidade do robô

O CORO processa a transcrição de cada teleconferência de resultados
(*earnings call*) e transforma uma pergunta qualitativa — "os executivos da
empresa estão alinhados no que dizem?" — em um número: a **Tone Distance**
(distância de tom), a dispersão do tom entre os gestores que falam na mesma
call. A saída é objetiva e operacionalizável: um escore por empresa-trimestre,
disponível minutos após a divulgação da transcrição, usável como sinal de
seleção e como indicador de risco.

## 2. Conceito e hipótese (o fenômeno, a justificativa, o teste)

**Fenômeno.** Numa earnings call, a empresa ensaia uma mensagem única, mas
vários executivos falam (CEO, CFO, diretores). Quando os tons deles divergem
entre si, essa divergência pode revelar informação privada que a mensagem
oficial tenta uniformizar: desacordo interno, incerteza sobre o trimestre,
narrativas que não fecham.

**Justificativa econômica.** A literatura de *disclosure* mostra que o mercado
extrai informação do que é mais difícil de gerenciar na comunicação (Huang et
al. 2014; Lee 2016). A divergência entre gestores é cara de coordenar em tempo
real no Q&A. Angelo, Johnston, Singh e Wan (2025, *The Financial Review*)
formalizam a medida (Tone Distance) e documentam, em ~188 mil calls
americanas: retorno anormal negativo no anúncio, risco futuro maior e retorno
médio maior nos meses seguintes (compensação pelo risco revelado).

**Hipóteses testáveis** (as três do artigo, pré-registradas):
- **H1**: mais Tone Distance → retorno anormal (CAR) menor no anúncio;
- **H2**: mais Tone Distance → mais risco realizado após o anúncio;
- **H3**: mais Tone Distance → piora de resultados operacionais futuros.

**Como testamos.** Replicação fiel do desenho econométrico do artigo em
universo, dados e código próprios — e, como contribuição, a mesma bateria com
**dois sensores de tom independentes** (léxico Loughran-McDonald e o modelo
neural FinBERT em nível de sentença), o que permite separar "achado real" de
"artefato do medidor de tom".

## 3. Dados e modelagem

**Fluxo: dados → modelo → decisão → teste.**

| etapa | conteúdo |
|---|---|
| Entrada | 33.362 transcrições de earnings calls (S&P 500, 2005–2025, 685 empresas; dataset público kurry/HuggingFace) |
| Papéis | cada fala é atribuída a gestor/analista/operador por classificador em dois níveis: **LLM via CLI** (12.678 cabeçalhos) + regras determinísticas; validado com matriz de confusão (~90% de acerto contra rotulagem manual) |
| Tom | dois sensores: contagem de palavras positivas/negativas do dicionário Loughran-McDonald oficial; e FinBERT (finbert-tone) sentença a sentença |
| Sinal | **Tone Distance** (Eq. 1 de Angelo): cada gestor vira um ponto (fração positiva, fração negativa); TD = distância média dos gestores ao centro da call; mínimo 2 gestores |
| Saída | escore por call + variantes (centroide agregado; ponderação pelo volume de fala) |
| Teste | painel com efeitos fixos de empresa e ano-trimestre, erros agrupados por empresa, winsorização 5/95 — a Eq. (3) do artigo — mais backtest calendar-time próprio |

**Retornos anormais (CAR)** calculados como no artigo, conferido no texto:
CAPM estimado em janela de 100 pregões (mínimo 70) com intervalo de 50 antes
do evento; janelas de evento [-1,+1], [-1,+2], [-1,+5] no dia do anúncio.
**Controles** reconstruídos de fontes primárias: fundamentos trimestrais do
SEC EDGAR (XBRL, alinhados pela data de protocolo — sem *look-ahead*),
indústria Fama-French 49 via código SIC de cada CIK, surpresa de lucro
(proxy yfinance), taxa livre de risco de Ken French.

**Replicabilidade** (critério explícito do desafio): repositório público com
código, documentação e **dados derivados versionados**; qualquer avaliador
reproduz as regressões com 4 comandos, sem reprocessamento. Testado por nós em
clone limpo: números idênticos.

**Um achado de modelagem que virou contribuição.** A fórmula do centro da
call no artigo admite duas leituras: (A) média simples das frações dos
gestores; (B) fração agregada (o "tom do transcript"). Implementamos as duas.
Na leitura A, um orador marginal desloca o centro — caso real: na call da
Apple de 2020, o profissional de relações com investidores falou 283 palavras,
das quais 6 "negativas" pelo dicionário eram o aviso legal padrão
(*risks, uncertainties*), e esse ponto respondia por um terço do peso do
sinal. A leitura B (e, além dela, a ponderação pelo volume de fala, que o
próprio artigo propõe na Tabela 8) elimina essa contaminação. Verificamos a
implementação com reimplementação independente nas 32.387 calls (diferença
máxima 8×10⁻¹⁷) e exemplo auditável à mão.

## 4. Resultados

**Validação da construção.** Nossa TD tem distribuição quase idêntica à do
artigo (média/mediana/desvio 0,0086/0,0080/0,0047 vs 0,0079/0,0074/0,0048),
e os controles reconstruídos batem com a Tabela 1 dele (alavancagem 0,24 vs
0,235; ETR mediana 0,22 vs 0,21; suavização 1,6 vs 1,4).

**H1 — o resultado central, em dois sensores.** Efeito da TD sobre o CAR do
anúncio (painel com efeitos fixos; estatística t; amostra 2009+, ~23 mil
eventos, ~540 empresas):

| construção do sinal | LM (dicionário) | FinBERT (neural) |
|---|---|---|
| centro = média simples (leitura A) | −0,7 / −0,6 / −0,3 | +0,1 / −0,2 / −0,5 |
| centro = tom agregado (leitura B) | **−2,0 / −2,1 / −1,7** | **−1,9 / −2,3 / −2,3** |
| ponderada pelo volume de fala (Tabela 8 do artigo) | **−2,9 / −2,7 / −2,3** | **−2,4 / −2,4 / −2,5** |

(t nas janelas [-1,+1] / [-1,+2] / [-1,+5]; efeito econômico: uma variação
interquartil de TD ≈ **−0,11% a −0,16%** de CAR, contra −0,21% no artigo.)

Três leituras desta tabela: (i) a hipótese H1 do artigo **replica** no S&P
500 quando o sinal respeita o volume de fala de cada gestor; (ii) a
progressão é monotônica — quanto mais peso ao ruído dos oradores marginais,
mais fraco o sinal — o que diagnostica com precisão por que a versão
igual-ponderada falha em large caps; (iii) os dois sensores, que concordam
entre si em apenas 38% (correlação), produzem o mesmo quadro — o achado não
é artefato do medidor de tom.

**H2 — risco: associação em amostra cheia que NÃO sobrevive à validação
temporal.** Na amostra completa, a TD ponderada associa-se a volatilidade
futura (t = +2,5 no sensor LM; nada no FinBERT). Mas o teste walk-forward
(estimar só com o passado em cada corte anual e conferir no futuro não
visto) reprova o canal na nossa especificação; a investigação (auditoria da
variável com recomputação independente — idêntica a 6 casas decimais — e
decomposição de janelas) localizou a origem da divergência com o artigo:
**a janela**. O artigo mede a vol a partir do dia +1 (ainda dentro da
ressaca do anúncio); nós, do +2 (fora dela). Com a janela do artigo, a H2
replica em amostra cheia (t=+2,1); com a janela limpa, não existe (t=+1,3);
e a diferença é o **eco mecânico do próprio anúncio** — empresas de TD alta
têm reação maior (a H1), e janelas que encostam no evento herdam essa
turbulência. Prospectivamente, nem a janela do artigo sustenta
(walk-forward: futuro na direção esperada em 28/48 momentos; previsão
condicional t=+1,0) — e a versão limpa falha por completo (0/48 no
horizonte longo). **Caracterização final: a H2 se confirma como descrição e
reprova como sinal.** Empresas de TD alta são, de fato, mais voláteis — na
carteira executada do §5, o top-10 escolhido sempre *antes* do mês realiza
vol de 20,6% contra 17,4% do bottom-10 —, mas essa é uma informação que a
própria volatilidade passada já entrega de graça: exigida a prever algo
*além* dela e dos controles, a TD fica muda em todas as 16 células
pré-especificadas que testamos. E até a versão descritiva está em extinção:
o gap de vol entre top-10 e bottom-10 cai de 6,1 p.p. (2009–14) para 1,2
p.p. (2015–25) — a mesma data de morte que o walk-forward condicional havia
apontado, agora confirmada por um método totalmente independente. O "canal
de risco" é sobretudo o prolongamento da reação ao anúncio, já contado na
H1; não o usamos como sinal — mas ele sobrevive como **explicação econômica
do prêmio** da carteira: quem carrega as empresas de TD alta ganha mais
correndo mais risco (§5), compensação, não almoço grátis.

**H3 — operacional.** O valor absoluto da surpresa de lucro seguinte aumenta
com a TD (t = +2,2 na amostra ampla), com a mesma concentração de regime.

**Retornos subsequentes (Tabela 6 do artigo).** Painel mensal (retorno dos 3
meses após a call; efeitos fixos de empresa e mês; controles de valor,
momentum, tamanho e reversão): coeficiente positivo nos dois sensores (t =
+2,1 LM; +1,8 FinBERT; artigo: +2,2) — consistente com o mecanismo do artigo:
o mercado penaliza no anúncio e exige retorno maior depois. Validação
walk-forward desta camada (cortes anuais 2014–2022): β futuro **positivo em
9 de 9 cortes** para a versão ponderada, com significância nos futuros
longos (t até +2,5) — e duas ressalvas declaradas: o lado passado só ganha
significância a partir de 2019, e os futuros dos cortes recentes são curtos
(t<1). Leitura: prêmio direcionalmente estável em todos os momentos, força
moderada.

**Validação temporal (matriz completa: 3 tipos de teste × 3 hipóteses,
todos os resultados reportados, inclusive os contrários).**

| teste | H1 (CAR) | H2 (vol) | H3 (\|SUE\|) |
|---|---|---|---|
| 1. carteira / realização calendar-time | quintis L-S nulos (melhor Sharpe 0,22, t=0,99) | vol da cesta alta − baixa: +0,4 a +2pp sobre ~33% (sem valor prático); FinBERT invertida (−11pp) | cesta trimestral = teste 3 |
| 2a. walk-forward anual do β condicional | 10/12 | **3/12** | 9/12 |
| 2b. walk-forward TRIMESTRAL (48 momentos) — passado / futuro na direção da hipótese | 48/48 · **40/48** | 48/48 · **15/48** | 48/48 · 32/48 |
| 3. previsão por evento sem controles (50 trimestres à parte) | 42% de acerto | 58% | 35% |
| 4. previsão por evento **condicional** (resíduo do modelo treinado no passado; o teste que casa com a hipótese) | **62% · spread −0,14% · t=−2,12** | 52% · t=+0,27 | **71% · spread +0,74% · t=+3,58** |

Duas lições estruturam a leitura. **Primeira (linha 2b):** nas três
hipóteses, o β estimado com o passado aponta a direção "esperada" em 48 de
48 momentos — em amostra corrente, as três pareceriam confirmadas. Só o
futuro as separa: H2 é contrariada em 69% dos momentos (canal extinto após
~2014) e é descartada; H1 e H3 sobrevivem. **Segunda (linhas 3 vs 4): o
teste precisa casar com a afirmação.** A versão incondicional (ordenar
empresas cruamente por TD) não prevê nada, porque mistura o sinal com
características de firma; quando a previsão usa o modelo condicional
treinado só no passado (resíduos), a H1 valida por evento fora da amostra
(t=−2,12; spread negativo em todos os anos desde 2019) e a H3 entrega o
resultado prospectivo mais forte do projeto (t=+3,58; 71% dos trimestres):
divergência de tom hoje prevê imprevisibilidade do lucro do trimestre
seguinte. Registro de método: a linha 4 não estava no desenho original —
nasceu da revisão interna; o que a defende é ser o teste teoricamente
correto da afirmação condicional, e o fato de reprovar a H2 junto (um teste
complacente não reprovaria). (Nota: cortes adjacentes compartilham dados;
as contagens medem estabilidade, não testes independentes.)

## 5. Backtest e estratégia

### 5.1 A tese executada: carteira das 10 maiores TD, mês a mês

O teste mais concreto que uma tese de investimento admite: **comprá-la**. No
fechamento de cada mês, a carteira compra em pesos iguais as **10 empresas
de maior Tone Distance** (call mais recente dos 3 meses anteriores), segura
o mês seguinte e rebalanceia. A seleção usa apenas informação disponível na
data (auditado mês a mês pelo próprio script: nenhuma call usada é posterior
ao início do mês investido); custos de 10 pontos-base por lado sobre o
turnover realizado (34%/mês). 199 meses, de fev/2009 a ago/2025:

| | Top-10 TD bruto | Top-10 TD líquido | Bottom-10 TD | Universo elegível EW | S&P 500 |
|---|---|---|---|---|---|
| Retorno acumulado | **+2.061%** | +1.783% | +959% | +1.208% | +682% |
| Retorno ao ano | **+20,4%** | +19,4% | +15,3% | +16,8% | +13,2% |
| Volatilidade a.a. | 20,6% | 20,6% | 17,4% | 17,3% | 14,9% |
| Sharpe | 1,01 | 0,96 | 0,91 | 0,99 | 0,91 |
| Drawdown máximo | −29% | −29% | −27% | −27% | −25% |

| ano | top-10 TD | bottom-10 | universo EW | S&P 500 |
|---|---|---|---|---|
| 2009 | +113,1% | +44,6% | +61,3% | +35,0% |
| 2010 | +19,2% | +10,1% | +23,1% | +12,8% |
| 2011 | +4,8% | +8,6% | +0,5% | −0,0% |
| 2012 | +39,3% | +24,6% | +20,2% | +13,4% |
| 2013 | +30,5% | +53,1% | +37,8% | +29,6% |
| 2014 | +10,3% | +5,8% | +15,6% | +11,4% |
| 2015 | +5,9% | +6,8% | −1,5% | −0,7% |
| 2016 | +6,2% | +15,5% | +17,6% | +9,5% |
| 2017 | +18,2% | +12,8% | +21,6% | +19,4% |
| 2018 | +0,4% | −14,5% | −7,2% | −6,2% |
| 2019 | +44,5% | +27,1% | +31,5% | +28,9% |
| 2020 | +11,8% | +2,7% | +15,8% | +16,3% |
| 2021 | +20,0% | +23,4% | +31,3% | +26,9% |
| 2022 | −14,1% | −0,4% | −10,4% | −19,4% |
| 2023 | +37,6% | +14,7% | +17,8% | +24,2% |
| 2024 | +18,9% | +19,5% | +15,2% | +23,3% |
| 2025* | +11,2% | +16,2% | +7,5% | +9,8% |

*(2025 até agosto. Composições conferíveis à mão: p.ex. ago/2025 = HII,
MCD, CAH, ES, MCK, MTCH, CSCO, ANET, PODD, ABNB.)*

**Como ler estes números com honestidade** — é aqui que o trabalho se separa
de um backtest de folheto. O excesso sobre o S&P 500 é **+7,1% a.a.
(t=2,60)**, com vitória em 58% dos meses e em 13 dos 17 anos. Mas a régua
justa para uma carteira equal-weight não é um índice ponderado por valor, e
sim o universo elegível equal-weight: contra ele, a seleção por TD adiciona
**+3,6% a.a. (t=1,59)** — positivo em todos os subperíodos testados (2010+:
+1,8%; 2015+: +1,9%; 2019+: +2,6% a.a.), porém nunca significativo. Ou
seja: cerca de metade do excesso sobre o índice vem do equal-weight em si,
e o tilt próprio da TD, embora consistente, dilui-se no ruído de uma
carteira de só 10 nomes — exatamente o que o efeito pequeno do painel (§4)
prevê. O Sharpe praticamente igual ao do universo (1,01 vs 0,99) diz o
resto: o retorno extra vem acompanhado de risco extra — compensação por
carregar as empresas mais tensas, a leitura econômica do próprio artigo,
não almoço grátis. A variante ponderada por palavras, melhor no painel,
empata com o universo na cauda extrema de 10 nomes (com 10 ativos, o ranking
simples captura melhor a ponta da distribuição). Séries mensais e
composições em `data/interim/sp500/port10_*.csv`; replicável por
`scripts/sp500_port10.py`.

### 5.2 Quintis long-short calendar-time (especificação congelada)

**Especificação congelada antes de rodar** (contra viés de escolha): carteira
calendar-time com tranches sobrepostas; entrada no fechamento do pregão
seguinte à call (T+1); manutenção de 63 pregões; ranking point-in-time do
sinal contra os 90 dias anteriores; quintis extremos (long TD alta / short TD
baixa — direção fixada a priori pelo artigo); custos de 5 pontos-base por
perna; amostra completa 2006–2025 sem seleção de período; benchmark S&P 500.

**Resultados líquidos** (long-short anualizado, todas as construções do
sinal sob a mesma especificação; 6 tentativas contadas para o Deflated
Sharpe): leitura A +2,0% a.a. (Sharpe 0,18); **leitura B +2,4% (Sharpe 0,22,
o melhor — e ainda assim t(Sharpe)=0,99, não significativo)**; ponderada
+0,3% (0,05); FinBERT ponderada +4,4% (0,17) — com a ressalva de que o
diagnóstico de pernas mostra correlação de só 0,6 entre long e short: o
quintil alto de FinBERT embute aposta em ações de baixa volatilidade, um
fator, não tom; histórico próprio +0,6% (0,07). Mercado no período: Sharpe
0,53. Nenhuma construção monetiza em quintis long-short.

### 5.3 O que a estratégia pode (e não pode) prometer

A bateria completa de implementação — carteira executada (5.1), quintis
long-short (5.2), previsão por evento em todos os momentos, out-of-sample
2023–25 — converge num veredito único: **ordenar ações por TD gera um tilt
comprado direcional e consistente, mas não alpha estatisticamente
demonstrável contra a régua justa em large caps.** O prêmio identificado no
painel é condicional (dentro da empresa, com controles), pequeno (~15 bps
por variação interquartil) e, no corte cruzado, disputa espaço com
características de firma (diagnóstico explícito: o quintil alto do sinal
FinBERT embute aposta em baixa volatilidade — correlação entre pernas de só
0,6).

O CORO é portanto apresentado pelo que a evidência sustenta: um **motor de
análise de eventos** cujo escore, minutos após cada call, carrega dois
conteúdos validados prospectivamente por evento (teste condicional, linha 4
da matriz): (i) **precificação no anúncio** — resíduo de CAR menor para
divergência alta (t=−2,12; negativo todos os anos desde 2019); (ii)
**incerteza do próximo lucro** — previsão da magnitude da surpresa do
trimestre seguinte (t=+3,58, o resultado prospectivo mais forte do
projeto). As aplicações coerentes com isso: condicionamento de exposição a
eventos, posicionamento de volatilidade em torno do PRÓXIMO anúncio das
empresas de divergência alta, e insumo num arcabouço multifator — não
long-short isolado por ordenação crua (reprovado e explicado). A camada de
retorno mensal também passou pela validação temporal (β futuro positivo em
9/9 cortes, força moderada — §4), sustentando a perna de tilt como
direcional. A engenharia dessas aplicações é o trabalho da entrega final.
Não há testes pendentes: toda afirmação deste relatório tem a sua
validação executada e reportada.

**Vieses tratados:** sem escolha oportunista de período (2005–2025 tudo);
execução T+1 com carimbo de hora da call (27% das calls pós-fechamento
deslocadas para o pregão seguinte — teste feito, resultado robusto);
sobrevivência: universo inclui 685 empresas históricas do índice, com a
limitação declarada de que ~77 deslistadas ficam sem preços na fonte
gratuita; custos incluídos; direção e janelas pré-fixadas pelo artigo (sem
mineração de especificação).

## 6. Análise crítica e fragilidades (declaradas)

1. **Papéis de orador reconstruídos** (~90% de acerto validado). O dado
   curado (Capital IQ) do artigo não tem esse ruído; parte da diferença entre
   nossos t e os do artigo pode vir daí.
2. **Proxies**: surpresa de lucro via yfinance (não I/B/E/S); sem
   *institutional ownership* (13F) e qualidade de accruals entre os
   controles — os demais 17 controles do artigo foram reconstruídos.
3. **Sobrevivência parcial**: ~3,4 mil eventos sem CAR por falta de preços de
   deslistadas em fonte gratuita.
4. **Efeito pequeno**: −0,15% por variação interquartil não paga custos como
   estratégia isolada de anúncio em large caps.
5. **Regime e validação temporal**: o canal de risco (H2) reprova como
   sinal incremental (extinto após ~2014, confirmado por dois métodos
   independentes — §4 e §5.1). H1 e H3 atravessam a validação temporal, mas
   cada um com sua ressalva: H1 só na forma condicional (não como ordenação
   crua) e H3, embora o resultado prospectivo mais forte no teste
   condicional (t=+3,58), tem acerto direcional apenas moderado no corte
   trimestral (32/48). O prêmio de retorno mensal só ganha significância no
   passado a partir de 2019.
6. **Ambiguidade do artigo**: a definição do centroide tem duas leituras;
   reportamos as duas (a igual-ponderada é nula; a agregada replica). Não
   escolhemos a leitura pelos resultados: a evidência textual do artigo
   (rótulo "Average Transcript Tone" da Figura 1) aponta para a agregada.

## 7. Conclusão e próximos passos

**O que o pré-relatório estabelece:** a divergência de tom entre gestores é
informação precificada — replicamos o resultado central de Angelo (2025) em
universo independente, com dados públicos, e mostramos que ele sobrevive à
troca completa do medidor de tom (léxico → neural). A contribuição própria é
dupla: o diagnóstico de *como* medir (o peso de fala separa sinal de ruído em
calls com muitos participantes) e a infraestrutura 100% replicável.

**A tese de investimento, em um parágrafo:**

> A divergência de tom entre os gestores na earnings call é informação
> precificada em dois tempos. No anúncio, o mercado penaliza a empresa cuja
> diretoria desafina (CAR condicional negativo, validado ano a ano desde
> 2019). Nos meses seguintes, carregar as empresas de maior divergência
> captura um prêmio de compensação: a carteira executada das 10 maiores TD
> rendeu +20,4% a.a. contra +13,2% do S&P 500 em 16 anos e meio — retorno
> extra pago por risco extra, coerente com a leitura econômica do artigo. E
> o mesmo sinal prevê a magnitude da surpresa do **próximo** lucro
> (t=+3,58), fazendo dele um termômetro de incerteza pré-anúncio. A
> promessa honesta: um tilt direcional consistente, cuja significância
> contra o universo equivalente pede diversificação maior que 10 nomes — um
> motor de eventos e de incerteza validado, não uma máquina de alpha
> isolada.

**Sobre a expansão small/mid caps (teste já realizado, pré-registrado):**
testamos a tese de expansão num universo de 2.250 firmas fora do S&P 500
(dataset Motley Fool, 2019–2023), com hipóteses e especificações congeladas
por pré-registro ANTES de qualquer resultado (docs/PREREG_MF.md, timestamp
em git). O painel não confirmou a hipótese: coeficientes positivos e não
significativos, com intervalo de confiança que contém tanto o efeito do S&P
quanto zero (inconclusivo por potência: janela de 3,5 anos, metade em 2021,
CAR 60% mais volátil, cobertura de preços de 69% com viés de sobrevivência).
O diagnóstico de reconciliação isolou a causa: no MESMO período 2019–23, o
efeito segue presente nas large caps (t=−1,96) — o problema é o universo ou
a qualidade dos dados dele, não a época. A fronteira segue aberta e
documentada; não a apresentamos como resultado.

**Próximos passos até a entrega final (17/08):**
1. **Desenho da integração multifator** do escore CORO (interações com
   fatores e regimes), com contagem honesta de tentativas e Deflated Sharpe;
2. Robustez final: sensibilidade a custos, análise de decay;
3. Se houver dados melhores (preços de deslistadas), revisitar a fronteira
   small/mid com o pré-registro já commitado.

**A lição que organiza o trabalho.** Nas três hipóteses, o modelo estimado
com o passado apontava a direção "esperada" em 48 de 48 momentos — em
amostra cheia, as três pareceriam confirmadas, e um relatório escrito nesse
ponto venderia uma miragem com convicção. Só a validação temporal separou o
real (H1, H3, o prêmio mensal) do artefato (H2). Esse filtro — e a
disposição de descartar um resultado bonito quando o futuro o desmente — é
o principal produto deste pré-relatório, e é o que a equipe leva para a
entrega final.

## 8. Uso de IA Generativa no processo

O uso de GenAI é **operacional e auditável**, não declaratório:

1. **Classificação de papéis por LLM (núcleo do pipeline).** Um robô por
   linha de comando (`claude -p`, modelo Haiku) classificou os 12.678
   cabeçalhos de orador em lotes, com cache versionado e reprocessamento
   determinístico (~4,6h de execução). Validação: matriz de confusão contra
   292 rótulos manuais (~90% de acerto global) e classificador determinístico
   independente para comparação (99% de concordância nos casos frequentes).
   Prompt, custos e validação documentados em `docs/GENAI_USAGE.md`.
2. **Par de engenharia e auditoria (processo inteiro).** Arquitetura do
   pipeline, código, testes, documentação e — com impacto direto no resultado
   — a auditoria da fórmula do centroide: a reimplementação independente e o
   teste das duas leituras da Eq. (1) que mudaram a conclusão central foram
   conduzidos em par com o assistente (Claude), com decisões econômicas e
   metodológicas sempre da equipe.
3. **O que NÃO chamamos de GenAI:** o FinBERT é um modelo de linguagem
   *encoder* (classificação), parte do modelo quantitativo — permitido pelo
   edital, mas contabilizado como NLP/ML, não como IA generativa.

---

*Todos os números deste relatório saem de scripts versionados no repositório
e foram reproduzidos em clone limpo. Nenhum número foi digitado à mão.*
