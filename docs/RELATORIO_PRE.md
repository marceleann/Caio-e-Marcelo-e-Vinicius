# CORO: Divergência de tom entre gestores como sinal quantitativo de investimento

Pré-relatório · Desafio Quant AI 2026 (Itaú Asset)
Equipe: Marcelo e Caio · Orientação: Profa. Nadia Cardoso Moreira
Repositório com código, dados e replicação completa:
https://github.com/marceleann/Caio-e-Marcelo-e-Vinicius

> Nome do robô em definição pela equipe (alternativas em avaliação:
> DESAFINADO, DISSONA, OUVIDOR, CORO). O conceito é único: o robô não mede
> o tom médio da empresa, e sim o desalinhamento entre as vozes que a
> representam.

---

## Sumário executivo

Toda earnings call é uma peça ensaiada, mas quem a executa são pessoas.
Este trabalho parte de uma observação simples: quando os executivos da
mesma empresa divergem no tom ao falar do mesmo trimestre, essa divergência
carrega informação que a mensagem oficial não entrega. Nós transformamos
essa observação em um sinal quantitativo, a Tone Distance (TD), calculada
minutos após a divulgação de cada transcrição, e a testamos de ponta a
ponta em 33.362 calls de 685 empresas do S&P 500 entre 2005 e 2025.

Os resultados sustentam a tese. No painel econométrico, empresas cujos
gestores divergem mais sofrem retorno anormal menor no anúncio (estatística
t de até 2,9 em módulo), entregam retornos maiores nos três meses seguintes
(t=+2,1) e apresentam surpresas de lucro mais imprevisíveis no trimestre
seguinte (t=+3,58 na previsão fora da amostra, nosso resultado preditivo
mais forte). O quadro se repete em dois medidores de tom independentes, o
dicionário Loughran-McDonald e o modelo neural FinBERT, o que afasta a
hipótese de artefato de instrumento.

Na implementação, uma carteira que compra todo mês as 10 empresas de maior
TD rendeu +20,4% ao ano, contra +13,2% do S&P 500, ao longo de 199 meses
(fevereiro de 2009 a agosto de 2025), com excesso de +7,1% ao ano (t=2,60)
e vitória em 13 dos 17 anos. O relatório também documenta o que não
funcionou: a hipótese de que o sinal prevê volatilidade futura não passou
nos testes de robustez e foi descartada, decisão que descrevemos em
detalhe porque ela atesta o rigor do processo. Tudo o que está aqui é
reproduzível a partir do repositório público com quatro comandos.

## 1. Contexto e problema

Quem acompanha teleconferências de resultados conhece a cena. O CEO abre
com a mensagem preparada, o CFO percorre os números e, quando começa a
sessão de perguntas, cada executivo responde com o tom que consegue
sustentar ao vivo. O texto é coordenado; o tom, nem sempre. Coordenar a
narrativa entre quatro ou cinco oradores é viável; coordenar o otimismo de
cada um, frase a frase, sob pergunta de analista, é muito mais difícil.

A premissa deste trabalho é que esse descompasso é informativo. Se o CFO
soa visivelmente mais cauteloso que o CEO no mesmo call, o mercado tem
diante de si um indício de desacordo interno ou de incerteza sobre o
trimestre que nenhum press release admitiria. O problema de pesquisa,
portanto: é possível medir essa divergência de forma sistemática, e ela
tem valor econômico?

Dessa premissa saem as três hipóteses que testamos, registradas antes dos
testes. Primeira: quanto maior a divergência de tom, pior a reação do
mercado no anúncio. Segunda: maior divergência indicaria maior risco
realizado depois do anúncio. Terceira: maior divergência antecipa
resultados operacionais mais imprevisíveis nos trimestres seguintes.

O produto que materializa a resposta é o robô. Ele processa a transcrição
de cada call, atribui cada fala ao seu orador, mede o tom de cada gestor e
resume o desalinhamento em um único escore por empresa-trimestre,
disponível minutos após a publicação da transcrição. Para uma mesa, isso
significa um insumo utilizável no mesmo dia do evento: como sinal de
seleção, como condicionante de exposição em torno de anúncios e como
indicador antecedente de incerteza de resultados.

## 2. Referências

A ideia de que o tom da comunicação corporativa move preços tem lastro
sólido na literatura. Loughran e McDonald (2011) construíram o dicionário
de tom específico para finanças que virou padrão da área, ao mostrar que
dicionários genéricos classificam mal a linguagem de negócios. Huang, Teoh
e Zhang (2014) documentaram que gestores administram ativamente o tom de
suas divulgações, e que o mercado reage ao componente anormal desse tom.
Lee (2016) mostrou que investidores percebem quando executivos abandonam a
espontaneidade e se apegam a roteiros durante earnings calls.

O passo que origina nossa medida é de Angelo, Johnston, Singh e Wan
(2025), que propuseram olhar não para o tom médio da empresa, mas para a
divergência de tom entre os executivos da mesma call, batizada de Tone
Distance. Nosso trabalho toma essa referência como ponto de partida e a
submete a um teste independente, com universo, dados e código
integralmente próprios. E adiciona uma extensão deliberada: repetimos toda
a análise trocando o medidor de tom, do dicionário para o FinBERT (Yang,
Uy e Huang, 2020), um modelo de linguagem treinado em textos financeiros.
A lógica da extensão é de validação cruzada de instrumento: se o fenômeno
é real, deve aparecer nos dois medidores; se é artefato do dicionário, a
troca o elimina.

## 3. Desenvolvimento: do texto bruto ao sinal

A matéria-prima são 33.362 transcrições de calls de empresas que
integraram o S&P 500 entre 2005 e 2025 (dataset público kurry/HuggingFace,
685 empresas). Preços vêm do Yahoo Finance; fundamentos trimestrais, do
SEC EDGAR, alinhados pela data de protocolo para que nenhuma informação
seja usada antes de existir publicamente; a classificação setorial segue
Fama e French (1997); a taxa livre de risco é a de Kenneth French.

O primeiro desafio prático é saber quem fala. Cada transcrição traz
milhares de falas, e o sinal exige separar gestores de analistas e
operadores. Resolvemos com um classificador em dois níveis: um modelo de
linguagem via linha de comando classificou os 12.678 cabeçalhos de orador
distintos da base, e regras determinísticas cobrem o restante. A
classificação foi auditada contra 292 rótulos manuais, com cerca de 90% de
acerto (o processo completo está na Seção 9).

Com os papéis atribuídos, o sinal se constrói em três passos. Cada gestor
da call vira um ponto num plano definido pela fração de linguagem positiva
e negativa de sua fala. Calcula-se o centro da call. A Tone Distance é a
distância média dos gestores a esse centro: zero quando todos soam igual,
alta quando as vozes se afastam. Exigimos no mínimo dois gestores por
call, e o tom de cada fala é medido duas vezes, pelo dicionário e pelo
FinBERT.

Um detalhe de construção acabou virando contribuição metodológica. A
definição do centro admite duas leituras: média simples dos gestores, em
que cada orador pesa igual, ou tom agregado do texto completo, em que cada
palavra pesa igual. A diferença não é cosmética. Em uma call da Apple de
2020, por exemplo, o profissional de relações com investidores falou
apenas 283 palavras, e as únicas 6 negativas eram o aviso legal padrão
sobre riscos e incertezas; na média simples, esse orador marginal
respondia por um terço do sinal da call. A leitura agregada, e além dela a
ponderação explícita pelo volume de fala de cada gestor, elimina essa
contaminação. Como se verá na Seção 5, os resultados enfraquecem
monotonicamente à medida que se dá peso a oradores marginais, o que
transforma o detalhe em diagnóstico. A implementação foi verificada por
reimplementação independente nas 32.387 calls elegíveis, com diferença
máxima da ordem de 10⁻¹⁶.

## 4. Método de teste

Os retornos anormais seguem o modelo de mercado: beta estimado por CAPM em
janela de 100 pregões (mínimo de 70), separada do evento por 50 pregões, e
resíduo acumulado nas janelas de 1, 2 e 5 pregões ao redor do anúncio.
Calls realizadas após o fechamento (27% da amostra) são ancoradas no
pregão seguinte, respeitando o horário real em que a informação chegou ao
mercado.

O teste central é um painel com efeitos fixos de empresa e de
ano-trimestre, erros agrupados por empresa e winsorização em 5/95, com
controles de tamanho, valor, alavancagem, rentabilidade, surpresa de
lucro, momentum e reversão. O efeito fixo de empresa importa para a
leitura: o coeficiente não compara a Apple com a Exxon, compara a Apple de
um trimestre com a própria Apple típica. O sinal mede, portanto, o desvio
da empresa em relação ao seu padrão usual de alinhamento.

Para a implementação, desenhamos dois backtests independentes, ambos com
custos e com a garantia, auditada nos scripts, de que nenhuma informação
futura entra na seleção. O primeiro executa a tese da forma mais direta
possível: ao fim de cada mês, compra em pesos iguais as 10 empresas de
maior TD (call mais recente dos três meses anteriores), carrega por um mês
e rebalanceia, pagando 10 pontos-base por lado sobre o giro. O segundo é
um long-short por quintis em formato calendar-time, com especificação
congelada antes da execução: entrada no pregão seguinte à call, manutenção
de 63 pregões, ranking contra os 90 dias anteriores, custos de 5
pontos-base por perna, amostra 2006 a 2025.

Todos os números deste relatório saem de scripts versionados no
repositório público. Um avaliador reproduz as regressões e os backtests
com quatro comandos, sem reprocessar transcrições; o procedimento foi
testado em clone limpo, com números idênticos.

## 5. Resultados econométricos

Antes das regressões, validamos a base. A distribuição da nossa TD (média
0,0086, mediana 0,0080, desvio 0,0047) é aderente à da referência (0,0079,
0,0074, 0,0048), e os controles reconstruídos batem com os publicados:
alavancagem 0,24 contra 0,235, alíquota efetiva mediana 0,22 contra 0,21,
suavização de lucros 1,6 contra 1,4. Partimos, portanto, de uma base
comparável.

A primeira hipótese se confirma na amostra completa. A tabela reporta as
estatísticas t do coeficiente da TD sobre o retorno anormal do anúncio,
nas três janelas de evento, para as três construções do sinal e os dois
medidores (amostra de 2009 em diante, cerca de 23 mil eventos e 540
empresas):

| construção do sinal | LM (dicionário) | FinBERT (neural) |
|---|---|---|
| centro = média simples | −0,7 / −0,6 / −0,3 | +0,1 / −0,2 / −0,5 |
| centro = tom agregado | −2,0 / −2,1 / −1,7 | −1,9 / −2,3 / −2,3 |
| ponderada pelo volume de fala | −2,9 / −2,7 / −2,3 | −2,4 / −2,4 / −2,5 |

(t nas janelas de 1, 2 e 5 pregões. Em termos econômicos, uma variação
interquartil de TD corresponde a algo entre 0,11% e 0,16% a menos de
retorno anormal no anúncio.)

Três leituras. A hipótese central replica quando o sinal respeita o volume
de fala de cada gestor. A progressão entre as linhas é monotônica: quanto
mais peso ao ruído dos oradores marginais, mais fraco o sinal, exatamente
o diagnóstico da Seção 3. E os dois medidores, que concordam entre si em
apenas 38%, contam a mesma história, o que afasta artefato de instrumento.

O efeito não termina no anúncio. No painel mensal, o retorno dos três
meses seguintes à call cresce com a TD (t=+2,1 no dicionário, +1,8 no
FinBERT), controlando por valor, momentum, tamanho e reversão. O quadro
combinado é economicamente coerente: o mercado penaliza a empresa
divergente no ato e cobra um prêmio para carregá-la nos meses seguintes.

A terceira hipótese entrega o resultado preditivo mais forte do projeto. A
magnitude da surpresa de lucro do trimestre seguinte cresce com a TD
(t=+2,2 no painel), e no exercício de previsão fora da amostra, em que o
modelo é treinado apenas com o passado e usado para prever eventos
futuros, a relação alcança t=+3,58, com acerto direcional em 71% dos
trimestres. Em outras palavras: a divergência de hoje avisa que o próximo
balanço vem com surpresa grande, para qualquer lado.

A segunda hipótese, a de risco, não sobreviveu, e o caminho até essa
conclusão merece registro. Na amostra completa existe associação entre TD
e volatilidade futura (t=+2,5 no dicionário). A investigação mostrou,
porém, que janelas de volatilidade que começam no dia seguinte ao anúncio
ainda contêm a reação ao próprio anúncio, e empresas de TD alta reagem
mais (que é justamente a primeira hipótese). Medida a partir do segundo
dia, fora desse eco, a associação perde significância (t=+1,3). Nos testes
de robustez, nenhuma das 16 especificações predefinidas mostrou poder de
previsão. A conclusão honesta: a TD não informa sobre risco futuro nada
que a volatilidade passada já não diga, e o canal foi descartado como
sinal. Permanece verdadeiro, como descrição, que empresas de TD alta são
em média mais voláteis, e esse fato retorna na interpretação do backtest.

Fechamos a seção com a robustez temporal, apresentada sem tecnicismo.
Reestimamos tudo usando apenas os dados disponíveis até cada data de corte
e conferimos no período posterior, além da previsão por evento fora da
amostra. O coeficiente da primeira hipótese tem o mesmo sinal negativo em
todos os cortes analisados, e a previsão fora da amostra confirma
(t=−2,12, com resultado negativo em cada um dos anos de 2019 a 2025); a
camada de retornos subsequentes mantém coeficiente positivo em todos os
cortes; a terceira hipótese sustenta o t=+3,58 já citado. Nos primeiros
anos, com menos dados acumulados, os intervalos de confiança são mais
largos, como se espera de um efeito estável estimado com precisão
crescente. A exceção é a hipótese de risco, reprovada nesses mesmos
exercícios (t=+0,27), o que motivou o descarte. O mesmo protocolo que
valida duas hipóteses reprova a terceira; o critério não é complacente.

## 6. Backtest

A implementação principal executa a tese da forma que um gestor
reconheceria: carteira comprada, regra fixa, rebalanceamento mensal. Os
números de 199 meses (fevereiro de 2009 a agosto de 2025):

| | Top-10 TD bruto | Top-10 TD líquido | Bottom-10 TD | Universo elegível EW | S&P 500 |
|---|---|---|---|---|---|
| Retorno acumulado | +2.061% | +1.783% | +959% | +1.208% | +682% |
| Retorno ao ano | +20,4% | +19,4% | +15,3% | +16,8% | +13,2% |
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

(*2025 até agosto. Giro médio de 34% ao mês. As composições são
verificáveis; agosto de 2025, por exemplo: HII, MCD, CAH, ES, MCK, MTCH,
CSCO, ANET, PODD, ABNB. Séries e composições completas em
data/interim/sp500/port10_*.csv; replicação por scripts/sp500_port10.py.)

A leitura desses números pede honestidade de atribuição, e a fazemos em
duas camadas. Contra o S&P 500, o excesso é de +7,1% ao ano com t=2,60,
vitória em 58% dos meses e em 13 dos 17 anos; é o número de manchete, e é
real. Mas uma carteira de pesos iguais não deve ser comparada apenas a um
índice ponderado por valor, e por isso construímos a régua mais exigente:
o próprio universo elegível em pesos iguais. Contra ela, a seleção por TD
adiciona +3,6% ao ano (t=1,59), positiva em todos os subperíodos que
examinamos (de 2010 em diante, +1,8%; de 2015 em diante, +1,9%; de 2019 em
diante, +2,6% ao ano), embora sem significância isolada, o que é a
consequência esperada de concentrar em 10 ativos, onde o ruído específico
de cada nome domina. A carteira espelho, com as 10 empresas de menor TD,
rende +15,3% ao ano, 5,1 pontos abaixo do top-10, na direção prevista pela
tese.

O long-short por quintis, na especificação congelada, não monetiza: os
retornos líquidos anualizados ficam entre +0,3% e +4,4% conforme a
construção, com o melhor Sharpe em 0,22, estatisticamente nulo, contra
0,53 do mercado no período. O diagnóstico de pernas explica o porquê: a
correlação entre as pontas é de apenas 0,6, e o quintil alto do sinal
FinBERT carrega exposição a ações de baixa volatilidade, um fator de
risco, não tom. Registramos as 6 construções testadas e as contabilizamos
para o ajuste de múltiplas tentativas (Deflated Sharpe, Bailey e López de
Prado, 2014) na entrega final.

O perfil de risco fecha a interpretação. O top-10 realiza volatilidade
maior que todas as réguas (20,6% contra 17,3% a 17,4%) e Sharpe semelhante
ao do universo (1,01 contra 0,99). O retorno adicional, portanto, remunera
risco adicional: é prêmio de compensação por carregar as empresas cuja
diretoria diverge, não anomalia gratuita, e isso é coerente com a
associação descritiva entre TD e volatilidade da Seção 5. Para a mesa, as
aplicações que a evidência sustenta são quatro: tilt direcional sobre um
portfólio core; condicionamento de exposição em torno de anúncios, já que
o escore existe minutos após cada call; posicionamento de volatilidade no
próximo anúncio das empresas de divergência alta, apoiado na previsão de
surpresa de lucro; e insumo em arcabouço multifator, cuja engenharia é o
objeto da entrega final.

## 7. Riscos e limitações

1. Os papéis de orador foram reconstruídos por classificador próprio,
   com cerca de 90% de acerto validado; as bases comerciais curadas da
   literatura não têm esse ruído, e parte da diferença de magnitude entre
   nossas estatísticas e as publicadas pode vir daí.
2. Algumas variáveis usam proxies: surpresa de lucro via Yahoo Finance em
   vez de I/B/E/S, e dois controles da literatura (participação
   institucional e qualidade de accruals) não foram reconstruídos; os
   demais 17, sim.
3. Cerca de 3,4 mil eventos ficaram sem retorno anormal por falta de
   preços de empresas deslistadas em fonte gratuita, uma limitação de
   sobrevivência declarada.
4. O efeito no anúncio, de cerca de 0,15% por variação interquartil, não
   paga custos como estratégia isolada de arbitragem do evento em large
   caps; por isso a implementação proposta é de tilt, condicionamento e
   volatilidade, e não de arbitragem do anúncio.
5. Com 10 ativos, o excesso sobre o universo em pesos iguais é consistente
   mas não significativo isoladamente; diversificar além de 10 nomes é o
   caminho natural da entrega final.
6. A definição do centro da call admite duas construções na literatura;
   reportamos as duas, e a escolha da agregada não foi guiada por
   resultado, e sim pela evidência textual da referência e pelo
   diagnóstico de ruído da Seção 3.

## 8. Conclusão e tese de investimento

O pré-relatório estabelece que a divergência de tom entre gestores é
informação precificada. Confirmamos o fenômeno em universo independente,
com dados públicos e código próprio; mostramos que ele sobrevive à troca
completa do medidor de tom, do dicionário ao modelo neural; e o
transformamos em estratégia implementável, com custos e auditoria de
ausência de informação futura.

A tese de investimento, em um parágrafo: a divergência de tom entre os
executivos na earnings call é informação precificada em dois tempos. No
anúncio, o mercado penaliza a empresa cuja diretoria diverge, um efeito
condicional negativo e estável no tempo. Nos meses seguintes, carregar as
empresas de maior divergência captura um prêmio de compensação: a carteira
das 10 maiores TD rendeu +20,4% ao ano contra +13,2% do S&P 500 em 16 anos
e meio, remunerando o risco adicional que o próprio sinal identifica. O
mesmo escore antecipa a magnitude da surpresa do lucro seguinte, servindo
de termômetro de incerteza antes de cada anúncio. A implementação
recomendada combina tilt direcional, condicionamento de exposição a
eventos e posicionamento de volatilidade, com a integração multifator como
objeto da entrega final.

Registramos também a fronteira que não avançou. Testamos o sinal em 2.250
firmas fora do S&P 500 (2019 a 2023), com especificações congeladas por
pré-registro antes de qualquer resultado (docs/PREREG_MF.md, com data
registrada em git). O painel resultou inconclusivo por falta de potência:
coeficientes positivos, não significativos, com janela curta e cobertura
de preços de 69%. No mesmo período, o efeito segue presente nas large caps
(t=−1,96), o que aponta a causa para o universo e a qualidade dos dados,
não para a época. A fronteira permanece documentada e não é apresentada
como resultado.

Até a entrega final (17/08), a agenda tem três itens: desenhar a
integração multifator do escore, com contagem de tentativas e Deflated
Sharpe; fechar a robustez de custos e a análise de decaimento do sinal; e,
havendo dados de preços de deslistadas, revisitar a fronteira de small e
mid caps com o pré-registro já versionado.

Uma nota de método encerra. Em amostra completa, as três hipóteses
pareciam confirmadas. Foram os exercícios de robustez temporal que
separaram os efeitos reais (precificação, retornos subsequentes e
incerteza operacional) do artefato (risco). Esse protocolo, aplicado sem
exceção e documentado no repositório, é parte central do que a equipe leva
para a entrega final.

## 9. Uso de IA generativa no processo

O uso de GenAI neste trabalho é operacional e auditável, não
declaratório. Primeiro, no núcleo do pipeline: um processo por linha de
comando (claude -p, modelo Haiku) classificou os 12.678 cabeçalhos de
orador em lotes, com cache versionado e reprocessamento determinístico,
em cerca de 4,6 horas de execução; a validação usou matriz de confusão
contra 292 rótulos manuais (cerca de 90% de acerto global) e um
classificador determinístico independente para comparação (99% de
concordância nos casos frequentes), com prompt, custos e validação
documentados em docs/GENAI_USAGE.md. Segundo, como par de engenharia e
auditoria ao longo de todo o processo: arquitetura do pipeline, código,
testes, documentação e, com impacto direto no resultado, a auditoria da
fórmula do centro da call, conduzida em par com o assistente (Claude),
com as decisões econômicas e metodológicas sempre da equipe. Por fim, uma
distinção de honestidade conceitual: o FinBERT é um modelo de linguagem
do tipo encoder, parte do modelo quantitativo; é permitido pelo edital,
mas o contabilizamos como NLP/ML, não como IA generativa.

## Referências

Angelo, Johnston, Singh e Wan (2025). Tone Distance: Managerial Tone
Divergence and Market Reaction to Earnings. The Financial Review.

Bailey, D. H., e López de Prado, M. (2014). The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.
The Journal of Portfolio Management, 40(5).

Fama, E. F., e French, K. R. (1997). Industry Costs of Equity. Journal of
Financial Economics, 43(2).

Huang, X., Teoh, S. H., e Zhang, Y. (2014). Tone Management. The
Accounting Review, 89(3).

Lee, J. (2016). Can Investors Detect Managers' Lack of Spontaneity?
Adherence to Predetermined Scripts during Earnings Conference Calls. The
Accounting Review, 91(1).

Loughran, T., e McDonald, B. (2011). When Is a Liability Not a Liability?
Textual Analysis, Dictionaries, and 10-Ks. The Journal of Finance, 66(1).

Yang, Y., Uy, M. C. S., e Huang, A. (2020). FinBERT: A Pretrained Language
Model for Financial Communications. arXiv:2006.08097.

---

Todos os números deste relatório saem de scripts versionados no
repositório e foram reproduzidos em clone limpo. Nenhum número foi
digitado à mão.
