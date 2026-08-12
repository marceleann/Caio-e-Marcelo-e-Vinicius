# CORO: Divergência de tom entre gestores como sinal quantitativo de investimento

Pré-relatório · Desafio Quant AI 2026 (Itaú Asset)
Equipe: Marcelo e Caio · Orientação: Profa. Nadia Cardoso Moreira
Repositório com código, dados e replicação completa:
https://github.com/marceleann/Caio-e-Marcelo-e-Vinicius

> O nome CORO vem da imagem que resume a estratégia: numa earnings call,
> os executivos formam um coro que ensaia uma mensagem única, e o robô
> mede o desalinhamento dessas vozes. Quando o coro desafina, há
> informação: não medimos o tom médio da empresa, e sim a divergência
> entre as vozes que a representam.

---

## Sumário executivo

Este pré-relatório apresenta a construção e a validação de um sinal
quantitativo extraído das teleconferências de resultados: a divergência de
tom entre os executivos de uma mesma empresa, medida pela Tone Distance
(TD). A motivação parte de uma observação simples do dia a dia do mercado.
Embora a mensagem de uma earnings call seja ensaiada, quem a executa são
pessoas, e quando os executivos divergem no tom ao falar do mesmo
trimestre, essa divergência pode carregar informação que o comunicado
oficial não entrega. Nesse sentido, foram processadas 33.362 transcrições
de calls de 685 empresas do S&P 500, entre 2005 e 2025, com o objetivo de
verificar se essa informação é precificada.

Os resultados confirmam a tese central. Empresas cujos gestores divergem
mais apresentam retorno anormal menor no dia do anúncio, retornos maiores
nos três meses seguintes e surpresas de lucro mais imprevisíveis no
trimestre posterior, sendo este último o resultado preditivo mais forte do
estudo. Além disso, o mesmo quadro se repete em dois medidores de tom
independentes, o dicionário Loughran-McDonald e o modelo neural FinBERT,
o que afasta a possibilidade de o achado ser um artefato do instrumento de
medida.

Na implementação, a carteira que compra mensalmente as 10 empresas de
maior TD rendeu +20,4% ao ano, contra +13,2% do S&P 500, ao longo de 199
meses (fevereiro de 2009 a agosto de 2025), com ganho em 11 dos 17 anos.
Por fim, o relatório também documenta o que não funcionou: a hipótese de
que o sinal prevê volatilidade futura não resistiu aos testes de robustez
e foi descartada, decisão que optamos por detalhar no texto justamente por
evidenciar o rigor do processo de validação. Todos os resultados podem ser
reproduzidos a partir do repositório público com quatro comandos.

## 1. Contexto e problema

Quem acompanha teleconferências de resultados conhece a cena. O CEO abre
com a mensagem preparada, o CFO percorre os números e, quando começa a
sessão de perguntas e respostas, cada executivo responde com o tom que
consegue sustentar ao vivo. O texto é coordenado, mas o tom nem sempre
acompanha, uma vez que alinhar a narrativa entre quatro ou cinco oradores
é viável, enquanto alinhar o otimismo de cada um, frase a frase, sob
pergunta de analista, é consideravelmente mais difícil.

Diante desse cenário, a premissa do trabalho é que esse descompasso
carrega informação. Um CFO que soa visivelmente mais cauteloso que o CEO
na mesma call representa um indício de desacordo interno, ou de incerteza
sobre o trimestre, que dificilmente apareceria em um press release. Dessa
forma, duas perguntas estruturam a pesquisa: é possível medir essa
divergência de forma sistemática? E, uma vez medida, ela tem valor
econômico?

Dessas perguntas derivam as três hipóteses do estudo, registradas antes de
qualquer teste:

- H1 (precificação): quanto maior a divergência de tom em uma call, menor
  o retorno anormal da empresa na janela do anúncio.
- H2 (risco): quanto maior a divergência, maior a volatilidade realizada
  da ação após o anúncio.
- H3 (operacional): quanto maior a divergência, mais imprevisível o
  resultado do trimestre seguinte.

Para responder a essas perguntas, desenvolveu-se o CORO, o robô que dá
identidade ao projeto. Ele lê a transcrição de cada call, atribui cada fala ao seu
orador, mede o tom de cada gestor e resume o desalinhamento em um escore
por empresa-trimestre, disponível minutos após a publicação da
transcrição. Na prática, isso se traduz em um insumo utilizável no mesmo
dia do evento, seja como sinal de seleção de ativos, seja como
condicionante de exposição em torno de anúncios, seja ainda como
termômetro antecedente de incerteza de resultados.

## 2. Referencial teórico

A ideia de que o tom da comunicação corporativa move preços é bem
estabelecida na literatura de finanças textuais. Loughran e McDonald
(2011) mostraram que dicionários genéricos de sentimento leem mal a
linguagem de negócios, uma vez que termos como liability são negativos no
uso comum e neutros em finanças, e construíram o dicionário específico que
se tornou padrão da área. Na mesma linha, Huang, Teoh e Zhang (2014)
documentaram que gestores administram ativamente o tom de suas
divulgações e que o mercado reage ao componente anormal desse tom.
Ademais, Lee (2016) demonstrou que investidores percebem, e penalizam,
executivos que abandonam a espontaneidade e se apegam a roteiros durante
as earnings calls.

Nesse contexto, a medida utilizada neste trabalho vem de Angelo,
Johnston, Singh e Wan (2025). Em vez da pergunta clássica sobre o tom
médio da empresa, os autores investigam o quanto os executivos de uma
mesma call divergem entre si, dando origem à Tone Distance. O argumento
econômico é o mesmo apresentado na Seção 1: a mensagem se ensaia, mas o
tom fino de cada orador não, de modo que a divergência funciona como um
vazamento de informação privada.

Com base nessa referência, o presente estudo realiza um teste
independente, com universo, dados e código próprios, e acrescenta uma
extensão deliberada: toda a análise foi refeita trocando o medidor de
tom, do dicionário para o FinBERT (Yang, Uy e Huang, 2020), um modelo de
linguagem treinado em textos financeiros que classifica cada sentença
pelo contexto, e não por palavras isoladas. A troca serve como
contraprova do instrumento, já que um fenômeno real deve aparecer nos
dois medidores, enquanto um artefato de dicionário desapareceria na
passagem de um para o outro.

## 3. Dados e construção do sinal

A amostra é composta por 33.362 transcrições de calls de empresas que
integraram o S&P 500 entre 2005 e 2025 (dataset público
kurry/HuggingFace, 685 empresas, incluindo as que saíram do índice, o que
reduz o viés de sobrevivência). Os preços vêm do Yahoo Finance, enquanto
os fundamentos trimestrais vêm do SEC EDGAR, alinhados pela data de
protocolo, de modo que nenhuma informação é utilizada antes de existir
publicamente. A classificação setorial segue Fama e French (1997), e a
taxa livre de risco é a da base de Kenneth French.

O primeiro obstáculo prático foi identificar quem fala. Uma transcrição
contém milhares de falas, e o sinal exige separar gestores de analistas e
operadores. Para isso, montou-se um classificador em dois níveis: um
modelo de linguagem, acionado via linha de comando, classificou os 12.678
cabeçalhos de orador distintos da base, e regras determinísticas cobrem o
restante. O resultado foi auditado contra 292 rótulos feitos à mão,
alcançando cerca de 90% de acerto (o processo completo está descrito na
Seção 9).

Com os papéis atribuídos, o cálculo do sinal ocorre em três passos.
Primeiramente, cada gestor i da call é representado por um vetor
g(i) = (p(i), n(i)), em que p e n são as frações de linguagem positiva e
negativa da sua fala. Em seguida, calcula-se o centro c da call. Por fim,
a Tone Distance é obtida como a média das distâncias euclidianas dos
gestores a esse centro: TD = (1/N) Σ ‖g(i) − c‖. A medida vale zero
quando todos soam igual e cresce conforme as vozes se afastam. Exige-se
um mínimo de dois gestores por call, e cada fala é medida duas vezes, uma
pelo dicionário e outra pelo FinBERT.

Vale destacar que um detalhe dessa construção acabou rendendo a principal
contribuição metodológica do trabalho: a definição do centro c. Existem
duas leituras possíveis. Na média simples, c é a média dos vetores dos
gestores, e cada orador pesa igual, tenha falado uma hora ou trinta
segundos. No tom agregado, c é o tom do texto completo da call, e cada
palavra pesa igual. A diferença parece cosmética, mas não é, e um caso
concreto ilustra o problema: em uma call da Apple de 2020, o profissional
de relações com investidores falou apenas 283 palavras, das quais as
únicas 6 classificadas como negativas pertenciam ao aviso legal padrão
sobre riscos e incertezas, e, ainda assim, na média simples esse orador
marginal respondia por um terço do sinal da call inteira. O tom agregado
corrige essa contaminação, e a ponderação explícita de cada gestor pelo
volume de fala leva a correção ao limite. Como será demonstrado na Seção
5, os resultados enfraquecem gradualmente conforme se devolve peso aos
oradores marginais, o que transforma o detalhe em diagnóstico. A
implementação foi verificada por reimplementação independente nas 32.387
calls elegíveis, com diferença máxima da ordem de 10⁻¹⁶, além de um
exemplo auditável manualmente.

## 4. Método de teste

Dois conceitos sustentam a parte empírica e merecem definição antes dos
números. O primeiro é o retorno anormal acumulado (CAR), que corresponde
à diferença entre o retorno observado da ação e o retorno que o CAPM
previa dado o movimento do mercado, acumulada ao longo da janela do
evento. O beta de cada evento é estimado em uma janela de 100 pregões
(mínimo de 70), separada do evento por 50 pregões para não contaminar a
estimativa, e o resíduo é acumulado nas janelas de 1, 2 e 5 pregões ao
redor do anúncio. Além disso, as calls realizadas após o fechamento do
mercado (26% da amostra) são ancoradas no pregão seguinte, respeitando o
momento em que a informação de fato chegou ao investidor.

O segundo conceito é o painel com efeitos fixos, que constitui a espinha
dorsal dos testes. A regressão atribui um intercepto próprio a cada
empresa e a cada trimestre, e controla por tamanho, valor, alavancagem,
rentabilidade, surpresa de lucro, momentum e reversão, com erros-padrão
agrupados por empresa e winsorização em 5/95, procedimento que trunca os
5% mais extremos de cada variável para conter o efeito de outliers. O
efeito fixo de empresa muda a natureza da comparação, uma vez que o
coeficiente não compara a Apple com a Exxon, e sim a Apple de um
trimestre com a própria Apple típica. O que se mede, portanto, é o efeito
de a empresa divergir mais do que o seu padrão usual.

Para a leitura dos resultados, reporta-se sempre a estatística t de cada
coeficiente, que expressa quantos erros-padrão o efeito estimado dista de
zero, sendo que, pela convenção usual, valores acima de 2 em módulo
indicam significância ao nível de 5%.

Na etapa de implementação, foram desenhados dois backtests
independentes, ambos com custos de transação e com uma garantia auditada
nos próprios scripts: nenhuma informação futura entra na seleção. O
primeiro executa a tese da forma mais direta possível: ao fim de cada
mês, compra em pesos iguais as 10 empresas de maior TD, com base na call
mais recente dos três meses anteriores, carrega a posição por um mês,
rebalanceia e paga 10 pontos-base por lado sobre o giro. O segundo é um
long-short por quintis, que compra o quinto de maior TD e vende o quinto
de menor, em formato calendar-time com tranches sobrepostas e
especificação congelada antes da execução (entrada no pregão seguinte à
call, manutenção de 63 pregões, ranking contra os 90 dias anteriores,
custos de 5 pontos-base por perna, amostra de 2006 a 2025).

Por fim, cabe registrar que todos os números do relatório saem de
scripts versionados no repositório público. Qualquer avaliador reproduz
as regressões e os backtests com quatro comandos, sem reprocessar
transcrições; o procedimento foi testado em clone limpo e os números
saíram idênticos.

## 5. Resultados econométricos

Antes das regressões, verificou-se se a base reconstruída é comparável à
da literatura. A distribuição da TD calculada (média 0,0086, mediana
0,0080, desvio 0,0047) fica próxima da referência (0,0079, 0,0074,
0,0048), e os controles batem com os publicados: alavancagem de 0,24
contra 0,235, alíquota efetiva mediana de 0,22 contra 0,21 e suavização
de lucros mediana de 1,6 contra 1,4. O ponto de partida, portanto, é
comparável.

### 5.1 H1, precificação no anúncio: confirmada

Se a divergência de tom revela informação negativa, o retorno anormal do
anúncio deve cair com a TD, tudo o mais constante. A tabela reporta as
estatísticas t do coeficiente da TD sobre o CAR nas três janelas de
evento, para as três construções do sinal e os dois medidores (amostra de
2009 em diante, com cerca de 23,6 mil eventos). Cada célula
responde à mesma pergunta: medindo desta forma, a divergência derruba o
retorno do anúncio?

| construção do sinal | LM (dicionário) | FinBERT (neural) |
|---|---|---|
| centro = média simples | −0,7 / −0,7 / −0,4 | +0,1 / −0,3 / −0,5 |
| centro = tom agregado | −2,0 / −2,1 / −1,7 | −1,9 / −2,3 / −2,3 |
| ponderada pelo volume de fala | −2,8 / −2,7 / −2,3 | −2,3 / −2,4 / −2,5 |

(t nas janelas de 1, 2 e 5 pregões. Em termos econômicos, pular do
quartil inferior ao superior da TD custa entre 0,11% e 0,16% de retorno
anormal no anúncio.)

A hipótese se confirma na amostra completa sempre que o sinal respeita o
volume de fala de cada gestor, como mostram as duas últimas linhas: a
ponderada é significativa em todas as janelas e nos dois medidores, e a
agregada, na maior parte delas. Além disso, o enfraquecimento é
gradual de baixo para cima, ou seja, quanto mais peso se dá ao ruído dos
oradores marginais, mais o sinal se dilui, exatamente como antecipado na
Seção 3. Por fim, as colunas se espelham: dois medidores que concordam
entre si em apenas 38% chegam ao mesmo desenho, o que enfraquece bastante
a explicação por artefato de instrumento.

### 5.2 Retornos subsequentes: confirmados

O efeito, contudo, não termina no anúncio. Em painel mensal, o retorno
dos três meses seguintes à call foi regredido contra a TD, com efeitos
fixos de empresa e mês e controles de valor, momentum, tamanho e
reversão, e o coeficiente saiu positivo nos dois medidores (t=+2,0 no
dicionário e +2,0 no FinBERT). Em conjunto com a H1, o desenho se fecha
de forma economicamente coerente: o mercado penaliza a empresa divergente
no momento do anúncio e passa a exigir retorno maior para carregá-la nos
meses seguintes, configurando o padrão clássico de compensação por risco.

### 5.3 H3, incerteza operacional: confirmada

Se a divergência reflete incerteza interna sobre o negócio, ela deve
antecipar surpresas de lucro maiores, em qualquer direção, razão pela
qual a variável testada é o valor absoluto da surpresa. No painel
completo, o coeficiente da TD é positivo (t=+2,5). Ademais, no exercício
de previsão fora da amostra, em que o modelo é treinado apenas com o
passado e tenta prever eventos que nunca viu, a relação alcança t=+3,58,
com acerto direcional em 71% dos trimestres, o que faz deste o resultado
preditivo mais forte do projeto. Em linguagem de mesa: a divergência de
hoje avisa que o próximo balanço tende a vir com surpresa grande.

### 5.4 H2, risco: descartada como sinal

O caminho até o descarte merece ser contado, pois ilustra o método. Na
amostra completa, existe associação entre TD e volatilidade futura
(t=+2,5 no dicionário). Entretanto, a investigação revelou um problema de
medição: janelas de volatilidade que começam no dia seguinte ao anúncio
ainda carregam a reação ao próprio anúncio, e empresas de TD alta reagem
mais, o que é justamente a H1. Quando a volatilidade é medida a partir do
segundo dia, fora desse eco, a associação perde significância (t=+1,3).
Diante disso, foram rodadas 16 especificações predefinidas, combinando a
presença ou não do eco, dois horizontes, nível e logaritmo e as duas
construções do sinal, e nenhuma delas mostrou poder de previsão.
Conclui-se que a TD não informa sobre risco futuro nada que a
volatilidade passada já não diga, e o canal foi retirado do conjunto de
sinais. Permanece verdadeira, como descrição, a constatação de que
empresas de TD alta são em média mais voláteis, e esse fato retorna na
interpretação do backtest.

### 5.5 Robustez e estabilidade temporal

Um resultado de amostra completa pode, em tese, ser fruto de um único
subperíodo favorável. Para verificar essa possibilidade, as estimações
foram refeitas usando apenas os dados disponíveis até cada data de corte,
com conferência no período posterior, além da previsão por evento fora da
amostra já descrita. O coeficiente da H1 mantém o sinal negativo em todos
os cortes analisados, e a previsão fora da amostra confirma o efeito
(t=−2,12, negativo em cada um dos anos de 2019 a 2025). Da mesma forma, a
camada de retornos subsequentes mantém coeficiente positivo em todos os
cortes, e a H3 sustenta o t=+3,58 citado. Nos primeiros anos, com menos
dados acumulados, os intervalos de confiança são naturalmente mais
largos, comportamento esperado de um efeito estável estimado com precisão
crescente. A exceção é a H2, reprovada nesses mesmos exercícios
(t=+0,27), o que motivou o descarte. Cabe destacar, por fim, que o mesmo
protocolo que valida duas hipóteses reprova a terceira, o que dá
confiança de que o critério não é complacente.

## 6. Backtest

A implementação principal executa a tese da forma que um gestor
reconheceria: carteira comprada, regra fixa e rebalanceamento mensal.
Para a leitura das tabelas, o índice de Sharpe é o retorno em excesso por
unidade de volatilidade (quanto maior, melhor o retorno ajustado a
risco), e o drawdown máximo é a maior queda acumulada do valor da
carteira no período. Os números cobrem 199 meses, de fevereiro de 2009 a
agosto de 2025:

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

(*2025 até agosto. Giro médio de 35% ao mês. As composições são
verificáveis; agosto de 2025, por exemplo: HII, MCD, CAH, ES, MCK, MTCH,
CSCO, ANET, PODD, ABNB. Séries e composições completas em
data/interim/sp500/port10_*.csv; replicação por scripts/sp500_port10.py.)

A leitura desses números foi feita em duas camadas. Contra o S&P 500, o
excesso é de +7,1% ao ano com t=2,60, com vitória em 58% dos meses e em
11 dos 17 anos, sendo este o número de manchete da estratégia. Contudo,
um leitor técnico notará uma sutileza: carteiras de pesos iguais tendem a
superar índices ponderados por valor simplesmente por darem mais espaço a
empresas menores. Para isolar o mérito da seleção, construiu-se a régua
mais exigente disponível, o próprio universo elegível em pesos iguais.
Contra essa régua, a TD adiciona +3,6% ao ano (t=1,59), com contribuição
positiva em todos os subperíodos analisados (+1,8% de 2010 em diante,
+1,9% de 2015 em diante e +2,6% de 2019 em diante), embora sem
significância isolada. Essa ausência de significância tem explicação
mecânica, uma vez que, com apenas 10 ativos, o ruído específico de cada
nome domina a variância da carteira e dilui um efeito que o painel estima
em cerca de 15 pontos-base por variação interquartil. Em paralelo, a
carteira espelho, composta pelas 10 empresas de menor TD, rende +15,3% ao
ano, 5,1 pontos abaixo do top-10, na direção prevista pela tese.

Ademais, a tabela anual permite identificar os cenários favoráveis e
desfavoráveis à estratégia. O top-10 destaca-se em anos de estresse ou de
recuperação, como 2009, 2012, 2018, 2019 e 2023, chegando a fechar 2018
em alta de 0,4% enquanto o índice caía 6,2%, e a cair menos que o S&P 500
em 2022 (−14,1% contra −19,4%). Em contrapartida, a carteira fica para
trás nos anos de alta ampla do mercado, como 2016 e 2021, quando
concentrar em 10 nomes rende menos do que carregar o universo inteiro.

O long-short por quintis, por sua vez, não monetiza. Na especificação
congelada, os retornos líquidos anualizados ficam entre +0,3% e +4,4%
conforme a construção, com o melhor Sharpe em 0,22, estatisticamente
nulo, contra 0,53 do mercado no período. O diagnóstico de pernas explica
o motivo: a correlação entre a ponta comprada e a vendida é de apenas
0,6, e o quintil alto do sinal FinBERT carrega exposição a ações de baixa
volatilidade, ou seja, um fator de risco conhecido, e não tom. As 6
construções testadas foram registradas e entram na contagem de tentativas
para o ajuste de múltiplos testes (Deflated Sharpe, Bailey e López de
Prado, 2014) na entrega final.

O perfil de risco fecha a interpretação. O top-10 realiza volatilidade
maior que todas as réguas (20,6% contra 17,3% a 17,4%) e Sharpe parecido
com o do universo (1,01 contra 0,99), o que indica que o retorno extra
remunera risco extra: trata-se de um prêmio de compensação por carregar
as empresas cuja diretoria diverge, coerente com a associação descritiva
da Seção 5.4, e não de uma anomalia gratuita. Dessa forma, a evidência
sustenta quatro usos práticos para uma mesa: o tilt direcional sobre um
portfólio core; o condicionamento de exposição em torno de anúncios, já
que o escore fica pronto minutos após cada call; o posicionamento de
volatilidade no próximo anúncio das empresas de divergência alta, apoiado
na previsão de surpresa de lucro; e o uso como insumo em um arcabouço
multifator, cuja engenharia é o objeto da entrega final.

## 7. Riscos e limitações

Como toda pesquisa empírica, este trabalho possui limitações que merecem
registro. A primeira diz respeito aos papéis de orador, que foram
reconstruídos por um classificador próprio com cerca de 90% de acerto
validado. As bases comerciais curadas utilizadas na literatura não têm
esse ruído, e parte da diferença de magnitude entre as estatísticas aqui
reportadas e as publicadas pode ter origem nesse ponto.

Ademais, algumas variáveis dependem de proxies. A surpresa de lucro vem
do Yahoo Finance, e não do consenso I/B/E/S, e dois controles presentes
na literatura, a participação institucional e a qualidade de accruals,
não foram reconstruídos, embora os outros 17 tenham sido. Na mesma linha,
cerca de 3 mil eventos ficaram sem retorno anormal por falta de preços
de empresas deslistadas em fonte gratuita, o que configura uma limitação
de sobrevivência que preferimos declarar a esconder.

No campo da implementação, o efeito no anúncio, da ordem de 0,15% por
variação interquartil, não paga custos como estratégia isolada de
arbitragem do evento em large caps, razão pela qual a proposta se
concentra em tilt, condicionamento e volatilidade. Além disso, com apenas
10 ativos, o excesso sobre o universo em pesos iguais é consistente, mas
não significativo isoladamente, de modo que diversificar além de 10 nomes
é o caminho natural da entrega final.

Por fim, a definição do centro da call admite duas construções na
literatura. Ambas foram reportadas, e a escolha da agregada não foi
guiada por resultado, e sim pela evidência textual da referência e pelo
diagnóstico de ruído apresentado na Seção 3.

## 8. Conclusão e tese de investimento

O presente trabalho teve como objetivo verificar se a divergência de tom
entre os gestores de uma mesma earnings call constitui informação
precificada e, em caso positivo, se ela pode ser transformada em
estratégia de investimento. As evidências apontam que sim. O fenômeno foi
confirmado em universo independente, com dados públicos e código próprio,
sobreviveu à troca completa do medidor de tom e se materializou em uma
estratégia implementável, com custos e auditoria de ausência de
informação futura.

A tese de investimento cabe em um parágrafo. A divergência de tom entre
os executivos na earnings call é informação precificada em dois tempos.
No anúncio, o mercado penaliza a empresa cuja diretoria diverge, um
efeito condicional negativo e estável no tempo. Nos meses seguintes,
carregar as empresas de maior divergência captura um prêmio de
compensação: a carteira das 10 maiores TD rendeu +20,4% ao ano contra
+13,2% do S&P 500 em 16 anos e meio, remunerando o risco adicional que o
próprio sinal identifica. Além disso, o mesmo escore antecipa a magnitude
da surpresa do lucro seguinte, servindo de termômetro de incerteza antes
de cada anúncio. A implementação recomendada combina tilt direcional,
condicionamento de exposição a eventos e posicionamento de volatilidade,
com a integração multifator como objeto da entrega final.

Cabe ainda registrar a fronteira que não avançou. O sinal foi testado em
2.250 firmas fora do S&P 500 (2019 a 2023), com especificações congeladas
por pré-registro antes de qualquer resultado (docs/PREREG_MF.md, com data
registrada em git). O painel resultou inconclusivo por falta de potência
estatística, com coeficientes positivos, não significativos, janela curta
e cobertura de preços de 69%. Entretanto, no mesmo período, o efeito
segue presente nas large caps (t=−1,96), o que aponta a causa para o
universo e a qualidade dos dados, e não para a época. A fronteira
permanece documentada e não é apresentada como resultado.

Até a entrega final (17/08), a agenda contempla três itens: o desenho da
integração multifator do escore, com contagem de tentativas e Deflated
Sharpe; a robustez final de custos e a análise de decaimento do sinal; e,
havendo dados de preços de deslistadas, a revisita à fronteira de small e
mid caps com o pré-registro já versionado.

Encerra-se com uma nota de método. Em amostra completa, as três hipóteses
pareciam confirmadas, e foram os exercícios de robustez temporal que
separaram os efeitos reais (precificação, retornos subsequentes e
incerteza operacional) do artefato (risco). Esse protocolo, aplicado sem
exceção e documentado no repositório, é parte central do que a equipe
leva para a entrega final.

## 9. Uso de IA generativa no processo

O uso de GenAI neste trabalho é operacional e auditável, não
declaratório. No núcleo do pipeline, um processo por linha de comando
(claude -p, modelo Haiku) classificou os 12.678 cabeçalhos de orador em
lotes, com cache versionado e reprocessamento determinístico, em cerca de
4,6 horas de execução. A validação usou matriz de confusão contra 292
rótulos manuais, com cerca de 90% de acerto global, além de um
classificador determinístico independente para comparação, com 99% de
concordância nos casos frequentes; prompt, custos e validação estão
documentados em docs/GENAI_USAGE.md. Ao longo de todo o processo, o
assistente (Claude) atuou como par de engenharia e auditoria, apoiando a
arquitetura do pipeline, o código, os testes, a documentação e, com
impacto direto no resultado, a auditoria da fórmula do centro da call,
sempre com as decisões econômicas e metodológicas a cargo da equipe. Por
fim, vale uma distinção conceitual: o FinBERT é um modelo de linguagem do
tipo encoder, parte do modelo quantitativo, permitido pelo edital, mas
contabilizado como NLP/ML, e não como IA generativa.

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

Yang, Y., Uy, M. C. S., e Huang, A. (2020). FinBERT: A Pretrained
Language Model for Financial Communications. arXiv:2006.08097.

---

Todos os números deste relatório saem de scripts versionados no
repositório e foram reproduzidos em clone limpo. Nenhum número foi
digitado à mão.
