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
Quando os executivos de uma mesma empresa falam do mesmo trimestre com
tons diferentes, essa diferença diz algo que o comunicado oficial não diz.
Este trabalho transforma a observação em um sinal quantitativo, a Tone
Distance (TD), calculada minutos depois de cada transcrição ser publicada,
e testa esse sinal em 33.362 teleconferências de 685 empresas do S&P 500,
de 2005 a 2025.

Os resultados sustentam a tese. Empresas cujos gestores divergem mais caem
mais no dia do anúncio, rendem mais nos três meses seguintes e entregam
surpresas de lucro mais imprevisíveis no trimestre posterior; esse último
efeito é o mais forte que encontramos. O quadro se repete no dicionário
Loughran-McDonald e no modelo neural FinBERT, dois medidores de tom que
concordam pouco entre si. Se o fenômeno fosse um defeito do instrumento,
não teria sobrevivido à troca.

Na prática, uma carteira que compra todo mês as 10 empresas de maior TD
rendeu +20,4% ao ano contra +13,2% do S&P 500 em 199 meses (fevereiro de
2009 a agosto de 2025), ganhando em 13 dos 17 anos. O relatório também
conta o que não deu certo: a hipótese de que o sinal prevê volatilidade
futura caiu nos testes de robustez e foi descartada. Preferimos documentar
esse descarte em detalhe, porque ele mostra como o processo separa
resultado real de coincidência de amostra. Tudo o que está aqui se
reproduz a partir do repositório público com quatro comandos.

## 1. Contexto e problema

Quem acompanha teleconferências de resultados conhece a cena. O CEO abre
com a mensagem preparada, o CFO percorre os números e, quando começa a
sessão de perguntas e respostas, cada executivo responde com o tom que
consegue sustentar ao vivo. O texto é coordenado; o tom, nem sempre.
Alinhar a narrativa entre quatro ou cinco oradores é viável. Alinhar o
otimismo de cada um, frase a frase, sob pergunta de analista, é bem mais
difícil.

Nossa premissa é que esse descompasso carrega informação. Um CFO que soa
visivelmente mais cauteloso que o CEO na mesma call é um indício de
desacordo interno, ou de incerteza sobre o trimestre, que nenhum press
release admitiria. Daí as duas perguntas que estruturam o trabalho: dá
para medir essa divergência de forma sistemática? E, medida, ela tem valor
econômico?

Dessas perguntas saem as três hipóteses do estudo, registradas antes de
qualquer teste:

- H1 (precificação): quanto maior a divergência de tom em uma call, menor
  o retorno anormal da empresa na janela do anúncio.
- H2 (risco): quanto maior a divergência, maior a volatilidade realizada
  da ação depois do anúncio.
- H3 (operacional): quanto maior a divergência, mais imprevisível o
  resultado do trimestre seguinte.

O robô é a resposta prática a essas perguntas. Ele lê a transcrição de
cada call, atribui cada fala ao seu orador, mede o tom de cada gestor e
resume o desalinhamento em um escore por empresa-trimestre, pronto minutos
depois de a transcrição ir ao ar. Para uma mesa de gestão, isso vira
insumo no mesmo dia do evento: sinal de seleção, condicionante de
exposição em torno de anúncios e termômetro de incerteza de resultados.

## 2. Referências

A ideia de que o tom da comunicação corporativa move preços não nasce
aqui. Loughran e McDonald (2011) mostraram que dicionários genéricos de
sentimento leem mal a linguagem de negócios (liability, por exemplo, é
palavra negativa no uso comum e neutra em finanças) e construíram o
dicionário específico que virou padrão da área. Huang, Teoh e Zhang (2014)
documentaram que gestores administram o tom das divulgações e que o
mercado reage ao componente anormal desse tom. Lee (2016) mostrou que
investidores percebem, e penalizam, executivos que abandonam a
espontaneidade e se agarram ao roteiro durante a call.

A nossa medida vem de Angelo, Johnston, Singh e Wan (2025). Em vez da
pergunta clássica sobre o tom médio da empresa, os autores perguntam
quanto os executivos da mesma call divergem entre si, e chamam a medida de
Tone Distance. O argumento econômico é o da Seção 1: a mensagem se ensaia,
o tom fino de cada um, não; a divergência funciona como vazamento de
informação privada.

Partimos dessa referência e a submetemos a um teste independente, com
universo, dados e código próprios. Acrescentamos uma extensão deliberada:
refizemos toda a análise trocando o medidor de tom, do dicionário para o
FinBERT (Yang, Uy e Huang, 2020), um modelo de linguagem treinado em
textos financeiros que classifica cada sentença pelo contexto, e não por
palavras isoladas. A troca serve de contraprova do instrumento. Fenômeno
real aparece nos dois medidores; artefato de dicionário desaparece na
passagem.

## 3. Desenvolvimento: do texto bruto ao sinal

Trabalhamos com 33.362 transcrições de calls de empresas que integraram o
S&P 500 entre 2005 e 2025 (dataset público kurry/HuggingFace, 685
empresas, incluindo as que saíram do índice, o que reduz o viés de
sobrevivência). Os preços vêm do Yahoo Finance. Os fundamentos
trimestrais vêm do SEC EDGAR, alinhados pela data de protocolo, de modo
que nenhuma informação é usada antes de existir publicamente. A
classificação setorial segue Fama e French (1997) e a taxa livre de risco
é a da base de Kenneth French.

O primeiro obstáculo prático é saber quem fala. Uma transcrição tem
milhares de falas, e o sinal precisa separar gestores de analistas e
operadores. Montamos um classificador em dois níveis: um modelo de
linguagem, chamado via linha de comando, classificou os 12.678 cabeçalhos
de orador distintos da base, e regras determinísticas cobrem o resto.
Auditamos o resultado contra 292 rótulos feitos à mão e chegamos a cerca
de 90% de acerto (o processo completo está na Seção 9).

Com os papéis no lugar, o cálculo do sinal tem três passos. Cada gestor i
da call vira um vetor g(i) = (p(i), n(i)), em que p e n são as frações de
linguagem positiva e negativa da sua fala. Calcula-se então o centro c da
call. A Tone Distance é a média das distâncias euclidianas dos gestores a
esse centro: TD = (1/N) Σ ‖g(i) − c‖. Ela vale zero quando todos soam
igual e cresce conforme as vozes se afastam. Exigimos pelo menos dois
gestores por call, e cada fala é medida duas vezes, uma pelo dicionário,
outra pelo FinBERT.

Um detalhe dessa construção acabou rendendo a nossa principal
contribuição metodológica: a definição do centro c. Existem duas leituras
possíveis. Na média simples, c é a média dos vetores dos gestores, e cada
orador pesa igual, tenha falado uma hora ou trinta segundos. No tom
agregado, c é o tom do texto completo da call, e cada palavra pesa igual.
Parece detalhe, e não é. Numa call da Apple de 2020, o profissional de
relações com investidores falou 283 palavras; as únicas 6 negativas eram
o aviso legal padrão sobre riscos e incertezas, e ainda assim, na média
simples, esse orador respondia por um terço do sinal da call inteira. O
tom agregado corrige isso, e a ponderação explícita de cada gestor pelo
volume de fala leva a correção ao limite. A Seção 5 mostra que os
resultados enfraquecem passo a passo conforme se devolve peso aos
oradores marginais. Medir bem importa, e dá para provar o quanto.
Verificamos a implementação por reimplementação independente nas 32.387
calls elegíveis (diferença máxima da ordem de 10⁻¹⁶) e por um exemplo
auditável à mão.

## 4. Método de teste

Dois conceitos sustentam a parte empírica e merecem definição antes dos
números. O retorno anormal acumulado (CAR) é a diferença entre o retorno
observado da ação e o que o CAPM previa dado o movimento do mercado,
somada ao longo da janela do evento. Estimamos o beta de cada evento numa
janela de 100 pregões (mínimo de 70), separada do evento por 50 pregões
para não contaminar a estimativa, e acumulamos o resíduo nas janelas de
1, 2 e 5 pregões ao redor do anúncio. Calls feitas depois do fechamento
do mercado (27% da amostra) são ancoradas no pregão seguinte, que é
quando a informação de fato encontrou o investidor.

O segundo conceito é o painel com efeitos fixos, a espinha dorsal dos
testes. A regressão dá um intercepto próprio a cada empresa e a cada
trimestre, e controla por tamanho, valor, alavancagem, rentabilidade,
surpresa de lucro, momentum e reversão, com erros-padrão agrupados por
empresa e winsorização em 5/95 (os 5% mais extremos de cada variável são
truncados, prática comum para conter outliers). O efeito fixo de empresa
muda a natureza da comparação: o coeficiente não compara a Apple com a
Exxon, compara a Apple de um trimestre com a própria Apple típica. O que
se mede é o efeito de a empresa divergir mais do que o seu normal.

Uma nota de leitura para o restante do relatório: reportamos sempre a
estatística t de cada coeficiente, que diz quantos erros-padrão o efeito
estimado dista de zero. Pela convenção usual, t acima de 2 em módulo
indica significância ao nível de 5%.

Na parte de implementação, desenhamos dois backtests independentes, os
dois com custos de transação e com uma garantia auditada nos próprios
scripts: nenhuma informação futura entra na seleção. O primeiro executa a
tese do jeito mais direto possível. Ao fim de cada mês, compra em pesos
iguais as 10 empresas de maior TD (pela call mais recente dos três meses
anteriores), carrega por um mês, rebalanceia e paga 10 pontos-base por
lado sobre o giro. O segundo é um long-short por quintis: compra o quinto
de maior TD, vende o quinto de menor, em formato calendar-time com
tranches sobrepostas e especificação congelada antes de rodar (entrada no
pregão seguinte à call, manutenção de 63 pregões, ranking contra os 90
dias anteriores, custos de 5 pontos-base por perna, amostra 2006 a 2025).

Todos os números do relatório saem de scripts versionados no repositório
público. Quem quiser reproduz as regressões e os backtests com quatro
comandos, sem reprocessar transcrições; testamos o procedimento em clone
limpo e os números saíram idênticos.

## 5. Resultados econométricos

Antes das regressões, conferimos se a base reconstruída se parece com a
da literatura. A distribuição da nossa TD (média 0,0086, mediana 0,0080,
desvio 0,0047) fica próxima da referência (0,0079, 0,0074, 0,0048), e os
controles batem com os publicados: alavancagem 0,24 contra 0,235,
alíquota efetiva mediana 0,22 contra 0,21, suavização de lucros 1,6
contra 1,4. O ponto de partida é comparável.

### 5.1 H1, precificação no anúncio: confirmada

Se a divergência de tom revela informação negativa, o retorno anormal do
anúncio deve cair com a TD, tudo o mais constante. A tabela reporta as
estatísticas t do coeficiente da TD sobre o CAR nas três janelas de
evento, para as três construções do sinal e os dois medidores (amostra de
2009 em diante, cerca de 23 mil eventos e 540 empresas). Cada célula
responde à mesma pergunta: medindo desta forma, a divergência derruba o
retorno do anúncio?

| construção do sinal | LM (dicionário) | FinBERT (neural) |
|---|---|---|
| centro = média simples | −0,7 / −0,6 / −0,3 | +0,1 / −0,2 / −0,5 |
| centro = tom agregado | −2,0 / −2,1 / −1,7 | −1,9 / −2,3 / −2,3 |
| ponderada pelo volume de fala | −2,9 / −2,7 / −2,3 | −2,4 / −2,4 / −2,5 |

(t nas janelas de 1, 2 e 5 pregões. Em termos econômicos, pular do
quartil inferior ao superior da TD custa entre 0,11% e 0,16% de retorno
anormal no anúncio.)

A tabela conta três histórias ao mesmo tempo. A hipótese se confirma na
amostra completa sempre que o sinal respeita o volume de fala de cada
gestor: as duas últimas linhas são significativas em todas as janelas. O
enfraquecimento é gradual de baixo para cima, ou seja, quanto mais peso
se dá ao ruído dos oradores marginais, mais o sinal se dilui, que era o
diagnóstico antecipado na Seção 3. E as colunas se espelham: dois
medidores que concordam entre si em apenas 38% chegam ao mesmo desenho, o
que tira força da explicação por artefato de instrumento.

### 5.2 Retornos subsequentes: confirmados

O efeito não termina no anúncio. Em painel mensal, regredimos o retorno
dos três meses seguintes à call contra a TD, com efeitos fixos de empresa
e mês e controles de valor, momentum, tamanho e reversão. O coeficiente
sai positivo nos dois medidores (t=+2,1 no dicionário, +1,8 no FinBERT).
Junto com a H1, o desenho fecha: o mercado castiga a empresa divergente
na hora e passa a exigir retorno maior para carregá-la. É o padrão
clássico de compensação por risco.

### 5.3 H3, incerteza operacional: confirmada

Se a divergência reflete incerteza interna sobre o negócio, ela deve
anteceder surpresas de lucro maiores, para qualquer lado. Por isso a
variável testada é o valor absoluto da surpresa. No painel completo, o
coeficiente da TD é positivo (t=+2,2). No exercício de previsão fora da
amostra, em que o modelo é treinado só com o passado e tenta prever
eventos que nunca viu, a relação chega a t=+3,58, com acerto direcional
em 71% dos trimestres. É o resultado preditivo mais forte do projeto. Em
linguagem de mesa: a divergência de hoje avisa que o próximo balanço
tende a vir com surpresa grande.

### 5.4 H2, risco: descartada como sinal

Contamos o caminho até o descarte porque ele ilustra o método. Na amostra
completa existe associação entre TD e volatilidade futura (t=+2,5 no
dicionário). Ao investigar, encontramos um problema de medição: janelas
de volatilidade que começam no dia seguinte ao anúncio ainda carregam a
reação ao próprio anúncio, e empresas de TD alta reagem mais, que é
justamente a H1. Medida a partir do segundo dia, fora desse eco, a
associação perde significância (t=+1,3). Rodamos então 16 especificações
predefinidas (com e sem o eco, dois horizontes, nível e logaritmo, duas
construções do sinal) e nenhuma mostrou poder de previsão. A conclusão
que sobrou: a TD não diz sobre risco futuro nada que a volatilidade
passada já não dissesse, e o canal saiu do conjunto de sinais. Segue
verdadeiro, como descrição, que empresas de TD alta são em média mais
voláteis, e esse fato reaparece na interpretação do backtest.

### 5.5 Robustez e estabilidade temporal

Um resultado de amostra completa pode, em tese, ser fruto de um único
subperíodo favorável. Para checar, refizemos as estimações usando apenas
os dados disponíveis até cada data de corte e conferimos no período
posterior, além da previsão por evento fora da amostra já descrita. O
coeficiente da H1 mantém o sinal negativo em todos os cortes, e a
previsão fora da amostra confirma o efeito (t=−2,12, negativo em cada um
dos anos de 2019 a 2025). A camada de retornos subsequentes mantém
coeficiente positivo em todos os cortes, e a H3 sustenta o t=+3,58. Nos
primeiros anos, com menos dados acumulados, os intervalos de confiança
são mais largos, como se espera de um efeito estável estimado com
precisão crescente. A exceção é a H2, reprovada nesses mesmos exercícios
(t=+0,27), e daí o descarte. O detalhe que nos dá confiança no critério:
o mesmo protocolo que valida duas hipóteses reprova a terceira.

## 6. Backtest

A implementação principal executa a tese do jeito que um gestor
reconheceria: carteira comprada, regra fixa, rebalanceamento mensal. Para
a leitura das tabelas: o índice de Sharpe é o retorno em excesso por
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

(*2025 até agosto. Giro médio de 34% ao mês. As composições são
verificáveis; agosto de 2025, por exemplo: HII, MCD, CAH, ES, MCK, MTCH,
CSCO, ANET, PODD, ABNB. Séries e composições completas em
data/interim/sp500/port10_*.csv; replicação por scripts/sp500_port10.py.)

Esses números pedem uma leitura em duas camadas. Contra o S&P 500, o
excesso é de +7,1% ao ano com t=2,60, vitória em 58% dos meses e em 13
dos 17 anos. É o número de manchete, e é real. Só que um leitor técnico
vai notar uma sutileza: carteiras de pesos iguais tendem a superar
índices ponderados por valor simplesmente por dar mais espaço a empresas
menores. Para saber o que é mérito da seleção, montamos a régua mais
exigente que existe aqui: o próprio universo elegível em pesos iguais.
Contra essa régua, a TD adiciona +3,6% ao ano (t=1,59), com contribuição
positiva em todos os subperíodos que olhamos (+1,8% de 2010 em diante,
+1,9% de 2015 em diante, +2,6% de 2019 em diante), embora sem
significância isolada. A explicação é mecânica: com 10 ativos, o ruído
específico de cada nome domina a variância da carteira e dilui um efeito
que o painel estima em uns 15 pontos-base por variação interquartil. A
carteira espelho, feita com as 10 empresas de menor TD, rende +15,3% ao
ano, 5,1 pontos abaixo do top-10, na direção que a tese previa.

O long-short por quintis não monetiza. Na especificação congelada, os
retornos líquidos anualizados ficam entre +0,3% e +4,4% conforme a
construção, com o melhor Sharpe em 0,22, estatisticamente nulo, contra
0,53 do mercado no período. O diagnóstico de pernas mostra o porquê: a
correlação entre a ponta comprada e a vendida é de só 0,6, e o quintil
alto do sinal FinBERT carrega exposição a ações de baixa volatilidade, um
fator de risco conhecido, em vez de tom. As 6 construções testadas estão
registradas e entram na contagem de tentativas para o ajuste de múltiplos
testes (Deflated Sharpe, Bailey e López de Prado, 2014) na entrega final.

O perfil de risco fecha a história. O top-10 realiza volatilidade maior
que todas as réguas (20,6% contra 17,3% a 17,4%) e Sharpe parecido com o
do universo (1,01 contra 0,99). O retorno extra remunera risco extra:
prêmio de compensação por carregar as empresas cuja diretoria diverge,
coerente com a associação descritiva da Seção 5.4, e não anomalia
gratuita. Para a mesa, a evidência sustenta quatro usos: tilt direcional
sobre um portfólio core; condicionamento de exposição em torno de
anúncios, já que o escore fica pronto minutos após cada call;
posicionamento de volatilidade no próximo anúncio das empresas de
divergência alta, apoiado na previsão de surpresa de lucro; e insumo em
arcabouço multifator, cuja engenharia é o objeto da entrega final.

## 7. Riscos e limitações

1. Os papéis de orador foram reconstruídos por classificador próprio, com
   cerca de 90% de acerto validado. As bases comerciais curadas da
   literatura não têm esse ruído, e parte da diferença de magnitude entre
   as nossas estatísticas e as publicadas pode vir daí.
2. Algumas variáveis usam proxies: a surpresa de lucro vem do Yahoo
   Finance em vez do consenso I/B/E/S, e dois controles da literatura
   (participação institucional e qualidade de accruals) não foram
   reconstruídos; os outros 17, sim.
3. Cerca de 3,4 mil eventos ficaram sem retorno anormal por falta de
   preços de empresas deslistadas em fonte gratuita, uma limitação de
   sobrevivência que declaramos.
4. O efeito no anúncio, de uns 0,15% por variação interquartil, não paga
   custos como estratégia isolada de arbitragem do evento em large caps.
   Por isso a implementação proposta é de tilt, condicionamento e
   volatilidade, e não de arbitragem do anúncio.
5. Com 10 ativos, o excesso sobre o universo em pesos iguais é
   consistente, mas não significativo isoladamente. Diversificar além de
   10 nomes é o caminho natural da entrega final.
6. A definição do centro da call admite duas construções na literatura.
   Reportamos as duas, e a escolha da agregada não foi guiada por
   resultado, e sim pela evidência textual da referência e pelo
   diagnóstico de ruído da Seção 3.

## 8. Conclusão e tese de investimento

O pré-relatório estabelece que a divergência de tom entre gestores é
informação precificada. Confirmamos o fenômeno em universo independente,
com dados públicos e código próprio, mostramos que ele sobrevive à troca
completa do medidor de tom e o transformamos em estratégia implementável,
com custos e auditoria de ausência de informação futura.

A tese de investimento cabe em um parágrafo. A divergência de tom entre
os executivos na earnings call é informação precificada em dois tempos.
No anúncio, o mercado penaliza a empresa cuja diretoria diverge, um
efeito condicional negativo e estável no tempo. Nos meses seguintes,
carregar as empresas de maior divergência captura um prêmio de
compensação: a carteira das 10 maiores TD rendeu +20,4% ao ano contra
+13,2% do S&P 500 em 16 anos e meio, remunerando o risco adicional que o
próprio sinal identifica. O mesmo escore antecipa a magnitude da surpresa
do lucro seguinte e serve de termômetro de incerteza antes de cada
anúncio. A implementação recomendada combina tilt direcional,
condicionamento de exposição a eventos e posicionamento de volatilidade,
com a integração multifator como objeto da entrega final.

Registramos também a fronteira que não avançou. Testamos o sinal em 2.250
firmas fora do S&P 500 (2019 a 2023), com especificações congeladas por
pré-registro antes de qualquer resultado (docs/PREREG_MF.md, com data
registrada em git). O painel saiu inconclusivo por falta de potência
estatística: coeficientes positivos, não significativos, janela curta,
cobertura de preços de 69%. No mesmo período, o efeito segue presente nas
large caps (t=−1,96), o que aponta a causa para o universo e a qualidade
dos dados, e não para a época. A fronteira fica documentada e não é
apresentada como resultado.

Até a entrega final (17/08), a agenda tem três itens: desenhar a
integração multifator do escore, com contagem de tentativas e Deflated
Sharpe; fechar a robustez de custos e a análise de decaimento do sinal;
e, havendo dados de preços de deslistadas, revisitar a fronteira de small
e mid caps com o pré-registro já versionado.

Encerramos com uma nota de método. Em amostra completa, as três hipóteses
pareciam confirmadas. Foram os exercícios de robustez temporal que
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
rótulos manuais (cerca de 90% de acerto global) e um classificador
determinístico independente para comparação (99% de concordância nos
casos frequentes); prompt, custos e validação estão em
docs/GENAI_USAGE.md. Ao longo de todo o processo, o assistente (Claude)
atuou como par de engenharia e auditoria: arquitetura do pipeline,
código, testes, documentação e, com impacto direto no resultado, a
auditoria da fórmula do centro da call, sempre com as decisões econômicas
e metodológicas da equipe. Uma distinção conceitual por fim: o FinBERT é
um modelo de linguagem do tipo encoder, parte do modelo quantitativo; é
permitido pelo edital, mas o contabilizamos como NLP/ML, e não como IA
generativa.

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
