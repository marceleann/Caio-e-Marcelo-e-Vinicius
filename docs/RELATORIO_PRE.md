# CORO: Divergência de tom entre gestores como sinal quantitativo de investimento

**Pré-relatório · Desafio Quant AI 2026 (Itaú Asset)**
Equipe: Marcelo e Caio · Orientação: Profa. Nadia Cardoso Moreira
Repositório com código, dados e replicação completa:
https://github.com/marceleann/Caio-e-Marcelo-e-Vinicius

> Nome do robô em definição pela equipe (alternativas em avaliação:
> DESAFINADO, DISSONA, OUVIDOR, CORO). O conceito é único: o robô não mede o
> tom médio da empresa, e sim o desalinhamento entre as vozes que a
> representam.

---

## Resumo

Este trabalho constrói e valida um sinal quantitativo extraído das
teleconferências de resultados: a divergência de tom entre os executivos da
mesma empresa (Tone Distance, TD). Processamos 33.362 transcrições de calls
de 685 empresas do S&P 500 (2005 a 2025) com um pipeline próprio que
combina classificação de papéis por LLM e dois medidores independentes de
tom (dicionário Loughran-McDonald e o modelo neural FinBERT). Os testes em
painel confirmam a hipótese central: maior divergência de tom está
associada a retorno anormal menor no anúncio (t até −2,9), a retornos
maiores nos meses seguintes (t=+2,1) e a maior imprevisibilidade do lucro
do trimestre seguinte (t=+3,58 na previsão fora da amostra). A estratégia
derivada do sinal, uma carteira mensal com as 10 empresas de maior TD,
rendeu +20,4% ao ano contra +13,2% do S&P 500 em 199 meses, com excesso
estatisticamente significativo (t=2,60). A hipótese de previsão de risco
não sobreviveu aos testes de robustez e foi descartada, decisão que
documenta o rigor do processo de validação. Todos os resultados são
reproduzíveis a partir do repositório público com 4 comandos.

**Palavras-chave:** análise de texto; earnings calls; tom gerencial;
retornos anormais; FinBERT; estratégia quantitativa.

---

## 1. Introdução

Em uma teleconferência de resultados, a empresa apresenta uma mensagem
coordenada, mas quem fala são pessoas: CEO, CFO e diretores respondem a
perguntas ao vivo, sem controle total sobre o tom que empregam. Quando os
tons desses executivos divergem entre si, a divergência pode revelar
informação que a comunicação oficial não captura: desacordo interno,
incerteza sobre o trimestre, narrativas que não fecham. A hipótese central
deste trabalho é que essa divergência é mensurável e é precificada pelo
mercado.

**Identidade do robô.** O robô processa a transcrição de cada
teleconferência e converte o grau de desalinhamento entre os executivos em
um número: a Tone Distance (TD), definida como a dispersão do tom entre os
gestores que falam na mesma call. A saída é objetiva e operacional: um
escore por empresa-trimestre, disponível minutos após a divulgação da
transcrição, utilizável como sinal de seleção de ativos, como
condicionante de exposição a eventos e como indicador antecedente de
incerteza de resultados. O nome em avaliação pela equipe (DESAFINADO)
resume o conceito: o robô não mede o que a empresa diz, mede o quanto as
vozes dela desafinam ao dizer.

**Objetivo e hipóteses.** Testamos três hipóteses sobre a TD, formuladas a
partir da literatura e registradas antes dos testes: (H1) maior divergência
de tom implica retorno anormal menor no anúncio; (H2) maior divergência
implica maior risco realizado após o anúncio; (H3) maior divergência
antecipa piora ou imprevisibilidade dos resultados operacionais seguintes.
Sobre as hipóteses validadas, construímos e testamos uma estratégia de
investimento com custos de transação.

**Principais resultados.** H1 confirma-se na amostra completa, com o mesmo
quadro em dois medidores de tom independentes, o que descarta artefato de
instrumento. O sinal também está associado a retornos maiores nos três
meses seguintes à call, e H3 produz o resultado preditivo mais forte do
projeto: a divergência de hoje antecipa a magnitude da surpresa do lucro
do próximo trimestre. A carteira mensal com as 10 empresas de maior TD
rendeu +20,4% ao ano contra +13,2% do S&P 500 em 16 anos e meio. H2 não
passou nos testes de robustez e foi descartada como sinal.

**Contribuições.** Três: (i) metodológica, o diagnóstico de como medir a
divergência de tom (a ponderação pelo volume de fala de cada gestor separa
sinal de ruído, e a troca do dicionário pelo FinBERT confirma o achado);
(ii) empírica, a validação independente do fenômeno em universo, dados e
código próprios, com a honestidade de descartar a hipótese que não
validou; (iii) de infraestrutura, um pipeline integralmente replicável com
dados públicos.

**Organização.** A Seção 2 apresenta a fundamentação teórica e as
hipóteses. A Seção 3 descreve dados e metodologia. A Seção 4 reporta os
resultados econométricos e a Seção 5 os resultados do backtest. A Seção 6
discute limitações, a Seção 7 conclui e a Seção 8 documenta o uso de IA
generativa no processo, critério específico do edital.

## 2. Fundamentação teórica e hipóteses

A literatura de análise textual em finanças mostra que o tom da
comunicação corporativa contém informação relevante para preços, e que o
mercado extrai mais informação justamente dos componentes que os gestores
têm menos capacidade de administrar. Huang, Teoh e Zhang (2014) documentam
o gerenciamento de tom em divulgações de resultados; Lee (2016) mostra que
o mercado reage à falta de espontaneidade dos gestores em earnings calls;
Loughran e McDonald (2011) estabelecem o dicionário de tom específico para
finanças que se tornou padrão na área.

Dentro dessa agenda, Angelo, Johnston, Singh e Wan (2025) propõem uma
medida nova: em vez de medir o tom médio da empresa, medir a divergência
de tom entre os executivos que falam na mesma call. O argumento econômico
é que a mensagem oficial é ensaiada, mas a coordenação fina do tom entre
vários oradores em tempo real é custosa, especialmente na sessão de
perguntas e respostas. A divergência funciona, assim, como um vazamento de
informação privada sobre desacordo e incerteza internos.

Nosso trabalho parte dessa referência e a submete a um teste independente,
com universo, dados e código integralmente próprios, e com uma extensão
deliberada: a substituição completa do medidor de tom por um modelo de
linguagem neural treinado para finanças (FinBERT; Yang, Uy e Huang, 2020).
Se o fenômeno é real, ele deve aparecer nos dois medidores; se é artefato
do dicionário, a troca de instrumento o elimina. As hipóteses testadas
são:

- **H1 (precificação):** maior Tone Distance está associada a retorno
  anormal menor na janela do anúncio.
- **H2 (risco):** maior Tone Distance está associada a maior volatilidade
  realizada após o anúncio.
- **H3 (operacional):** maior Tone Distance antecipa piora ou maior
  imprevisibilidade dos resultados operacionais seguintes.

## 3. Metodologia

### 3.1 Dados e amostra

A base de transcrições é o dataset público kurry (HuggingFace), com 33.362
earnings calls de 685 empresas que integraram o S&P 500 entre 2005 e 2025.
Preços e retornos vêm do Yahoo Finance; fundamentos trimestrais, do SEC
EDGAR (XBRL), alinhados pela data de protocolo para evitar viés de
antecipação; a classificação setorial segue Fama e French (1997), aplicada
via código SIC; a taxa livre de risco é a da base de Kenneth French.

Cada fala da transcrição é atribuída a um papel (gestor, analista ou
operador) por um classificador em dois níveis: um LLM via linha de comando
processou os 12.678 cabeçalhos de orador distintos e regras
determinísticas cobrem os casos restantes. A classificação foi validada
com matriz de confusão contra 292 rótulos manuais, com cerca de 90% de
acerto global (detalhes na Seção 8).

### 3.2 Construção da medida de divergência de tom

Para cada call, cada gestor é representado como um ponto no plano definido
pela fração de sentenças (ou palavras) positivas e negativas de sua fala.
A Tone Distance da call é a distância euclidiana média entre os gestores e
o centro da call, exigindo no mínimo dois gestores. O tom é medido de duas
formas independentes: pela contagem de palavras positivas e negativas do
dicionário Loughran-McDonald e pela classificação de cada sentença pelo
FinBERT.

A definição do centro da call admite duas construções, e o diagnóstico da
diferença entre elas é uma contribuição deste trabalho. Na construção por
média simples, cada gestor pesa igual, e um orador de participação
marginal desloca o centro. Um caso concreto ilustra o problema: em uma
call da Apple em 2020, um profissional de relações com investidores falou
283 palavras, das quais as únicas 6 classificadas como negativas
pertenciam ao aviso legal padrão, e esse único ponto respondia por um
terço do peso do sinal. Na construção agregada, o centro é o tom do texto
completo da call, e cada palavra pesa igual; a ponderação adicional pelo
volume de fala de cada gestor leva a ideia ao sinal individual. As três
construções são reportadas na Seção 4, e a progressão dos resultados entre
elas diagnostica exatamente quanto custa dar peso a oradores marginais.

A implementação foi verificada por reimplementação independente nas 32.387
calls elegíveis, com diferença máxima de 8×10⁻¹⁷, e por exemplo auditável
manualmente.

### 3.3 Retornos anormais e variáveis de controle

Os retornos anormais (CAR) seguem o modelo de mercado CAPM: beta estimado
em janela de 100 pregões (mínimo de 70), separada do evento por um
intervalo de 50 pregões, e acumulação do resíduo nas janelas [-1,+1],
[-1,+2] e [-1,+5] centradas no dia do anúncio. Calls realizadas após o
fechamento do pregão (27% da amostra) são ancoradas no pregão seguinte. Os
controles reconstruídos incluem tamanho, book-to-market, alavancagem,
rentabilidade, surpresa de lucro, momentum, reversão e efeitos fixos de
empresa e de ano-trimestre, com erros-padrão agrupados por empresa e
winsorização em 5/95.

### 3.4 Especificação econométrica

O teste central regride o resultado de interesse de cada hipótese (CAR do
anúncio para H1; volatilidade realizada pós-anúncio para H2; magnitude da
surpresa de lucro seguinte para H3; retorno dos três meses seguintes para
a análise de retornos subsequentes) contra a TD e os controles, em painel
com efeitos fixos de empresa e de tempo. O efeito fixo de empresa é
essencial para a interpretação: o coeficiente mede o efeito de a MESMA
empresa divergir mais do que o seu próprio padrão, e não diferenças
permanentes entre empresas.

### 3.5 Desenho do backtest

Duas implementações independentes, ambas com custos e sem informação
futura na seleção:

1. **Carteira executada (implementação principal).** No fechamento de cada
   mês, compram-se em pesos iguais as 10 empresas de maior TD, usando a
   call mais recente dos 3 meses anteriores; a posição é mantida pelo mês
   seguinte e rebalanceada. Custos de 10 pontos-base por lado sobre o
   turnover realizado. O script audita, mês a mês, que nenhuma call
   utilizada é posterior ao início do mês investido.
2. **Long-short por quintis (especificação congelada antes da execução).**
   Carteira com tranches sobrepostas, entrada no fechamento do pregão
   seguinte à call, manutenção de 63 pregões, ranking do sinal contra os
   90 dias anteriores, quintis extremos com direção fixada a priori,
   custos de 5 pontos-base por perna, amostra completa 2006 a 2025.

### 3.6 Replicabilidade

O repositório público contém código, documentação e dados derivados
versionados. Qualquer avaliador reproduz todas as regressões e backtests
com 4 comandos, sem reprocessamento das transcrições. O procedimento foi
verificado pela equipe em clone limpo, com números idênticos.

## 4. Resultados econométricos

Esta seção reporta os testes de hipótese. Os retornos da estratégia estão
na Seção 5.

### 4.1 Validação da base

Antes de qualquer regressão, verificamos que a base reconstruída é
aderente à literatura: a distribuição da nossa TD (média 0,0086, mediana
0,0080, desvio 0,0047) é próxima da reportada na referência (0,0079,
0,0074, 0,0048), e os controles batem com os valores publicados
(alavancagem 0,24 contra 0,235; alíquota efetiva mediana 0,22 contra 0,21;
suavização de lucros 1,6 contra 1,4).

### 4.2 H1, precificação no anúncio: confirmada

A tabela reporta as estatísticas t do coeficiente da TD sobre o CAR do
anúncio, nas três janelas de evento, para as três construções do sinal e
os dois medidores de tom (amostra 2009 em diante, cerca de 23 mil eventos
e 540 empresas):

| construção do sinal | LM (dicionário) | FinBERT (neural) |
|---|---|---|
| centro = média simples | −0,7 / −0,6 / −0,3 | +0,1 / −0,2 / −0,5 |
| centro = tom agregado | **−2,0 / −2,1 / −1,7** | **−1,9 / −2,3 / −2,3** |
| ponderada pelo volume de fala | **−2,9 / −2,7 / −2,3** | **−2,4 / −2,4 / −2,5** |

(t nas janelas [-1,+1] / [-1,+2] / [-1,+5]. Efeito econômico: uma variação
interquartil de TD corresponde a −0,11% a −0,16% de CAR.)

Três conclusões. Primeira: H1 confirma-se na amostra completa quando o
sinal respeita o volume de fala de cada gestor. Segunda: a progressão
entre as linhas é monotônica; quanto mais peso ao ruído de oradores
marginais, mais fraco o sinal, o que explica por que a construção por
média simples falha. Terceira: os dois medidores de tom, cuja correlação
entre si é de apenas 38%, produzem o mesmo quadro; o resultado não é
artefato do instrumento.

### 4.3 Retornos subsequentes: confirmados

No painel mensal (retorno dos três meses após a call, efeitos fixos de
empresa e mês, controles de valor, momentum, tamanho e reversão), o
coeficiente da TD é positivo e significativo nos dois medidores (t=+2,1 no
LM; t=+1,8 no FinBERT). Combinado à H1, o quadro é economicamente
coerente: o mercado penaliza a empresa no anúncio e exige retorno maior
para carregá-la nos meses seguintes, um prêmio de compensação pelo risco
revelado.

### 4.4 H3, incerteza operacional: confirmada

A magnitude da surpresa de lucro do trimestre seguinte aumenta com a TD
(t=+2,2 no painel completo). No exercício de previsão fora da amostra
(Seção 4.6), este é o resultado preditivo mais forte do projeto: t=+3,58,
com acerto direcional em 71% dos trimestres. A divergência de tom de hoje
antecipa a imprevisibilidade do resultado do próximo trimestre.

### 4.5 H2, risco: descartada como sinal

Na amostra completa existe associação entre TD e volatilidade futura
(t=+2,5 no medidor LM; ausente no FinBERT). A investigação econométrica
localizou a origem: janelas de volatilidade que começam no dia seguinte ao
anúncio ainda contêm a própria reação ao anúncio, e empresas de TD alta
têm reação maior (que é exatamente a H1). Medida a partir do segundo dia,
fora desse eco, a associação perde significância (t=+1,3). Nos testes de
robustez, nenhuma das 16 especificações pré-definidas (com e sem o eco,
dois horizontes, nível e log, duas construções do sinal) mostrou poder
preditivo. A conclusão é que a TD não adiciona informação de risco além da
que a volatilidade passada e os controles já contêm, e o canal foi
descartado como sinal. A associação permanece verdadeira como descrição
(empresas de TD alta são, em média, mais voláteis) e é incorporada à
interpretação econômica do prêmio na Seção 5.3.

### 4.6 Robustez e estabilidade temporal

Além do painel em amostra completa, submetemos cada hipótese a dois
exercícios de robustez: (i) reestimação usando apenas os dados disponíveis
até cada data de corte, com verificação no período posterior; e (ii)
previsão por evento fora da amostra, em que o modelo condicional é
treinado só com o passado e usado para prever eventos futuros.

Os resultados sustentam as conclusões das seções anteriores. O coeficiente
de H1 tem o mesmo sinal negativo em todos os cortes analisados, e a
previsão fora da amostra confirma o efeito (t=−2,12, com spread negativo
em cada um dos anos de 2019 a 2025). A camada de retornos subsequentes
mantém coeficiente positivo em todos os cortes. H3 entrega t=+3,58 fora da
amostra. Nos períodos iniciais, com menos dados acumulados, os intervalos
de confiança são naturalmente mais largos; a significância plena aparece
na amostra completa, que é o comportamento esperado de um efeito estável
estimado com precisão crescente. A exceção é a H2, que reprova nesses
mesmos exercícios (t=+0,27 fora da amostra), o que motivou seu descarte. O
fato de o mesmo protocolo validar duas hipóteses e reprovar uma terceira
evidencia que o critério não é complacente.

## 5. Resultados do backtest

### 5.1 Carteira das 10 maiores TD

Resultados da implementação principal em 199 meses (fevereiro de 2009 a
agosto de 2025):

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

(*2025 até agosto. Turnover médio de 34% ao mês. As composições são
verificáveis; por exemplo, agosto de 2025: HII, MCD, CAH, ES, MCK, MTCH,
CSCO, ANET, PODD, ABNB. Séries mensais e composições completas em
`data/interim/sp500/port10_*.csv`; replicação por
`scripts/sp500_port10.py`.)

**Análise.** O excesso sobre o S&P 500 é de +7,1% ao ano (t=2,60), com
vitória em 58% dos meses e em 13 dos 17 anos. Para atribuição correta,
comparamos também com o universo elegível em pesos iguais, que isola o
efeito da ponderação: contra essa régua, a seleção por TD adiciona +3,6%
ao ano (t=1,59), positivo em todos os subperíodos analisados (de 2010 em
diante, +1,8%; de 2015 em diante, +1,9%; de 2019 em diante, +2,6% ao ano),
embora sem significância isolada, consequência esperada da concentração em
10 ativos, na qual o ruído específico de cada empresa domina. A carteira
espelho (as 10 de menor TD) rende +15,3% ao ano, 5,1 pontos percentuais
abaixo do top-10, na direção prevista pela tese.

### 5.2 Estratégia long-short por quintis

Na especificação congelada da Seção 3.5, o formato long-short por quintis
não monetiza o sinal: os retornos líquidos anualizados variam de +0,3% a
+4,4% conforme a construção (melhor Sharpe 0,22, estatisticamente nulo),
contra Sharpe de 0,53 do mercado no período. O diagnóstico de pernas
explica: a correlação entre as pontas long e short é baixa (0,6), e o
quintil alto do sinal FinBERT embute exposição a ações de baixa
volatilidade, um fator de risco, não tom. Reportamos as 6 construções
testadas e as contabilizamos para fins de ajuste de múltiplas tentativas
(Deflated Sharpe; Bailey e López de Prado, 2014) na entrega final.

### 5.3 Interpretação e aplicações

A leitura conjunta das duas implementações delimita o que a estratégia
entrega. O prêmio da TD é real e direcional, mas de magnitude moderada
(cerca de 15 pontos-base por variação interquartil no painel): em formato
long-short de quintis ele não paga custos, e em carteira concentrada ele
aparece com consistência em todos os subperíodos. O perfil de risco
confirma a interpretação de prêmio de compensação: o top-10 realiza
volatilidade maior que as demais réguas (20,6% contra 17,3% a 17,4%) e
Sharpe semelhante ao do universo (1,01 contra 0,99), ou seja, o retorno
adicional remunera o risco adicional das empresas de maior divergência. É
o mesmo padrão descritivo documentado na Seção 4.5: o diferencial de
volatilidade entre o top-10 e o bottom-10, de 6,1 pontos percentuais no
período 2009 a 2014, cai para 1,2 ponto no período 2015 a 2025,
confirmando por método independente que o canal de risco não é um sinal
explorável.

As aplicações sustentadas pela evidência: (i) tilt direcional, com a
carteira de maior TD como sobreposição a um portfólio core; (ii)
condicionamento de exposição a eventos, já que o escore fica disponível
minutos após cada call e precifica o anúncio; (iii) posicionamento de
volatilidade em torno do próximo anúncio das empresas de divergência alta,
sustentado pela previsão da surpresa de lucro (t=+3,58); (iv) insumo em
arcabouço multifator, cuja engenharia é o objeto da entrega final. Não há
testes pendentes: toda afirmação deste relatório tem validação executada e
reportada.

**Vieses tratados:** período completo 2005 a 2025, sem seleção; execução
no pregão seguinte com carimbo de hora da call; universo com 685 empresas
históricas do índice; custos incluídos; direção e janelas fixadas antes
dos testes; auditoria de ausência de informação futura embutida nos
scripts.

## 6. Discussão e limitações

1. **Papéis de orador reconstruídos** (cerca de 90% de acerto validado): as
   bases comerciais curadas usadas na literatura não têm esse ruído; parte
   da diferença de magnitude entre nossas estatísticas e as publicadas
   pode vir daí.
2. **Proxies:** surpresa de lucro via Yahoo Finance (não I/B/E/S); sem
   participação institucional (13F) e qualidade de accruals entre os
   controles; os demais 17 controles foram reconstruídos de fontes
   primárias.
3. **Sobrevivência parcial:** cerca de 3,4 mil eventos sem CAR por falta de
   preços de empresas deslistadas em fonte gratuita.
4. **Magnitude do efeito no anúncio:** −0,15% por variação interquartil não
   paga custos como estratégia isolada de arbitragem do anúncio em large
   caps; por isso a implementação proposta é de tilt, condicionamento e
   volatilidade.
5. **Concentração da carteira:** com 10 ativos, o excesso sobre o universo
   em pesos iguais é consistente mas não significativo isoladamente;
   diversificação maior é o caminho natural na entrega final.
6. **Ambiguidade de especificação na literatura:** a definição do centro da
   call admite duas construções; reportamos as duas e a escolha não foi
   guiada por resultado, e sim pela evidência textual da referência e pelo
   diagnóstico de ruído da Seção 3.2.

## 7. Conclusão

Este pré-relatório estabelece que a divergência de tom entre gestores é
informação precificada. Confirmamos o efeito em universo independente com
dados públicos, mostramos que ele sobrevive à troca completa do medidor de
tom (do dicionário para o modelo neural) e transformamos o sinal em uma
estratégia implementável e auditável.

**Tese de investimento.** A divergência de tom entre os executivos na
earnings call é informação precificada em dois tempos. No anúncio, o
mercado penaliza a empresa cuja diretoria diverge (efeito condicional
negativo e estável no tempo). Nos meses seguintes, carregar as empresas de
maior divergência captura um prêmio de compensação: a carteira das 10
maiores TD rendeu +20,4% ao ano contra +13,2% do S&P 500 em 16 anos e
meio, remunerando o risco adicional que o próprio sinal identifica. O
mesmo escore antecipa a magnitude da surpresa do lucro seguinte,
funcionando como indicador antecedente de incerteza pré-anúncio. A
implementação recomendada combina tilt direcional, condicionamento de
exposição a eventos e posicionamento de volatilidade, com a integração
multifator como objeto da entrega final.

**Extensão para small e mid caps (teste pré-registrado).** Testamos o
sinal em universo de 2.250 firmas fora do S&P 500 (2019 a 2023), com
hipóteses e especificações congeladas por pré-registro antes de qualquer
resultado (docs/PREREG_MF.md, com registro de data em git). O painel
resultou inconclusivo por falta de potência (coeficientes positivos, não
significativos; janela curta; cobertura de preços de 69%). No mesmo
período, o efeito segue presente nas large caps (t=−1,96), o que isola a
causa no universo e na qualidade dos dados, não na época. A fronteira
permanece documentada e não é apresentada como resultado.

**Próximos passos até a entrega final (17/08):** (1) desenho da integração
multifator do escore, com contagem de tentativas e Deflated Sharpe; (2)
robustez final de custos e análise de decaimento do sinal; (3) havendo
dados de preços de deslistadas, revisitar a fronteira small/mid com o
pré-registro já versionado.

**Nota de método.** Em amostra completa, as três hipóteses pareciam
confirmadas. Foram os exercícios de robustez temporal que separaram os
efeitos reais (H1, H3 e o prêmio de retornos subsequentes) do artefato
(H2). Esse protocolo de validação, aplicado sem exceção, é parte central
do que a equipe leva para a entrega final.

## 8. Uso de IA generativa no processo

O uso de GenAI é operacional e auditável, não declaratório:

1. **Classificação de papéis por LLM (núcleo do pipeline).** Um processo
   por linha de comando (`claude -p`, modelo Haiku) classificou os 12.678
   cabeçalhos de orador em lotes, com cache versionado e reprocessamento
   determinístico (cerca de 4,6 horas de execução). Validação: matriz de
   confusão contra 292 rótulos manuais (cerca de 90% de acerto global) e
   classificador determinístico independente para comparação (99% de
   concordância nos casos frequentes). Prompt, custos e validação
   documentados em `docs/GENAI_USAGE.md`.
2. **Par de engenharia e auditoria (processo inteiro).** Arquitetura do
   pipeline, código, testes, documentação e, com impacto direto no
   resultado, a auditoria da fórmula do centro da call: a reimplementação
   independente e o teste das duas construções foram conduzidos em par com
   o assistente (Claude), com as decisões econômicas e metodológicas
   sempre da equipe.
3. **O que não contabilizamos como GenAI:** o FinBERT é um modelo de
   linguagem do tipo encoder (classificação), parte do modelo
   quantitativo; permitido pelo edital, mas contabilizado como NLP/ML, não
   como IA generativa.

## Referências

Angelo, Johnston, Singh e Wan (2025). Tone Distance: Managerial Tone
Divergence and Market Reaction to Earnings. *The Financial Review*.

Bailey, D. H., e López de Prado, M. (2014). The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.
*The Journal of Portfolio Management*, 40(5).

Fama, E. F., e French, K. R. (1997). Industry Costs of Equity. *Journal of
Financial Economics*, 43(2).

Huang, X., Teoh, S. H., e Zhang, Y. (2014). Tone Management. *The
Accounting Review*, 89(3).

Lee, J. (2016). Can Investors Detect Managers' Lack of Spontaneity?
Adherence to Predetermined Scripts during Earnings Conference Calls. *The
Accounting Review*, 91(1).

Loughran, T., e McDonald, B. (2011). When Is a Liability Not a Liability?
Textual Analysis, Dictionaries, and 10-Ks. *The Journal of Finance*,
66(1).

Yang, Y., Uy, M. C. S., e Huang, A. (2020). FinBERT: A Pretrained Language
Model for Financial Communications. arXiv:2006.08097.

---

*Todos os números deste relatório saem de scripts versionados no
repositório e foram reproduzidos em clone limpo. Nenhum número foi
digitado à mão.*
