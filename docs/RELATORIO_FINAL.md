# CORO: Divergência de tom entre gestores como sinal quantitativo de investimento

Relatório Final · Desafio Quant AI 2026 (Itaú Asset)
Equipe: Marcelo e Caio · Orientação: Profa. Nadia Cardoso Moreira
Repositório com código, dados e replicação completa:
https://github.com/marceleann/Caio-e-Marcelo-e-Vinicius

> O nome CORO resume a estratégia: numa earnings call, os executivos formam
> um coro que ensaia uma mensagem única, e o robô mede o desalinhamento
> dessas vozes. Quando o coro desafina, há informação.

---

## Sumário executivo

Este relatório apresenta a construção e a validação de um sinal
quantitativo extraído das teleconferências de resultados: a divergência de
tom entre os executivos de uma mesma empresa, medida pela Tone Distance
(TD). Foram processadas 33.362 transcrições de 685 empresas do S&P 500
(2005 a 2025), e os testes em painel confirmam a tese: empresas cujos
gestores divergem mais sofrem retorno anormal menor no anúncio (t até
−2,8), rendem mais nos três meses seguintes (t=+2,0) e entregam surpresas
de lucro mais imprevisíveis no trimestre posterior (t=+3,58 fora da
amostra, o resultado preditivo mais forte). O quadro se repete em dois
medidores de tom independentes, o que afasta artefato de instrumento.

Na implementação, a carteira que compra mensalmente as 10 empresas de
maior TD rendeu +20,4% ao ano contra +13,2% do S&P 500 em 199 meses
(fev/2009 a ago/2025). O relatório também documenta o que não funcionou: a
hipótese de que o sinal prevê volatilidade futura caiu nos testes de
robustez e foi descartada, decisão detalhada por evidenciar o rigor do
processo. Todos os resultados se reproduzem do repositório público com
quatro comandos.

## 1. Contexto, problema e identidade do robô

Quem acompanha earnings calls conhece a cena: o CEO abre com a mensagem
preparada e, no Q&A, cada executivo responde com o tom que consegue
sustentar ao vivo. O texto é coordenado; o tom, nem sempre. A premissa do
trabalho é que esse descompasso carrega informação: um CFO visivelmente
mais cauteloso que o CEO na mesma call sinaliza desacordo interno ou
incerteza que nenhum press release admitiria. Daí as duas perguntas da
pesquisa: é possível medir essa divergência de forma sistemática? E,
medida, ela tem valor econômico?

Dessas perguntas derivam as três hipóteses, registradas antes dos testes:

- H1 (precificação): maior divergência de tom, menor o retorno anormal na
  janela do anúncio.
- H2 (risco): maior divergência, maior a volatilidade realizada após o
  anúncio.
- H3 (operacional): maior divergência, mais imprevisível o resultado do
  trimestre seguinte.

O CORO é a resposta prática: lê a transcrição de cada call, atribui cada
fala ao seu orador, mede o tom de cada gestor e resume o desalinhamento em
um escore por empresa-trimestre, pronto minutos após a publicação da
transcrição. Para a mesa, é insumo no mesmo dia do evento: sinal de
seleção, condicionante de exposição a anúncios e termômetro de incerteza.
A identidade visual traduz o mecanismo: o robô ouvinte, de fones, diante
de vozes que chegam alinhadas e uma que desafina.

![mascote](identidade/coro_mascote.png)

## 2. Referencial teórico

A literatura de finanças textuais estabelece que o tom da comunicação
corporativa move preços: Loughran e McDonald (2011) construíram o
dicionário de tom para finanças; Huang, Teoh e Zhang (2014) mostraram que
gestores administram o tom e que o mercado reage ao seu componente
anormal; Lee (2016) documentou que investidores penalizam executivos
presos a roteiros. A medida deste trabalho vem de Angelo, Johnston, Singh
e Wan (2025), que deslocam a pergunta do tom médio para a divergência
entre os executivos da mesma call, a Tone Distance. Partindo dessa
referência, realizamos um teste independente, com universo, dados e
código próprios, e uma extensão deliberada: toda a análise foi refeita
trocando o medidor de tom pelo FinBERT (Yang, Uy e Huang, 2020), modelo
neural que classifica cada sentença pelo contexto. A troca é a contraprova
do instrumento: fenômeno real aparece nos dois medidores.

## 3. Dados e construção do sinal

A amostra reúne 33.362 transcrições de empresas que integraram o S&P 500
entre 2005 e 2025 (dataset público kurry/HuggingFace, 685 empresas,
incluindo as que saíram do índice). Preços vêm do Yahoo Finance;
fundamentos, do SEC EDGAR alinhados pela data de protocolo, sem
antecipação de informação; setores seguem Fama e French (1997). Para
separar gestores de analistas e operadores, um classificador em dois
níveis (LLM via linha de comando mais regras determinísticas) tratou os
12.678 cabeçalhos de orador da base, com cerca de 90% de acerto auditado
contra 292 rótulos manuais.

O cálculo do sinal tem três passos: cada gestor i vira um vetor
g(i) = (p(i), n(i)) com as frações de linguagem positiva e negativa da sua
fala; calcula-se o centro c da call; e a TD é a média das distâncias
euclidianas ao centro, TD = (1/N) Σ ‖g(i) − c‖, exigindo ao menos dois
gestores. A definição do centro rendeu a principal contribuição
metodológica. Na média simples cada orador pesa igual, e numa call da
Apple de 2020 um orador de 283 palavras, cujas únicas 6 negativas eram o
aviso legal padrão, respondia por um terço do sinal. O tom agregado (cada
palavra pesa igual) corrige a contaminação, e a ponderação pelo volume de
fala leva a correção ao limite; a Seção 5 mostra que o sinal enfraquece
conforme se devolve peso a oradores marginais. A implementação foi
verificada por reimplementação independente nas 32.387 calls elegíveis
(diferença máxima da ordem de 10⁻¹⁶).

## 4. Método de teste

O retorno anormal acumulado (CAR) é a diferença entre o retorno observado
e o previsto pelo CAPM, acumulada nas janelas de 1, 2 e 5 pregões ao redor
do anúncio, com beta estimado em 100 pregões separados do evento; calls
após o fechamento (26% da amostra) são ancoradas no pregão seguinte. O
teste central é um painel com efeitos fixos de empresa e trimestre,
controles de tamanho, valor, alavancagem, rentabilidade, surpresa,
momentum e reversão, erros agrupados por empresa e winsorização em 5/95.
O efeito fixo importa para a leitura: compara-se a Apple de um trimestre
com a própria Apple típica, não a Apple com a Exxon. Reportamos sempre a
estatística t; pela convenção usual, valores acima de 2 em módulo indicam
significância a 5%.

Na implementação, dois backtests independentes, ambos com custos e com
auditoria, nos próprios scripts, de que nenhuma informação futura entra na
seleção. O primeiro compra, ao fim de cada mês, as 10 empresas de maior TD
em pesos iguais (call mais recente dos três meses anteriores), carrega por
um mês e rebalanceia, a 10 pontos-base por lado sobre o giro. O segundo é
um long-short por quintis calendar-time com especificação congelada antes
da execução (entrada no pregão seguinte à call, 63 pregões de manutenção,
custos de 5 pontos-base por perna, 2006 a 2025). Todos os números saem de
scripts versionados; a replicação exige quatro comandos e foi testada em
clone limpo, com resultados idênticos.

## 5. Resultados econométricos

A base reconstruída é comparável à da literatura: a distribuição da TD
(média 0,0086, mediana 0,0080, desvio 0,0047) fica próxima da referência
(0,0079, 0,0074, 0,0048), e os controles batem com os publicados.

### 5.1 H1, precificação no anúncio: confirmada

A tabela reporta as estatísticas t da TD sobre o CAR nas três janelas,
para as três construções do sinal e os dois medidores (2009 em diante,
cerca de 23,6 mil eventos):

| construção do sinal | LM (dicionário) | FinBERT (neural) |
|---|---|---|
| centro = média simples | −0,7 / −0,7 / −0,4 | +0,1 / −0,3 / −0,5 |
| centro = tom agregado | −2,0 / −2,1 / −1,7 | −1,9 / −2,3 / −2,3 |
| ponderada pelo volume de fala | −2,8 / −2,7 / −2,3 | −2,3 / −2,4 / −2,5 |

Três conclusões saem da tabela. A hipótese confirma-se quando o sinal
respeita o volume de fala (a ponderada é significativa em todas as janelas
e medidores; a agregada, na maior parte). O enfraquecimento de baixo para
cima é gradual: quanto mais peso ao ruído dos oradores marginais, mais o
sinal se dilui. E dois medidores que concordam entre si em apenas 38%
chegam ao mesmo desenho. Em termos econômicos, pular do quartil inferior
ao superior da TD custa entre 0,11% e 0,16% de retorno no anúncio.

### 5.2 Retornos subsequentes: confirmados

Em painel mensal, o retorno dos três meses seguintes cresce com a TD
(t=+2,0 nos dois medidores), com controles de valor, momentum, tamanho e
reversão. Junto com a H1, o desenho fecha: o mercado penaliza no anúncio e
cobra prêmio para carregar a empresa divergente, o padrão clássico de
compensação por risco.

### 5.3 H3, incerteza operacional: confirmada

A magnitude da surpresa de lucro do trimestre seguinte cresce com a TD
(t=+2,5 no painel). Na previsão fora da amostra, com modelo treinado só no
passado, a relação alcança t=+3,58 e acerto direcional em 71% dos
trimestres, o resultado preditivo mais forte do projeto: a divergência de
hoje avisa que o próximo balanço tende a vir com surpresa grande.

### 5.4 H2, risco: descartada como sinal

Na amostra completa existe associação entre TD e volatilidade futura
(t=+2,5 no dicionário). A investigação, porém, revelou um vício de
medição: janelas que começam no dia seguinte ao anúncio ainda carregam a
reação ao próprio anúncio, e empresas de TD alta reagem mais (a H1). Fora
desse eco, a associação perde significância (t=+1,3), e nenhuma das 16
especificações predefinidas mostrou poder de previsão. A TD não informa
sobre risco futuro nada que a volatilidade passada já não diga; o canal
foi descartado. Permanece verdadeiro, como descrição, que empresas de TD
alta são mais voláteis, o que retorna na interpretação do backtest.

### 5.5 Robustez temporal

Reestimando tudo apenas com os dados disponíveis até cada data de corte, o
coeficiente da H1 mantém o sinal negativo em todos os cortes, e a previsão
fora da amostra confirma (t=−2,12, negativa em cada ano de 2019 a 2025); a
camada de retornos mantém coeficiente positivo em todos os cortes; a H3
sustenta o t=+3,58. A exceção é a H2, reprovada (t=+0,27), o que motivou o
descarte. O mesmo protocolo que valida duas hipóteses reprova a terceira:
o critério não é complacente.

## 6. Backtest

Resultados da carteira principal em 199 meses (fev/2009 a ago/2025); o
Sharpe é o retorno em excesso por unidade de volatilidade, e o drawdown, a
maior queda acumulada:

| | Top-10 TD bruto | Top-10 TD líquido | Bottom-10 TD | Universo elegível EW | S&P 500 |
|---|---|---|---|---|---|
| Retorno acumulado | +2.061% | +1.783% | +959% | +1.208% | +682% |
| Retorno ao ano | +20,4% | +19,4% | +15,3% | +16,8% | +13,2% |
| Volatilidade a.a. | 20,6% | 20,6% | 17,4% | 17,3% | 14,9% |
| Sharpe | 1,01 | 0,96 | 0,91 | 0,99 | 0,91 |
| Drawdown máximo | −29% | −29% | −27% | −27% | −25% |

A leitura é feita em duas camadas. Contra o S&P 500, o excesso é de +7,1%
ao ano (t=2,60), com vitória em 58% dos meses e em 11 dos 17 anos. Como
carteiras de pesos iguais tendem a superar índices ponderados por valor,
construiu-se a régua mais exigente, o próprio universo elegível em pesos
iguais: contra ela, a seleção por TD adiciona +3,6% ao ano (t=1,59),
positiva em todos os subperíodos (+1,8% de 2010 em diante; +2,6% de 2019
em diante), embora sem significância isolada, consequência esperada de
concentrar em 10 ativos, nos quais o ruído específico domina. A carteira
espelho (menor TD) rende 5,1 pontos menos ao ano. Quanto a cenários, o
top-10 destaca-se em anos de estresse ou recuperação (fechou 2018 em
+0,4% contra −6,2% do índice e caiu menos em 2022, −14,1% contra −19,4%)
e fica para trás em altas amplas como 2016 e 2021. O giro médio é de 35%
ao mês e as composições são verificáveis no repositório.

O long-short por quintis não monetiza: retornos líquidos entre +0,3% e
+4,4% conforme a construção, melhor Sharpe de 0,22 (estatisticamente
nulo) contra 0,53 do mercado. O diagnóstico de pernas explica: correlação
de só 0,6 entre as pontas, e o quintil alto do FinBERT embute exposição a
baixa volatilidade, um fator, não tom. As 6 construções registradas
entram no ajuste de múltiplos testes (Bailey e López de Prado, 2014).

O perfil de risco fecha a interpretação: o top-10 realiza volatilidade
maior (20,6% contra 17,3%) e Sharpe próximo ao do universo (1,01 contra
0,99), ou seja, o retorno extra remunera risco extra, coerente com a
Seção 5.4. A evidência sustenta quatro usos: tilt direcional sobre um
portfólio core; condicionamento de exposição em torno de anúncios;
posicionamento de volatilidade no próximo anúncio das empresas de
divergência alta, apoiado na previsão de surpresa; e insumo em arcabouço
multifator.

## 7. Riscos e limitações

Como toda pesquisa empírica, o trabalho tem limitações que preferimos
declarar. Os papéis de orador foram reconstruídos por classificador
próprio (cerca de 90% de acerto), ruído que as bases comerciais curadas
não têm; algumas variáveis usam proxies (surpresa via Yahoo Finance em vez
de I/B/E/S; dois dos 19 controles não reconstruídos); e cerca de 3 mil
eventos ficaram sem CAR por falta de preços de deslistadas em fonte
gratuita. Na implementação, o efeito no anúncio (cerca de 0,15% por
variação interquartil) não paga custos como arbitragem isolada do evento,
razão pela qual a proposta se concentra em tilt, condicionamento e
volatilidade; e, com 10 ativos, o excesso sobre o universo em pesos iguais
é consistente mas não significativo isoladamente, de modo que diversificar
é o caminho natural de evolução. Por fim, o centro da call admite duas
construções na literatura; reportamos ambas, e a escolha da agregada
seguiu a evidência textual da referência e o diagnóstico de ruído da Seção
3, não o resultado.

## 8. Conclusão e próximos passos

O trabalho verificou se a divergência de tom entre gestores constitui
informação precificada e se pode virar estratégia. As evidências apontam
que sim: o fenômeno foi confirmado em universo independente, sobreviveu à
troca completa do medidor de tom e se materializou em carteira
implementável, com custos e auditoria de ausência de informação futura.

A tese de investimento cabe em um parágrafo. A divergência de tom é
precificada em dois tempos: no anúncio, o mercado penaliza a empresa cuja
diretoria diverge; nos meses seguintes, carregar as empresas de maior
divergência captura um prêmio de compensação (+20,4% ao ano contra +13,2%
do S&P 500 em 16 anos e meio), remunerando o risco que o próprio sinal
identifica. O mesmo escore antecipa a magnitude da surpresa do lucro
seguinte, servindo de termômetro de incerteza pré-anúncio. Registramos
também a fronteira que não avançou: em 2.250 firmas fora do S&P 500 (2019
a 2023), com pré-registro versionado, o painel saiu inconclusivo por
potência; no mesmo período o efeito segue nas large caps (t=−1,96).

Como evolução, três caminhos: a integração multifator do escore, com
contagem de tentativas e Deflated Sharpe; a diversificação além de 10
nomes, com sensibilidade a custos e análise de decaimento; e a revisita a
small e mid caps havendo preços de deslistadas. Encerra-se com a nota de
método que organiza o trabalho: em amostra completa, as três hipóteses
pareciam confirmadas, e foi a robustez temporal que separou os efeitos
reais do artefato. Esse protocolo, documentado no repositório, é o
principal ativo metodológico da equipe.

## 9. Uso de IA generativa no processo

O uso de GenAI é operacional e auditável. No núcleo do pipeline, um
processo por linha de comando (claude -p, modelo Haiku) classificou os
12.678 cabeçalhos de orador, com cache versionado e reprocessamento
determinístico (~4,6h), validado por matriz de confusão contra 292
rótulos manuais (cerca de 90% de acerto) e por classificador
determinístico independente (99% de concordância nos casos frequentes);
prompt, custos e validação estão em docs/GENAI_USAGE.md. Ao longo do
projeto, o assistente (Claude) atuou como par de engenharia e auditoria,
incluindo a auditoria da fórmula do centro da call, sempre com as decisões
econômicas e metodológicas a cargo da equipe. O FinBERT, modelo encoder de
classificação, é contabilizado como NLP/ML, não como IA generativa.

## Referências

Angelo, Johnston, Singh e Wan (2025). Tone Distance: Managerial Tone
Divergence and Market Reaction to Earnings. The Financial Review.

Bailey, D. H., e López de Prado, M. (2014). The Deflated Sharpe Ratio.
The Journal of Portfolio Management, 40(5).

Fama, E. F., e French, K. R. (1997). Industry Costs of Equity. Journal of
Financial Economics, 43(2).

Huang, X., Teoh, S. H., e Zhang, Y. (2014). Tone Management. The
Accounting Review, 89(3).

Lee, J. (2016). Can Investors Detect Managers' Lack of Spontaneity? The
Accounting Review, 91(1).

Loughran, T., e McDonald, B. (2011). When Is a Liability Not a Liability?
The Journal of Finance, 66(1).

Yang, Y., Uy, M. C. S., e Huang, A. (2020). FinBERT: A Pretrained
Language Model for Financial Communications. arXiv:2006.08097.

---

Todos os números deste relatório saem de scripts versionados no
repositório, foram reproduzidos em clone limpo e revalidados na data da
entrega. Nenhum número foi digitado à mão.
