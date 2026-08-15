# CORO: Divergência de tom entre gestores como sinal quantitativo de investimento

Relatório Final · Desafio Quant AI 2026 (Itaú Asset)

> O nome CORO resume a estratégia: numa earnings call, os executivos formam
> um coro que ensaia uma mensagem única, e o robô mede o desalinhamento
> dessas vozes. Quando o coro desafina, há informação.

---

## Sumário executivo

Este relatório apresenta um sinal de investimento que quase ninguém está
olhando: a divergência de tom entre os executivos de uma mesma empresa,
medida pela Tone Distance (TD). A ideia é simples de enunciar e difícil de
copiar. Processamos 33.362 transcrições de 685 empresas do S&P 500 (2005 a
2025) e mostramos que, quando os gestores de uma call desafinam entre si,
o mercado reage: a ação sofre no anúncio (t até −2,8), rende mais nos três
meses seguintes (t=+2,0) e entrega surpresa de lucro maior no trimestre
posterior (t=+3,58 fora da amostra, com 71% de acerto). Poucos sinais
sobrevivem, como este sobreviveu, à troca completa do instrumento de
medida: todo o quadro se repete em dois medidores de tom independentes.

O resultado prático é direto. A carteira que compra mensalmente as 10
empresas de maior TD rendeu +20,4% ao ano contra +13,2% do S&P 500 em 199
meses (fev/2009 a ago/2025): quase o triplo do retorno acumulado do índice,
ganhando inclusive nos dois piores anos do período. E o relatório mostra
também o que reprovou: a hipótese de prever volatilidade caiu na validação
temporal e foi descartada, o que diz muito sobre o rigor com que as outras
duas foram aprovadas. Tudo se reproduz do repositório do projeto com
quatro comandos (código e dados à disposição da banca mediante
solicitação).

## 1. Contexto, problema e identidade do robô

Toda temporada de resultados repete o mesmo ritual. A empresa prepara uma
mensagem única para a sua teleconferência, o CEO a abre com o discurso
ensaiado e, quando começam as perguntas dos analistas, cada executivo
passa a responder com o tom que consegue sustentar ao vivo. O texto é
coordenado; o tom, nem sempre. Nesse descompasso reside a oportunidade
que motiva este trabalho, uma vez que um CFO visivelmente mais cauteloso
que o CEO na mesma call revela desacordo interno, ou incerteza sobre o
trimestre, que dificilmente chegaria a um press release. Enquanto o
mercado lê o que a empresa diz, o CORO mede o que a empresa deixa
escapar. Dessa observação nascem as duas perguntas que estruturam a
pesquisa: é possível medir a divergência de tom de forma sistemática? E,
uma vez medida, ela tem valor econômico?

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
transcrição. Essa velocidade importa: o prêmio do sinal se realiza ao
longo de meses, então quem o calcula no dia do evento chega antes. Para a
mesa, é insumo imediato: sinal de seleção, condicionante de exposição a
anúncios e termômetro de incerteza. A identidade visual traduz o
mecanismo: o robô ouvinte, de fones, diante de vozes que chegam alinhadas
e uma que desafina.

![mascote](identidade/coro_mascote.png)

## 2. Referencial teórico

A literatura de finanças textuais já estabeleceu que o tom da comunicação
corporativa move preços: Loughran e McDonald (2011) construíram o
dicionário de tom para finanças; Huang, Teoh e Zhang (2014) mostraram que
gestores administram o tom e que o mercado reage ao seu componente
anormal; Lee (2016) documentou que investidores penalizam executivos
presos a roteiros. A medida deste trabalho vem de Angelo, Johnston, Singh
e Wan (2025), que deslocam a pergunta do tom médio, já muito explorado,
para um território quase intocado: a divergência entre os executivos da
mesma call, a Tone Distance. Partindo dessa referência, fizemos o que
diferencia replicação de convicção: um teste independente, com universo,
dados e código próprios, e toda a análise refeita trocando o medidor de
tom pelo FinBERT (Yang, Uy e Huang, 2020), modelo neural que classifica
cada sentença pelo contexto. A troca é a contraprova do instrumento:
fenômeno real aparece nos dois medidores, e este apareceu.

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
gestores. Foi na definição desse centro que o trabalho produziu sua
contribuição mais valiosa, e ela é uma barreira de entrada: medir isso bem
não é trivial. Na média simples cada orador pesa igual, e numa call da
Apple de 2020 um orador de 283 palavras, cujas únicas 6 negativas eram o
aviso legal padrão, respondia por um terço do sinal. O tom agregado (cada
palavra pesa igual) corrige a contaminação, e a ponderação pelo volume de
fala leva a correção ao limite; a Seção 5 mostra que o sinal enfraquece
conforme se devolve peso a oradores marginais. Quem medir do jeito errado
não encontra nada, e é provavelmente por isso que o sinal segue vivo. A
implementação foi verificada por reimplementação independente nas 32.387
calls elegíveis (diferença máxima da ordem de 10⁻¹⁶).

## 4. Método de teste

O retorno anormal acumulado (CAR) é a diferença entre o retorno observado
e o previsto pelo CAPM, acumulada nas janelas de 1, 2 e 5 pregões ao redor
do anúncio, com beta estimado em 100 pregões separados do evento; calls
após o fechamento (26% da amostra) são ancoradas no pregão seguinte. O
teste central é um painel com efeitos fixos de empresa e trimestre,
controles de tamanho, valor, alavancagem, rentabilidade, surpresa,
momentum e reversão, erros agrupados por empresa e winsorização em 5/95.
O efeito fixo torna a comparação exigente: mede-se a Apple de um trimestre
contra a própria Apple típica, não contra a Exxon. Reportamos sempre a
estatística t; pela convenção usual, valores acima de 2 em módulo indicam
significância a 5%.

Na implementação, dois backtests independentes, ambos com custos. O
primeiro compra, ao fim de cada mês, as 10 empresas de maior TD em pesos
iguais (call mais recente dos três meses anteriores), carrega por um mês e
rebalanceia, a 10 pontos-base por lado sobre o giro. O segundo é um
long-short por quintis calendar-time com especificação congelada antes da
execução (entrada no pregão seguinte à call, 63 pregões de manutenção,
custos de 5 pontos-base por perna, 2006 a 2025). As defesas contra vieses
são estruturais, não promessas: contra look-ahead, auditoria automática de
datas nos scripts e fundamentos pela data de protocolo; contra
overfitting, especificação congelada, período completo sem seleção e
contagem de todas as tentativas para ajuste de múltiplos testes; contra
sobrevivência, universo com as empresas que saíram do índice. Todos os
números saem de scripts versionados; a replicação exige quatro comandos e
foi testada em clone limpo, com resultados idênticos.

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

A tabela conta três histórias favoráveis de uma vez. A hipótese confirma
quando o sinal respeita o volume de fala (a ponderada é significativa em
todas as janelas e nos dois medidores; a agregada, na maior parte). O
enfraquecimento gradual de baixo para cima mostra que sabemos exatamente
de onde vem o sinal e de onde vem o ruído. E dois medidores que concordam
entre si em apenas 38% chegam ao mesmo desenho, o teste de estresse mais
duro que um sinal textual pode passar. Em termos econômicos, pular do
quartil inferior ao superior da TD custa entre 0,11% e 0,16% de retorno no
anúncio.

### 5.2 Retornos subsequentes: confirmados

O anúncio é só o começo. Em painel mensal, o retorno dos três meses
seguintes cresce com a TD (t=+2,0 nos dois medidores), com controles de
valor, momentum, tamanho e reversão. O desenho fecha com elegância
econômica: o mercado penaliza na hora e paga prêmio a quem tem estômago
para carregar a empresa divergente. É esse prêmio, persistente e
mensurável, que a carteira da Seção 6 captura.

### 5.3 H3, incerteza operacional: confirmada

A magnitude da surpresa de lucro do trimestre seguinte cresce com a TD
(t=+2,5 no painel). Na previsão fora da amostra, com modelo treinado só no
passado, a relação alcança t=+3,58 e acerto direcional em 71% dos
trimestres. Poucas variáveis antecipam surpresa de lucro com essa força: a
divergência de hoje avisa, com um trimestre de antecedência, que o próximo
balanço tende a vir fora do roteiro. Para quem opera volatilidade em torno
de eventos, isso é um mapa.

### 5.4 H2, risco: descartada como sinal

Na amostra completa existe associação entre TD e volatilidade futura
(t=+2,5 no dicionário). A investigação, porém, revelou um vício de
medição: janelas que começam no dia seguinte ao anúncio ainda carregam a
reação ao próprio anúncio, e empresas de TD alta reagem mais (a H1). Fora
desse eco, a associação perde significância (t=+1,3), e nenhuma das 16
especificações predefinidas mostrou poder de previsão. Descartamos o canal
sem hesitar, e esse descarte é um argumento de venda do processo: quem
reprova a própria hipótese quando os dados mandam merece crédito quando
aprova as outras duas.

### 5.5 Robustez temporal

Reestimando tudo apenas com os dados disponíveis até cada data de corte, o
coeficiente da H1 mantém o sinal negativo em todos os cortes, e a previsão
fora da amostra confirma (t=−2,12, negativa em cada ano de 2019 a 2025); a
camada de retornos mantém coeficiente positivo em todos os cortes; a H3
sustenta o t=+3,58. Nada aqui depende de um subperíodo de sorte. A exceção
é a H2, reprovada (t=+0,27) pelo mesmo protocolo que aprovou as demais.

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

Os números de manchete são raros de encontrar juntos: excesso de +7,1% ao
ano sobre o S&P 500 (t=2,60, significativo mesmo sob a convenção mais
exigente), vitória em 58% dos meses e em 11 dos 17 anos, e quase o triplo
do retorno acumulado do índice, líquido de custos. E a estratégia mostra o
seu melhor exatamente quando mais importa: fechou 2018 em alta de 0,4%
enquanto o índice caía 6,2%, e caiu 5,3 pontos a menos que o mercado em
2022 (−14,1% contra −19,4%). Fica atrás em altas amplas como 2016 e 2021,
o comportamento esperado de uma carteira concentrada, e o giro de 35% ao
mês é plenamente executável em large caps, as ações mais líquidas do
mundo.

Fomos além da comparação fácil. Como carteiras de pesos iguais tendem a
superar índices ponderados por valor, medimos a seleção contra a régua
mais dura possível, o próprio universo elegível em pesos iguais: a TD
adiciona +3,6% ao ano (t=1,59), positiva em todos os subperíodos (+1,8% de
2010 em diante; +2,6% de 2019 em diante), sem depender de nenhum ano
excepcional. A significância isolada ainda não fecha com 10 ativos, em que
o ruído específico domina, mas a direção é consistente e a carteira
espelho (menor TD) rende 5,1 pontos menos ao ano, exatamente como a tese
prevê. O long-short por quintis, a contraprova, não monetiza (melhor
Sharpe 0,22 contra 0,53 do mercado; a ponta FinBERT embute fator de baixa
volatilidade), e é por isso que a proposta é o tilt comprado, não o
long-short; as 6 construções registradas entram no ajuste de múltiplos
testes (Bailey e López de Prado, 2014).

O perfil de risco completa o argumento: o top-10 realiza volatilidade
maior (20,6% contra 17,3%) e Sharpe acima do índice e em linha com o
universo (1,01 contra 0,91 e 0,99), ou seja, o investidor é pago pelo
risco que aceita, coerente com a Seção 5.4. A evidência sustenta quatro
usos imediatos: tilt direcional sobre um portfólio core; condicionamento
de exposição em torno de anúncios; posicionamento de volatilidade no
próximo anúncio das empresas de divergência alta, apoiado na previsão de
surpresa; e insumo em arcabouço multifator.

## 7. Riscos e limitações

Como toda pesquisa empírica, o trabalho tem limitações, e conhecê-las com
precisão é parte do valor. Os papéis de orador foram reconstruídos por
classificador próprio (cerca de 90% de acerto), ruído que as bases
comerciais curadas não têm, o que sugere que nossos resultados são, se
algo, conservadores; algumas variáveis usam proxies (surpresa via Yahoo
Finance em vez de I/B/E/S; dois dos 19 controles não reconstruídos); e
cerca de 3 mil eventos ficaram sem CAR por falta de preços de deslistadas
em fonte gratuita. Na implementação, o efeito no anúncio (cerca de 0,15%
por variação interquartil) não paga custos como arbitragem isolada do
evento, razão pela qual a proposta se concentra em tilt, condicionamento e
volatilidade; e, com 10 ativos, o excesso sobre o universo em pesos iguais
é consistente mas não significativo isoladamente, de modo que diversificar
é o caminho natural de evolução, com espaço claro para melhorar. Por fim,
o centro da call admite duas construções na literatura; reportamos ambas,
e a escolha da agregada seguiu a evidência textual da referência e o
diagnóstico de ruído da Seção 3, não o resultado.

## 8. Conclusão e próximos passos

O trabalho perguntou se a divergência de tom entre gestores constitui
informação precificada e se pode virar estratégia. A resposta é sim nas
duas pontas, e com margem: o fenômeno foi confirmado em universo
independente, sobreviveu à troca completa do medidor de tom e virou uma
carteira implementável que entregou quase o triplo do acumulado do índice,
líquida de custos e auditada contra informação futura.

A tese cabe em um parágrafo. A divergência de tom é precificada em dois
tempos: no anúncio, o mercado penaliza a empresa cuja diretoria diverge;
nos meses seguintes, carregar as empresas de maior divergência captura um
prêmio de compensação (+20,4% ao ano contra +13,2% do S&P 500 em 16 anos e
meio), remunerando um risco que o próprio sinal identifica e antecipa. O
mesmo escore avisa, com um trimestre de antecedência, quando o próximo
balanço tende a surpreender. É um sinal novo, rápido, difícil de copiar e
barato de operar, construído sobre dados públicos. Registramos também a
fronteira que não avançou: em 2.250 firmas fora do S&P 500 (2019 a 2023),
com pré-registro versionado, o painel saiu inconclusivo por potência; no
mesmo período o efeito segue nas large caps (t=−1,96).

Como evolução, três caminhos já mapeados: a integração multifator do
escore, com contagem de tentativas e Deflated Sharpe; a diversificação
além de 10 nomes, com sensibilidade a custos e análise de decaimento; e a
revisita a small e mid caps havendo preços de deslistadas. Encerra-se com
a nota de método que organiza o trabalho: em amostra completa, as três
hipóteses pareciam confirmadas, e foi a robustez temporal que separou os
efeitos reais do artefato. Esse protocolo, documentado no repositório, é o
principal ativo metodológico do projeto, e viaja junto com a estratégia.

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

Todos os números deste relatório saem de scripts versionados, foram
reproduzidos em clone limpo e revalidados na data da entrega. Nenhum
número foi digitado à mão. Código, dados e documentação completa ficam à
disposição da banca mediante solicitação.
