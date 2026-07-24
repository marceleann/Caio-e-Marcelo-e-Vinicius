# CORO — Divergência de tom entre gestores como sinal quantitativo

**Pré-relatório · Desafio Quant AI 2026 (Itaú Asset)**
Equipe: Marcelo e Caio · Orientação: Profa. Nadia Cardoso Moreira
Repositório (código + dados derivados + replicação em 4 comandos):
https://github.com/marceleann/Caio-e-Marcelo-e-Vinicius

> **[NOME A CONFIRMAR PELO TIME]** Proposta: **CORO** — o robô escuta o coro
> de gestores de cada earnings call; quando o coro desafina, o mercado ouve.
> O nome resume a tese em uma imagem: não medimos o tom médio da empresa, e
> sim o desalinhamento entre as vozes que a representam.

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
horizonte longo). Caracterização final: o "canal de risco" é sobretudo o
prolongamento da reação ao anúncio, já contado na H1; risco de médio prazo
genuinamente preditivo não existe nos nossos dados, e não o utilizamos.

**H3 — operacional.** O valor absoluto da surpresa de lucro seguinte aumenta
com a TD (t = +2,2 na amostra ampla), com a mesma concentração de regime.

**Retornos subsequentes (Tabela 6 do artigo).** Painel mensal (retorno dos 3
meses após a call; efeitos fixos de empresa e mês; controles de valor,
momentum, tamanho e reversão): coeficiente positivo nos dois sensores (t =
+2,1 LM; +1,8 FinBERT; artigo: +2,2) — consistente com o mecanismo do artigo:
o mercado penaliza no anúncio e exige retorno maior depois. (Validação
walk-forward desta camada: pendente — declarado.)

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

## 5. Backtest

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
0,53. Nenhuma construção monetiza em quintis.

**Leitura honesta e o que a estratégia pode (e não pode) prometer.** A
bateria completa de testes de implementação — carteira de quintis, previsão
por evento em todos os momentos, out-of-sample 2023–25 — dá o mesmo
veredito: **ordenar ações por TD, sozinho, não gera alpha negociável em
large caps.** O prêmio identificado no painel é condicional (dentro da
empresa, com controles), pequeno (~15 bps por variação interquartil) e é
soterrado por características de firma no corte cruzado (diagnóstico
explícito: o quintil alto do sinal FinBERT embute aposta em baixa
volatilidade — correlação entre pernas de só 0,6).

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
long-short isolado por ordenação crua (reprovado e explicado). A
engenharia dessas aplicações é o trabalho da entrega final; o walk-forward
da camada de retorno mensal (Tabela 6) segue pendente e declarado.

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
5. **Regime e validação temporal**: o canal de risco (H2) reprova no
   walk-forward (extinto após ~2014) e o de operacional (H3) é fraco
   prospectivamente; só o canal de preço (H1) atravessa todos os testes
   temporais — e apenas na forma condicional, não como ordenação crua.
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
1. **Walk-forward da camada de retorno (Tabela 6)** — o teste que decide se
   a perna de tilt pode ser prometida; sem ele, não será;
2. **Desenho da integração multifator** do escore CORO (interações com
   fatores e regimes), com contagem honesta de tentativas e Deflated Sharpe;
3. Robustez final: sensibilidade a custos, análise de decay;
4. Se houver dados melhores (preços de deslistadas), revisitar a fronteira
   small/mid com o pré-registro já commitado.

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
