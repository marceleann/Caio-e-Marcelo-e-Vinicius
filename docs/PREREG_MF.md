# Pré-registro — extensão small/mid caps (Motley Fool)

**Escrito ANTES de qualquer regressão de retorno ou backtest neste universo**
(21-22/07/2026; TD construída e validada por distribuição, preços em download;
nenhum CAR calculado até aqui). Congela hipóteses, especificações e critérios
de leitura — contra mineração de especificação. Mudanças posteriores a este
documento serão declaradas como exploratórias.

## Amostra

- Universo: 16.897 calls com TD (2.851 tickers, 2019-04 a 2023-02).
- **Análise principal: só os ~2.271 tickers FORA do nosso S&P 500** (teste
  em universo genuinamente novo); braço com todos como robustez.
- Limitações declaradas de antemão: período curto (~3,5 anos, metade em
  2021, inclui COVID e bear de 2022); cobertura de preços parcial em
  deslistadas (yfinance); sem fundamentos EDGAR nesta fase (controles
  textuais + tamanho apenas). Resultados serão INDICATIVOS.

## H1 no painel (teste central)

- CAR CAPM idêntico ao S&P: estimação 100 pregões (mín. 70), gap 50,
  janelas [-1,+1], [-1,+2], [-1,+5]; mercado ^GSPC; rf Ken French.
- Regressão: CAR ~ sinal + disclosure_tone + analyst_tone +
  analyst_tone_disp + length + lagged_td + ln_mktcap (se cobertura ≥70%)
  + FE firma + FE ano-tri; cluster por firma; winsor 5/95.
- Sinais testados (fixados): **td_b** (centro agregado) e **td_w**
  (ponderada) — as duas construções que replicaram no S&P; td_a reportada
  como referência histórica. Sem outras variantes.
- **Hipótese registrada:** coeficientes negativos; magnitude (efeito
  interquartil) MAIOR ou igual à do S&P (−0,11% a −0,16%), conforme nota 6
  do artigo (efeito mais forte em firmas menos complexas).

## Backtest (UMA especificação principal + UMA robustez)

- **Principal (comparabilidade):** espelho exato da spec congelada do S&P —
  calendar-time, entrada no fechamento T+1, 63 pregões, tranches
  sobrepostas, quintis extremos por percentil point-in-time dos 90 dias
  anteriores (mín. 20 eventos), long TD alta / short TD baixa, custos de
  5 bps por perna. Sinal: **td_w**. Custos sensibilizados a 10 bps
  (small caps) reportados junto.
- **Robustez (única):** mesmo desenho com sinal relativo ao histórico da
  própria firma (percentil contra calls passadas, mínimo 4).
- **Critério de leitura registrado:** declararemos "monetiza" apenas se o
  long-short líquido tiver Sharpe ≥ 0,5 no período; abaixo disso, o
  resultado entra como indicativo (t de ~3,5 anos não fecha significância
  sem Sharpe alto — dito de antemão). Comparação-chave: spread vs os
  +0,3% a.a. do S&P.

## O que NÃO faremos

Sem novas variantes de sinal, holding, thresholds ou custos após ver
resultados; sem escolha de subperíodo a posteriori; sem trocar o universo
principal. Contagem honesta de tentativas do projeto (para DSR): S&P — 4
sinais × 1 spec; MF — 2 sinais × 1 spec + 1 robustez.

*Commit deste arquivo = timestamp do pré-registro.*
