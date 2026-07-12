# Replicação S&P 500 completo — resultados (2026-07-12)

Teste pré-comprometido: replicar o H1 e o prêmio do Angelo (2025) no universo
~igual ao do paper (S&P 500, todos os setores), com a construção FIEL (LM,
palavras) validada pela distribuição, especificação completa (7 controles + SUE,
FE firma+trimestre, cluster por firma, winsor 5/95) e PODER adequado.

## Sanity checks (todos pré-comprometidos; ordem de leitura: antes dos β)

| check | resultado | critério |
|---|---|---|
| Distribuição TD_LM vs paper | mean 0,0085 / median 0,0079 / sd 0,0049 / CV 0,576 | 0,0079 / 0,0074 / 0,0048 / 0,61 ✅ quase exato |
| corr(z, gestor mais quieto) | +0,063 | ≈ 0 ✅ (crua: −0,272 — viés presente como no paper) |
| sd do CAR[-1,+1] | 6,2% | < 7,6% de tech ✅ |
| beat rate SUE | 72% | ~75–80% ✅ aproximado |
| cobertura | 28.989 CARs; 22,4k obs nas regressões; **481–483 firmas** | 5× os clusters de tech ✅ |

(Bug pego pelo sanity: 1ª rodada veio com ^GSPC vazio → CAR=0 casos; corrigido o
fetch MultiIndex do índice e re-rodado. Nada foi interpretado antes do conserto.)

## Resultados (β por desvio-padrão da TD; t entre parênteses)

| | H1: CAR anúncio (esp. −) | prêmio 1m (esp. +) | prêmio 3m (esp. +) |
|---|---|---|---|
| TD crua (fiel ao paper) | −0,0002 (−0,63) | −0,0004 (−0,71) | +0,0009 (+0,96) |
| TD_adj (debias null) | **+0,0010 (+2,93)** ⚠ | +0,0002 (+0,43) | −0,0004 (−0,45) |
| z (debias null, padronizado) | +0,0001 (+0,39) | +0,0004 (+0,77) | −0,0001 (−0,17) |

n ≈ 22,4k · firmas ≈ 481 · R² 0,12–0,24.

## Leitura econométrica

1. **Agora o teste TEM poder — e rejeita o efeito do tamanho publicado.**
   SE do coeficiente H1 (TD crua) ≈ 0,0003. O efeito do Angelo (−0,00156/SD)
   implicaria t ≈ −5. Medimos t = −0,63; IC 95% = [−0,0008, +0,0004] — **exclui
   o efeito do paper por larga margem**. Não é mais "sem poder": é o efeito
   publicado NÃO replicando em large caps 2005–2025 com construção fiel
   (distribuição validada) e especificação completa.
2. **O prêmio (a premissa da estratégia) tampouco aparece** — |t| < 1 em todas
   as specs, nos dois horizontes.
3. **Anomalia a investigar (regra pré-comprometida: sinal "errado" significativo
   → investigar, não celebrar):** TD_adj com +2,93 no H1 — positivo (contrário
   ao paper) e SIGNIFICATIVO. Fragilidades: o z (padronização mais limpa do
   mesmo debias) mostra ZERO (t=0,39) — inconsistência entre as duas formas
   debiasadas sugere que o td_adj pode estar carregando estrutura de escala do
   null (nº de gestores/palavras), não desacordo. NÃO usar sem investigação.
   Nota relacionada: E[null] > TD observada em média (gestores mais PARECIDOS
   entre si do que o acaso) — "coordenação" é o fenômeno dominante nas calls.

## Reconciliação com o paper (hipóteses, honestas)

- **Composição da amostra (principal):** Angelo = 188k calls, 7.526 firmas
  (Capital IQ, inclui small/mid caps); nós = S&P 500 (large caps, 683 firmas).
  Anomalias de atenção/processamento de informação tipicamente vivem em small
  caps, onde a cobertura de analistas é menor. Nosso dado NÃO permite testar
  small caps (fronteira estrutural do dataset aberto).
- Fonte de transcrição (Capital IQ vs dataset aberto): atribuição de orador pode
  diferir.
- Parte do efeito publicado pode refletir o artefato de volume (corr −0,27
  também na construção dele) + especificidades de amostra.
- Período: incluímos 2023–2025 (pós-publicação; efeitos publicados decaem —
  McLean & Pontiff).

## Limitações declaradas desta replicação

- Survivorship: 77/685 tickers sem preço no Yahoo (deslistadas) ficam fora dos
  CARs; 7.491 calls sem setor (deslistadas) ficam sem industry_tone.
- mktcap extrapolado pré-2015 em 11.728 eventos (flag).
- Papéis: 12.678 headers via LLM validado + 56.809 via determinístico (99% de
  concordância nos frequentes; não re-validado por amostra no S&P completo).
