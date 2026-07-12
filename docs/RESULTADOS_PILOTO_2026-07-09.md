# Resultados do PILOTO — 2026-07-09 (amostra, NÃO o resultado final)

**Aviso:** tudo abaixo é uma **amostra piloto de 1.200 calls** (embaralhamento
determinístico, seed 42) ≈ 22% da base. Serve para validar o método e ver a
direção; **os números finais virão do run completo** (5.452 calls, rodando em
background, ~20h). Nada aqui é para o relatório ainda.

## Stage 2 — scoring por sentença (pronto no piloto)

1.200 calls, **511.174 sentenças** pontuadas (~29 sent/s, 4,9h, CPU). Validação
do P1: a fração de sentenças positivas por gestor ocupa a banda informativa
(0.05–0.95) em **~70%** dos gestores, vs **16,5%** no scoring antigo por fala. O
sensor novo de-satura — no nível do gestor, que é o que importa.

## Stage 3 — Tone Distance + permutation null (1.084 calls com TD)

| métrica | valor | paper (L-M) |
|---|---|---|
| TD crua: mean / median / sd / CV | 0.161 / 0.163 / 0.054 / **0.338** | 0.0079 / 0.0074 / 0.0048 / 0.61 |
| E[TD_null] médio | 0.069 | — |
| TD_adj (= TD − E[null]) médio | 0.092 | — |

**P2 (viés de volume) — confirmado e tratado:**

| correlação | TD crua | após debias |
|---|---|---|
| × sentenças do gestor mais quieto | **−0.436** | z: **+0.019** ✅ (TD_adj: −0.172) |
| × nº de gestores | +0.048 | — |
| × tom positivo médio da call | +0.601 | (permanece: +0.63) |

Leituras:
- **~43% da TD crua é ruído** de amostragem (E[null]/TD = 0.069/0.161).
- **O sinal padronizado `z` passa no sanity check**: corr(z, gestor mais quieto)
  = +0.019 ≈ 0. O permutation null funciona — `z` é o sinal limpo do viés de
  volume. Por isso a decision rule (|corr crua| = 0.436 > 0.25) manda usar o
  **null como sinal primário**.
- O viés de **nº de gestores sumiu** (+0.048; era forte no scoring por fala).
- Sobra o confundidor de **nível de tom** (+0.60) — NÃO é volume, o null não
  remove; trata-se com os **controles do Angelo** (`disclosure_tone`) na regressão.

## Stage 4 — replicação do Angelo (FE empresa+trimestre, cluster empresa)

Coeficiente = efeito de +1 desvio-padrão da TD (winsorizada 5/95). Universo
`strict_tech` (852 calls; core dá quase idêntico).

| regressão | sinal esperado | `z` (debiasado) | t | p |
|---|---|---|---|---|
| **H1** CAR[-1,+1] | negativo | +0.0034 | 0.67 | 0.50 |
| **Prêmio 1 mês** ret_21 | positivo | **+0.0063** | **1.35** | 0.18 |
| **Prêmio 3 meses** ret_63 | positivo | +0.0056 | 0.94 | 0.35 |

- **H1 (CAR): nada** — consistente com a limitação declarada: entramos em **T+1**,
  então a janela do anúncio que o Angelo mede não é observável aqui.
- **Prêmio: sinal CERTO (positivo), mas subdimensionado** — t ≈ 1.0–1.35, não
  significativo. É a direção da tese (comprar TD alta rende mais em 1–3 meses).
- O `z` dá os maiores coeficientes na direção certa; resultado estável entre
  universos strict/core.

## Interpretação honesta (não superinterpretar)

No piloto (~22% dos dados) o efeito do prêmio é **direcionalmente correto e não
significativo**. Se o efeito for real, a amostra completa (~4× mais calls no
strict) escalaria o t por ~√4 = 2× → ret_21 `z` iria de 1.35 para ~2.7
(significativo). **Mas isso é uma projeção, não um resultado** — o run completo
é o juiz. Também é possível que continue não-significativo, e aí o resultado
honesto é "efeito fraco/ausente nesta base", que o edital valoriza mais que um
número inflado.

## Próximo (automático, quando o run completo terminar ~amanhã)

Re-rodar Stage 3 + Stage 4 na base inteira; rodar a sensibilidade com/sem blocos
de universo; medir o `statistical power`. Depois: event study e backtest (Stages
5–6) com o sinal `z`.
