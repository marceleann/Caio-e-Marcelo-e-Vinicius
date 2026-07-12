# Resultados — BASE COMPLETA (2026-07-10)

Scoring por sentença completo: **5.452 calls, ~2,36M sentenças** (CPU, ~23,5h).
Tone Distance re-agregada com dois conjuntos de papéis: heurística (regex) e
**LLM** (Stage 1 GenAI). Comparação honesta abaixo. Nada aqui é inflado.

## Stage 3 — diagnóstico do sinal (base completa)

| | heurística | papéis LLM |
|---|---|---|
| corr(TD_crua, gestor mais quieto) | −0,417 | −0,406 |
| corr(z, gestor mais quieto) — sanity | +0,054 ✓ | +0,133 |
| E[null]/TD (fração de ruído) | ~43% | ~54% |
| gestores/call | concentra em 3 | espalha 3–8 |

O LLM recupera mais gestores por call (a heurística rotulava alguns como
analista) — fiel ao Angelo ("todos os representantes da empresa"), mas adiciona
ruído de amostragem (E[null] sobe). O permutation null (`z`) segue passando no
sanity nas duas fontes.

## Stage 4 — H1 e prêmio (universo strict, FE empresa+trimestre, cluster empresa)

Coeficiente por desvio-padrão da TD (winsorizada 5/95). n ≈ 3.200–3.600.

**H1 — CAR[-1,+1]** (esperado NEGATIVO):

| sinal | heurística | LLM |
|---|---|---|
| TD crua | +0,003 (t=1,54) | −0,003 (t=−1,23) |
| z | +0,003 (t=1,58) | −0,000 (t=−0,23) |

Com papéis LLM o sinal fica NEGATIVO (direção do Angelo), mas não-significativo.
Consistente com a limitação declarada: entramos em **T+1**, não observamos a
janela do anúncio.

**Prêmio 1 mês — ret_21** (esperado POSITIVO):

| sinal | heurística | LLM |
|---|---|---|
| TD crua (enviesada) | +0,37%/dp, **t=2,42, p=0,016** | −0,08%, t=−0,47 |
| **z (sinal a priori)** | +0,34%, t=1,89, p=0,06 | +0,19%, t=1,28, p=0,20 |

**Prêmio 3 meses — ret_63**: nulo nas duas fontes (t < 1,2).

## Interpretação (honesta, sem spin)

1. **Mais dados confirmaram a direção do piloto** (prêmio positivo em 1 mês).
2. **A significância da TD crua (t=2,42) é provavelmente ARTEFATO de viés**: some
   com papéis limpos (LLM → t=−0,47). Isto valida termos fixado a priori o sinal
   **debiasado** (`z`), não a TD crua.
3. **O sinal `z` é robusto em DIREÇÃO** (prêmio +1 mês positivo nas duas fontes)
   **mas marginal em significância** (t≈1,3–1,9). Prêmio 3 meses nulo.
4. **H1 (CAR)** fica no sinal certo (negativo) com papéis LLM, não-significativo.
5. **Efeito modesto e frágil**, sensível ao método — direção coerente com o paper,
   significância marginal. NÃO é um resultado forte. Declarado como tal.

## O que NÃO fizemos (anti-p-hacking)

Não buscamos especificações até achar p<0,05. Todas as variantes testadas (fonte
de papel × sinal × horizonte) estão reportadas acima, inclusive as nulas — isso
alimenta o `n_trials` do Deflated Sharpe (Stage 7).

## Rodada de correções (2026-07-10, após revisão crítica do Marcelo)

Quatro correções aplicadas, uma a uma, com o efeito medido em cada passo:

1. **Janela do H1 corrigida** (erro real): o CAR estava ancorado no decision_date
   (T+1) com janela [-1,+10]; o H1 do Angelo é MEDIÇÃO no anúncio, CAR[-1,+1] na
   data da call (`compute_announcement_car.py`, market model, 5.030 eventos).
   Efeito: com papéis heurísticos, H1 "significativo" mas no SINAL ERRADO (z:
   t=+3,0) — artefato de má classificação; com papéis LLM, direção certa
   (negativo) e fraco (t≈−0,6 a −1,4).
2. **Controles corrigidos**: `ln_mktcap` real (close BRUTO × shares outstanding;
   yfinance get_shares_full — cobertura 2015+, extrapolado p/ trás com flag em
   1.888 eventos) + `industry_tone` PIT 90d (`build_controls_v2.py`). Efeito:
   marginal (H1 z: −0,56→−0,68).
3. **Papéis de alta confiança** (LLM ∩ determinístico concordam): achatou tudo
   (t≈0,9) — cortar ruído NÃO revelou efeito escondido.
4. **Baseline Loughran-McDonald** (réplica FIEL do sensor do Angelo: frações de
   PALAVRAS, dicionário oficial 354/2.355, texto integral, papéis LLM —
   `stage3_lm_baseline.py`). A DISTRIBUIÇÃO BATE COM O PAPER (mean 0,0098 vs
   0,0079; median 0,0093 vs 0,0074; sd 0,0042 vs 0,0048) → construção validada.
   MAS a regressão dá ZERO em tudo (|t|<0,9, H1 e prêmio). **Não era o sensor:**
   o FinBERT não estava nos atrapalhando (até tinha mais direção que o LM).

## Power analysis (a peça que fecha a leitura)

Efeito do Angelo ≈ 0,156% por SD. Com os SEs REAIS das nossas regressões:
- **Poder de detecção a 5%: entre 13% e 43%** conforme a spec (precisaria ~80%).
- IC 95% do coef LM H1: **[−0,0024, +0,0026]** — contém AO MESMO TEMPO o efeito
  do Angelo (−0,0016) e o zero.

**Veredito honesto: nossa amostra (94 empresas tech, ~3.600 calls nas regressões,
cluster por empresa) NÃO TEM PODER para confirmar nem refutar o efeito do Angelo.**
O resultado não é "o efeito não existe"; é "não é detectável nesta escala".

## Fato novo relevante

O dataset bruto (`data/raw/calls_all.parquet`, ~33k calls do S&P 500 INTEIRO,
todos os setores) JÁ ESTÁ EM DISCO — nós o filtramos para tech cedo (ADR-001).
Uma réplica LM + papéis determinísticos no S&P 500 completo (~5× mais clusters,
poder estimado ~70–85%) é factível SEM FinBERT (contagem de palavras = minutos;
gargalo = baixar preços de ~500 tickers, ~1h). É o teste que decide a tese.
