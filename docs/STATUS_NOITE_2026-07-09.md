# Status da noite — 2026-07-09 (para o Marcelo revisar de manhã)

Trabalho autônomo enquanto você dormia. Tudo baseado em números que rodaram de
verdade; nada inventado. O que ainda não terminou está marcado como **em curso**.

## 1. Reavaliação do universo (feita)

Ver `docs/UNIVERSO_REAVALIACAO.md`. Resumo: núcleo de tech inequívoco (87 tickers,
4.153 calls) + 5 blocos econômicos discutíveis (payments, non_gics_tech, payroll,
solar_tech, ems) + 1 novo toggle proposto `core_borderline` (GRMN, TDY, FTV, VNT,
ROP, LDOS, UIS — instrumentos/defesa/conglomerado). **Nada foi removido**: criei
`data/interim/call_blocks.parquet` etiquetando cada call, então "com/sem" é um
filtro nos Stages 3–4. Você decide o recorte; o primário sugerido é `strict_tech`
(core + borderline = 4.378 calls).

## 2. Diagnóstico reproduzido (feito)

- **P1 (saturação) confirmado exatamente:** p_pos<0.01 em 53,5%, >0.99 em 21,4%,
  banda intermediária 16,5% (idêntico ao prompt).
- **P2 (viés de volume) confirmado, porém MENOR que no prompt:** corr(TD antiga,
  sentenças do gestor mais quieto) = **−0,262** (prompt: −0,545). Motivo provável:
  o repo já mitigou parte via ADR-025 (filtro só-Q&A). Achado novo: corr(TD, tom
  positivo médio) = **+0,464** — confundidor forte não previsto.
- Contagens reconciliadas: 5.452 calls, 116 tickers com call, universo=150 (34 sem
  call). O "166" não existe nos dados (era candidatos/memória).

## 3. Stage 2 — scoring por sentença (em curso)

**Sem GPU** (torch cpu-only). Benchmark real: ~28–31 sent/s; quantização int8 só
deu 1,2× (não ajuda). Full = ~23h. Por isso: **piloto de 1.200 calls (~4,5h)** em
background agora; run completo depois.

**Validação crítica do P1 (o motivo de todo o rebuild):** a de-saturação NÃO
aparece na sentença crua (o FinBERT é quase one-hot por sentença — cada sentença é
um "voto"). Ela aparece no **nível do gestor**, agregando ~48 sentenças (mediana):
a *fração de sentenças positivas* por gestor vai de **16,5% → ~70%** na banda
informativa. É o análogo exato do Loughran-McDonald do Angelo (fração de palavras
positivas). **O sensor novo funciona.** (Registro honesto: meu 1º teste mediu o
nível errado e quase me fez abortar — corrigido.)

## 4. Stages 3 e 4 — escritos e validados, rodam quando o piloto terminar

- `scripts/stage3_tone_distance.py`: TD (fração de sentenças, hard-label) +
  **permutation null** (embaralha sentenças entre gestores preservando contagem;
  TD_adj e z) + decision rule + sanity check.
- `scripts/stage4_replicate_angelo.py`: **H1** `CAR[-1,+1] ~ TD + controles` e
  **prêmio** `ret_21`/`ret_63 ~ TD`, com `fixed effects` empresa+trimestre e
  `clustered SE` por empresa, winsorize 5/95, TD padronizada (coef por SD).
  Reusa `car` e controles já calculados (independem da TD).

## 5. O que você verá quando o piloto terminar (automático)

Vou rodar Stage 3 (1.200 calls, 200 permutações) e Stage 4 (H1 + prêmio, universo
strict e core), escrever os resultados aqui e **disparar o run completo** (5.452
calls) em background. Os números do piloto virão **claramente rotulados como
amostra piloto** — não o resultado final.

## Pendências que dependem de você

- **Stage 1 (LLM classifica papéis de orador)** = o uso de GenAI no núcleo (15% da
  nota). Precisa de decisão sua: qual API/modelo usar e se há chave disponível. O
  scoring por sentença é agnóstico a papel, então trocar heurística→LLM depois é
  barato (só re-agrega). O piloto usa a heurística atual.
- Confirmar o recorte de universo primário (strict_tech vs core).
- Reverter a suspensão do PC quando terminar os runs (`powercfg /change
  standby-timeout-ac <min>`).
