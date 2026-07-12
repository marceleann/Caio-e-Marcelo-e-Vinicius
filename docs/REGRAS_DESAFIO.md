# REGRAS_DESAFIO.md — Síntese dos documentos oficiais do Desafio Quant AI 2026

> Fonte: três PDFs oficiais (Edital Oficial; Manual de Avaliação Oficial; Guia de
> Primeiros Passos/FAQ), lidos integralmente em 2026-07-04. Este arquivo é a
> referência interna do time — TODA decisão de projeto deve respeitá-lo.
> Canal oficial: https://meu.itau/CanalDesafioQuantAI2026 (acompanhar SEMPRE;
> instruções detalhadas do Relatório Final ainda serão divulgadas).

## 1. O que o desafio pede (Edital)

Desenvolver uma **estratégia quantitativa de investimento** com pesquisa,
modelagem e **backtest implementado pela própria equipe**, usando **IA
Generativa como apoio em pelo menos uma etapa** (obrigatório). A equipe atua
como um time quant de uma gestora: propor tese, modelar, testar, analisar
criticamente e apresentar.

- Equipes de **2 a 3 estudantes de graduação** (matrícula ativa na inscrição).
- Fluxo conceitual exigido: **dados entram → modelo processa → decisão sai →
  decisão é testada historicamente**.
- A saída deve ser **objetiva e operacionalizável** (sinal, ranking, pesos,
  rebalanceamento...) — testável em backtest.
- Classe de ativos, período de backtest e fonte de dados: **livres** (com
  justificativa razoável). Dados pagos ou gratuitos, estruturados ou não.
- Custos/slippage: **não obrigatórios** — mas coerência entre sofisticação e
  realismo é valorizada.
- Benchmark: **recomendado**, escolha livre e coerente com o universo.
- Out-of-sample/walk-forward: **não exigidos** formalmente (liberdade
  metodológica; se usar, conta como qualidade técnica).
- Excel/Python/bibliotecas prontas: permitidos. **Proibido** delegar o processo
  inteiro a uma plataforma externa que "entregue pronto" o resultado; a lógica e
  os resultados precisam ser da equipe, que deve saber explicá-los.
- Código-fonte: não exigido nas etapas iniciais; **finalistas podem ser
  solicitados** a compartilhar materiais para validação metodológica.

## 2. Cronograma oficial (sujeito a ajustes pelo canal)

| Marco | Data |
|---|---|
| Inscrições (encerramento) | 31/05/2026 |
| **Pré-relatório (etapa intermediária)** | **31/07/2026** |
| **Entrega final** | **17/08/2026** |
| Quartas de final (online, 20–30 equipes, apresentação gravada) | 31/08/2026 |
| Semifinal (online, 8–10 equipes, ao vivo + Q&A com banca técnica) | 09/09/2026 |
| Final (presencial, 3 equipes, Faria Lima/SP, gestores Itaú Asset e Itaú BBA) | 26/09/2026 |

## 3. Critérios de avaliação e pesos (Manual de Avaliação)

| # | Critério | Peso |
|---|---|---|
| 1 | Apresentação do robô (nome e identidade) | **5%** |
| 2 | Conceito da estratégia (criatividade e inovação) | **20%** |
| 3 | Modelagem (estrutura e lógica do modelo) | **20%** |
| 4 | Backtest (rigor e mitigação de vieses) | **15%** |
| 5 | Análise dos resultados (clareza e profundidade) | **15%** |
| 6 | Conclusão e próximos passos | **10%** |
| 7 | Uso de IA generativa no processo | **15%** |

Princípios transversais: prioridade à **qualidade da construção** (não só ao
resultado); **coerência entre hipótese, modelagem e teste**; **neutralidade
quanto à complexidade** (simples bem-feito > complexo mal explicado);
**capacidade de explicação** (dados, premissas, lógica, backtest, limitações);
avaliação por especialistas, possivelmente **anônima** nas etapas aplicáveis.

### Detalhes por critério (pontos positivos e negativos)

1. **Robô (5%)** — nome coerente com a estratégia, identidade conceitual clara.
   Negativo: nome genérico; excesso visual sem função; baixa clareza.
2. **Conceito (20%)** — hipótese clara; coerência quantitativa/econômica;
   originalidade; consistência como estratégia de investimento. Explicitar: o
   FENÔMENO capturado, a JUSTIFICATIVA da existência e COMO será testado.
   Negativo: sem hipótese; propostas desconexas; **complexidade sem fundamentação**.
3. **Modelagem (20%)** — inputs claros, processamento descrito, saída objetiva,
   processo sistemático e replicável. Negativo: lógica de decisão obscura;
   processos não replicáveis; **ferramentas externas sem compreensão**.
4. **Backtest (15%)** — implementação própria; coerência modelo↔simulação;
   adequação do período; consistência metodológica; preocupação com vieses
   (**escolhas oportunistas de período; simulações incoerentes com a lógica;
   ausência de justificativas**). Valorizados: replicável, transparente, coerente.
5. **Análise (15%)** — retorno E risco; limitações do modelo; comportamento ao
   longo do tempo; cenários favoráveis/desfavoráveis. Negativo: apresentação
   **exclusivamente descritiva de métricas**; sem análise crítica; omissão de
   fragilidades.
6. **Conclusão (10%)** — coerência resultados↔conclusões; recomendações de
   evolução realistas; reconhecer limites; **evitar conclusões desproporcionais
   às evidências**.
7. **GenAI (15%)** — uso prático, relevante e claramente explicado, em qualquer
   etapa (ideias, código, organização, interpretação, identidade do robô...).
   O modelo quantitativo NÃO precisa conter IA. Negativo: uso superficial/
   declaratório; sem impacto prático; não compreender o papel da IA na solução.

## 4. Implicações diretas para o nosso projeto

- **Prazo real**: o pré-relatório (31/07) chega antes do fim do nosso plano de
  fases — reservar tempo para redigi-lo a partir de DECISIONS/PROGRESS.
- **Nome/identidade do robô (5%)**: criar marca própria para a estratégia
  (candidata natural: algo em torno de "Tone Distance", a tese atual de Angelo —
  distância de tom entre gestores). PENDENTE.
- **GenAI (15%)**: nosso uso é real e profundo (projeto inteiro pareado com
  Claude Code: arquitetura, código, revisão, documentação). DOCUMENTAR de forma
  objetiva em `docs/GENAI_USAGE.md` (etapa × contribuição prática). Atenção à
  nuance: **FinBERT é NLP/ML no núcleo do modelo (permitido, opcional), não é
  "GenAI"** — a exigência de GenAI é cumprida pelo uso no PROCESSO.
- **Explicabilidade**: a banca pode arguir qualquer linha — o padrão de
  docstrings PT-BR + ADRs existe exatamente para isso; manter.
- **Vieses de backtest**: nossos guarda-rails (T+1, timestamps completos,
  survivorship, placebo intra-data, DSR com n_trials real, amostra completa
  2005–2025 sem cherry-picking de período) atacam explicitamente o que o
  critério 4 pune. Manter e EVIDENCIAR no relatório.
- **Análise (15%) e Conclusão (10%)**: o relatório final deve ir além das
  métricas — regimes, cenários desfavoráveis, limitações declaradas, evolução
  realista. A análise de decay pré/pós-2015 é ativo forte aqui (resultado
  honesto em qualquer direção).
- **Não-obrigatórios que fazemos por qualidade** (custos, holdout, validação
  purgada): manter, apresentando como rigor voluntário — conta em "consistência
  das escolhas metodológicas".
