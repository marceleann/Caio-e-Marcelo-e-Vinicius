# GENAI_USAGE.md — Registro do uso de IA Generativa no projeto

> Exigência do Desafio Quant AI 2026 (critério 7, peso 15%): uso de GenAI em
> pelo menos uma etapa, **descrito de forma objetiva** — em que etapa foi usada
> e qual foi a contribuição prática. Este arquivo é esse registro, mantido
> continuamente (não reconstruído de memória no fim).

## Ferramenta

**Claude Code (Anthropic)** — agente de codificação baseado em LLM, operado em
sessões de pair-programming com o time. O time dirige (decide tese, universo,
critérios, aceita/rejeita propostas); a IA executa e propõe (código, revisões,
documentação, análise de documentos).

## Distinção importante (para a banca)

- **GenAI no NÚCLEO do pipeline (uso principal):** um LLM (Claude Haiku 4.5, via
  CLI `claude -p`) classifica o **papel de cada orador** (gestor × analista ×
  operador) — a pré-condição sobre a qual toda a Tone Distance se apoia. Roda como
  **script reproduzível** (`scripts/stage1_llm_classify.py`), não como conversa
  avulsa. Detalhe na seção abaixo. É o uso com **impacto efetivo** que satisfaz o
  critério 7 sem ser declaratório.
- **GenAI no PROCESSO (secundário):** Claude Code como apoio de engenharia,
  documentação e organização (log detalhado na tabela adiante). Sozinho seria
  "declaratório"; por isso o uso do núcleo acima é o que apresentamos como central.
- **NLP/ML no NÚCLEO (não é GenAI):** o FinBERT (`yiyanghkust/finbert-tone`), que
  pontua o tom das falas, é um classificador transformer — **não é IA generativa**
  e não é apresentado como tal. ML no modelo é permitido e opcional pelo edital.

## GenAI no NÚCLEO — classificação de papel do orador (Stage 1)

**Problema:** a tese (Tone Distance = desacordo de tom **entre gestores** da mesma
call) exige saber, para cada fala, se quem falou é gestor da empresa, analista
sell-side ou operador. O dataset traz só o NOME. Sem isso, analistas contaminam o
conjunto de gestores e a métrica vira lixo.

**Solução (robô reproduzível):** `scripts/stage1_llm_classify.py` varre os
**12.678 headers únicos** e, em lotes de 60, chama **Claude Haiku 4.5** via
`claude -p --model haiku` (headless, na assinatura — sem custo de API extra).
O prompt (constante `SYS` no script) define os 4 papéis e passa como DICA os
sinais determinísticos por header: nº de empresas em que o nome aparece (analista
cobre muitas; executivo 1, ou 2-3 se trocou de emprego), se dá prepared remarks
(só a empresa dá) e se há cargo/firma no header. Saída forçada em JSON, **cacheada
e resumível** (`data/interim/llm_roles_cache.jsonl` → `speaker_roles_llm.parquet`);
re-execução produz o mesmo dicionário.

**Impacto efetivo (casos que regra/heurística erram e o LLM acerta):**
- executivos que trocaram de empresa (aparecem sob vários tickers): Nikesh Arora
  (CEO PANW), George Davis/David Zinsner (CFOs Intel), Bruce Kiddoo (CFO Maxim);
- analistas boutique (sob 1 empresa, pareceriam gestor): Ivan Feinseth (Tigress),
  Julian Mitchell (Barclays);
- colisão de sobrenome com banco: Peter *Oppenheimer* (CFO Apple), Pamela *Craig*
  (CFO Accenture) → gestores;
- CEOs que estreiam no Q&A (Tim Cook, Steve Sanghi, Reed Hastings), que a
  heurística posicional rotulava como analistas.

**Validação:** (1) concordância com um classificador determinístico independente
(`scripts/stage1_speaker_roles.py`, baseado em nº de empresas + firma/cargo +
remarks) — nos top 120 headers por frequência (151.312 falas): **99,2%**; (2)
**matriz de confusão** LLM × rótulo-referência em 292 headers (amostra
estratificada por papel, seed 42; referência rotulada de forma independente lendo
uma fala de exemplo de cada orador — `scripts/stage1_confusion_matrix.py` +
`stage1_gold_rules.py`):

| classe (referência) | precisão | recall |
|---|---|---|
| manager | 0,87 | 0,89 |
| analyst | 0,91 | 0,87 |
| operator | 1,00 | 1,00 |

- **Acurácia gestor↔analista (o que corrompe a TD se errar): 89,3% por header ·
  91,7% ponderada por falas.** Acurácia global 88,0%.
- A amostra estratificada super-representa a cauda de baixa frequência (nomes de
  1-2 falas); nos headers de alta frequência (que dominam o sinal) a concordância
  é ~99%.

**Limitação declarada (honestidade — edital critério 4.5/4.7):** o robô classifica
lendo apenas o header + dicas (nº de empresas, se dá remarks), **não a fala**. Por
isso erra em (a) analistas de poucos tickers (Nigel Coe/Wolfe, Craig Maurer) que
parecem executivos pela contagem de empresas, e (b) alguns executivos sem sinal
forte (Mark Hurd). A dica "dá remarks" herda ruído do rótulo de seção. A
contaminação residual do conjunto de gestores é limitada (precisão 0,87) e é
atenuada a jusante pelo permutation null. Uma v2 dando uma fala de exemplo ao LLM
elevaria a acurácia, ao custo de re-processar os 12.678 headers.

**Alternativas equivalentes (trocáveis sem mudar o método):** free tier
(Groq/Gemini) ou LLM local (Ollama) rodando o MESMO prompt. Haiku é modelo pequeno
— por isso a validação por concordância + matriz de confusão é parte do método.

## Registro por etapa (etapa × contribuição prática × papel do time)

| Etapa | Contribuição da GenAI | Papel do time (decisão humana) |
|---|---|---|
| Formulação da tese | Estruturação da hipótese em formato testável (fenômeno → justificativa → teste); levantamento de armadilhas conhecidas de sinais NLP (look-ahead, survivorship, p-hacking) | Escolha da tese e fixação do escopo |
| Pivô para Angelo (2025) | Leitura e análise do paper de Angelo et al. (2025, *Financial Review*), "Tone Distance"; tradução do sinal (distância de tom **entre gestores** dentro da mesma call, não analistas×gestão) para uma feature computável com FinBERT; contraste com a tese anterior de Brockman (2015) e mapeamento do que reaproveitar (dados/scoring) versus reconstruir (feature + provas); registro dos ADR-021/022 como superados | Decisão do pivô (tese mais nova, inovadora e alinhada à orientadora); fixação a priori da direção (long distância alta) e da modernização (FinBERT no lugar de Loughran-McDonald) |
| Definição do universo | Proposta do método híbrido (rede programática via setor yfinance ∪ lista curada) e da regra de subtração dos blocos de sensibilidade; identificação de nomes esquecidos (FICO, Teradata, Novell, Siebel...) | Decisão das 3 escolhas estruturais: método, fronteira do núcleo (IT + internet/mídia interativa) e os 4 blocos de sensibilidade |
| Arquitetura do repositório | Geração do esqueleto completo: `config.yaml` comentado, Makefile, requirements pinados, separação biblioteca×scripts | Aprovação fase a fase; exigência de padrões (docstrings PT-BR, zero números mágicos) |
| Implementação (Fase 1) | Escrita dos módulos `config/transcripts/prices/universe/diagnostics` com type hints e docstrings; testes unitários com casos verificáveis à mão | Revisão e validação; conferência dos critérios metodológicos |
| Qualidade de código | Detecção e correção de bug real via teste (ticker `ON` interpretado como booleano pelo YAML — ADR-014); lint/format contínuos | Confirmação da correção; decisão de blindar o config com falha explícita |
| Documentação metodológica | Redação de `DECISIONS.md` (ADRs), `data_schemas.md`, `PROGRESS.md` | Conteúdo das decisões é do time; a IA registra e organiza |
| Análise do regulamento | Extração e leitura integral dos 3 PDFs oficiais (30 págs., escaneados — OCR via renderização); síntese em `REGRAS_DESAFIO.md`; análise de conformidade e identificação de 4 gaps (nome do robô, este arquivo, pré-relatório, análise crítica) | Fornecimento dos documentos; priorização dos gaps |
| Revisão de código multi-agente (Fase 2, 3 rodadas) | Rodada 1: 4 revisores paralelos (corretude, spec, robustez, testes) + céticos; achou e reproduziu um **bug crítico** (boilerplate do Operator disparava o Q&A e toda a gestão virava analista, silenciosamente) + ~6 menores. Rodada 2 (verificação pós-correção): céticos REPRODUZIRAM 2 buracos remanescentes → design final com 3 classes de marcadores (fortes/fracos/anti-aviso); endureceram config, `_hard_split` e o teste fake de reordenação. Rodada 3: céticos **derrubaram 2 descartes humanos MEDINDO a base real** (grafia divergente do mesmo executivo = artefato sistemático em ~130 calls → chave de pareamento sobrenome\|inicial; contrato chunk≤512 pinado com tokenizer real) | Decisão de aplicar/refutar cada achado; 2 descartes humanos sobreviveram aos céticos, 2 foram derrubados com evidência empírica e corrigidos; suíte final: **54 testes** (incl. sentinela com FinBERT real) |

| Auditoria da Fase 3 (features + PIT) | Workflow com 4 lentes (caça a vazamento, matemática, spec, testes) + céticos; achados com reprodução: (1) **crítico** — regressão no limite direito da janela `_idio` (própria call e calls simultâneas — 38% da base compartilha timestamp — entravam na "média dos pares"); (2) **major 3/3** — regra same-open ignorava a DURAÇÃO da call (674 calls reais negociáveis com a call em andamento) → buffer de 75 min; (3) **major 3/3** — `merge_asof` com match exato vazaria ~6,5h de dados EOD → `allow_exact_matches=False`; + bug real no script 04 (`has_time` ausente) e ~6 endurecimentos de teste | Verificação da regressão no ambiente local antes de aceitar; decisão do buffer (75 min) e da semântica estrita; suíte final 89 testes nos 2 interpretadores |

| Auditoria da Fase 4 (backtest) | Workflow 4 lentes + céticos; VALIDOU numericamente o núcleo (DSR, somas-prefixo do event study, purga/embargo, PBO) e mediu problemas de casca com reprodução: exposição líquida residual do long-short (~52% dos cohorts unilaterais), viés de empate no gatilho percentil, feriados fabricados por `freq='B'`, 4+ parâmetros no-op no config, 8 famílias de mutantes de teste sobreviventes — todos corrigidos com testes (suíte: 138) | Decisão sobre cada correção (ex.: reportar+hedgear exposição em vez de forçar neutralidade); ajuste de expectativas de testes ao comportamento novo |

| Auditoria da Fase 5 (decay + report) | Workflow 3 lentes + céticos; achado **crítico de TESTE**: a suíte não fixava o bootstrap de blocos móveis — trocá-lo por i.i.d. sobrevivia e inverteria a conclusão do decay em dados autocorrelacionados (demonstrado com AR(1)); + bug de retorno-total no `relative_capital_table` (descartava o dia 0) e bootstrap não-circular com viés de borda. Corrigidos com teste AR(1) que pina a materialidade do bloco | Decisão de aplicar; distinção "achado de teste" vs "achado de código"; suíte final: 155 testes |
| Teste de integração pré-rodada | Geração de mundo sintético (alpha plantado só pré-2015) e execução dos scripts 05–09 de ponta a ponta antes da rodada real; validou que a máquina detecta o decay plantado e pegou 1 bug (benchmark ausente virava linha de zeros silenciosa) | Decisão de de-riscar antes do run caro; interpretação dos resultados sintéticos |

| Análise controlada do sinal (pós-pivô, workflow multi-agente) | Construção da feature de distância de tom conforme Angelo e da regressão com efeitos fixos de empresa+trimestre e erros agrupados por empresa; medição do contraste sinal **cru** (não prevê nada, t < 0,4) versus sinal **limpo** (resíduo dos confundidores: 1 mês +0,41%/dp, t=2,39; 3 meses +1,01%/dp, t=2,50; event study +1,5%, p≈0,000) e do comportamento no holdout (limpo positivo, cru vira negativo); grade de 60 combinações com o sinal limpo campeã | Fixação dos confundidores e do horizonte a priori (pelo paper); leitura crítica dos coeficientes e aceitação do sinal limpo como produto |
| Verificação independente da regressão | Um agente separado reproduziu a regressão principal do zero (sem ver a implementação original) e bateu exatamente os coeficientes; conferência de ausência de vazamento de futuro nos controles (estritamente-passados) | Solicitação da reprodução independente como guarda antifraude; validação do batimento |

## Etapas futuras previstas (atualizar conforme acontecer)

- Pré-relatório (31/07) e relatório final (17/08): apoio de estruturação e
  revisão de texto a partir de DECISIONS/PROGRESS; conteúdo analítico e
  conclusões são do time.
- Identidade do robô: brainstorming de nomes coerentes com a tese de distância
  de tom (decisão final é do time).

## Evidências

O histórico das sessões de Claude Code (transcrições locais) e o próprio
repositório (commits, docstrings, ADRs) documentam materialmente o fluxo
IA-propõe → time-decide. Disponível para a banca se solicitado.
