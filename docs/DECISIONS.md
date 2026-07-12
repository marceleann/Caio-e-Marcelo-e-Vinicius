# DECISIONS.md — Registro de Decisões de Arquitetura (ADRs)

Formato ADR curto. Cada decisão metodológica não-óbvia é registrada com
**Contexto**, **Alternativas** e **Decisão/Justificativa**. Este arquivo é a
fonte primária do capítulo de metodologia do relatório final e a base da
defesa oral perante a banca.

> Status legenda: `Aceita` = vale no código; `Proposta` = a decidir em fase futura.

---

## ADR-000 — Nomenclatura da métrica

- **Status:** SUPERADA pelo ADR-023 (pivô Angelo). Quando esta decisão foi
  tomada, a nossa métrica-tese era a divergência de tom gestão×analistas, que
  batizamos de "Tone Divergence" para diferenciá-la do "Tone Distance" do
  Angelo. Depois o projeto adotou justamente a tese do Angelo (distância de tom
  ENTRE OS GESTORES, medida com FinBERT) como tese principal, e a antiga
  "Tone Divergence" gestão×analistas virou feature de comparação/robustez. O
  pacote segue chamado `tonediv` por legado. Registro mantido por honestidade —
  a virada faz parte da história do trabalho.
- **Contexto:** Precisamos de um nome próprio, defensável e sem colisão com a
  literatura existente.
- **Alternativas:** "Tone Distance" já é usado por Angelo (2025, *Financial
  Review*) com OUTRA definição — usá-lo induziria confusão e enfraqueceria a
  originalidade.
- **Decisão:** Nossa métrica gestão×analistas chama-se **"Tone Divergence"**. A
  variação vs. a call anterior da mesma empresa é **"ΔTone" / "JS-distance"**.
- **Diferenciação precisa do "Tone Distance" (Angelo et al. 2025, Financial
  Review, lido em 2026-07-07):** o "Tone Distance" deles é a **variância de tom
  ENTRE MANAGERS** dentro de uma call (desacordo gerencial — manager A vs.
  manager B), associada negativamente ao retorno do evento e positivamente a
  risco/volatilidade. É um conceito DIFERENTE do nosso: nós medimos a
  divergência **GESTÃO × ANALISTAS** (Brockman et al. 2015). Por isso NÃO
  usamos o nome "Tone Distance" — e por isso nossa `tone_dispersion` (feature 4,
  desvio dos net_tones das falas) é a nossa aproximação MAIS PRÓXIMA do conceito
  do Angelo (embora a nossa seja sobre TODAS as falas, não só as da gestão).
  Diferenciação registrada; nome próprio evita colisão.

---

## ADR-001 — Universo definido por LISTA EXPLÍCITA, não por setor GICS

- **Status:** Aceita
- **Contexto:** O universo é "empresas de tecnologia dos EUA". A classificação
  GICS mudou duas vezes no período amostral: 2018 (criação de *Communication
  Services* — Google/Meta/Netflix saíram do IT/Consumer) e 2023 (pagamentos:
  V/MA/etc. de IT para *Financials*; ADP/PAYX para *Industrials*). Logo
  "setor = IT" NÃO é um critério estável em 2005–2025.
- **Alternativas:** (a) filtrar por GICS-IT vigente hoje — anacrônico e instável;
  (b) filtrar por GICS point-in-time — exigiria uma fonte histórica de GICS não
  open-source.
- **Decisão (revisada em conjunto com o time):** universo construído por
  **método híbrido rede-programática ∪ lista curada**, com núcleo em
  IT + internet/mídia interativa e fronteira reportada por sensibilidade. O
  filtro é aplicado CEDO, em `universe.py`, ANTES do scoring FinBERT (pontuar
  33k calls quando precisamos de ~5–8k desperdiçaria a etapa mais cara).

  **Restrição binding:** o dataset é `kurry/sp500_earnings_transcripts` (só
  S&P 500). Não escolhemos "tech no mundo", e sim **quais nomes do S&P 500
  (atuais e históricos) contam como tech**. Universo FINAL = candidatos ∩
  transcrições no dataset ∩ preços no yfinance; ausências vão ao `quality_report.md`.

  **Construção (3 passos, em `universe.py`):**
  1. **Rede programática** — `sector`/`industry` atuais de cada ticker via
     yfinance; inclui automaticamente `sector == "Technology"` e as indústrias de
     Communication Services que são tech de fato (Internet Content & Information,
     Electronic Gaming & Multimedia) + NFLX. Exclui telecom/mídia tradicional.
  2. **União** com a lista curada (`core_it`, `core_internet_media`,
     `delisted_check`) — a lista cobre as **deslistadas**, que a rede atual não
     alcança porque já não existem no yfinance (survivorship).
  3. **Subtração** — todo ticker de `sensitivity_blocks` é REMOVIDO do núcleo
     automático. Sem isso, a Yahoo classifica UBER/ENPH/FSLR/JBL/FLEX como
     "Technology" e eles entrariam no núcleo, deixando de ser removíveis e
     **quebrando o teste de sensibilidade**.

  **Núcleo escolhido (sempre dentro):** IT clássico (`core_it`, agora com FICO,
  TDC, COHR, PAYC, JKHY, que faltavam) + internet/mídia **interativa**
  (`core_internet_media`: Google, Meta, Netflix, games). Telecom e mídia
  tradicional ficam de fora do núcleo.

  **Quatro blocos de sensibilidade (rodam com/sem cada um):**
  `payments` (V, MA, PYPL, FIS, FI, GPN), `non_gics_tech` (AMZN, TSLA, UBER,
  ABNB, BKNG, EXPE, EBAY), `solar_tech` (ENPH, FSLR, SEDG) e `ems` (JBL, FLEX).

  - **Caso Amazon (declarado):** a AMZN NUNCA foi GICS-IT (sempre *Consumer
    Discretionary*). Incluí-la é decisão NOSSA de definição de "tech", assumida
    nesses termos; por isso vive em `non_gics_tech` (com/sem), não no núcleo.
  - **Redistribuições:** EBAY (marketplace) foi para `non_gics_tech`; ADP, PAYX e
    BR (folha de pagamento / back-office) saíram do universo — claim de "tech"
    fraco e sem bloco de sensibilidade próprio nesta rodada. Reversível se o time
    quiser um bloco `payroll` depois.
  - **Revisão pós-diagnóstico real (2026-07-04):** a rede programática pescou
    ADP/PAYX/BR de volta (a Yahoo os classifica como Technology), reabrindo a
    questão. Resolução coerente com o framework: criado o bloco **`payroll`**
    (ADP, PAYX, BR) previsto acima — eles saem do núcleo automático pela regra
    de subtração e viram pergunta empírica como os demais. Pela mesma
    consistência, SANM (EMS) foi movido para o bloco `ems` e CPAY (pagamentos
    corporativos) para `payments`. Total: **5 blocos de sensibilidade**. Os
    demais 12 achados da rede (CIEN, GRMN, IAC, IPGP, LDOS, PD, ROP, TDY, UIS,
    VIAV, FTV, VNT) permanecem no núcleo via `net_discovery` — o método
    declarado decide, não o gosto.

---

## ADR-002 — Timestamp da decisão e regra T+1

- **Status:** Aceita
- **Contexto:** O dataset traz data+hora da call. Uma estratégia operável precisa
  saber em que pregão a informação estaria REALMENTE disponível para negociar.
- **Alternativas:** entrar no close do dia da call (vaza: a call pode ter sido
  após o fechamento) ou ignorar horário (perde-se granularidade e vaza no limite).
- **Decisão:** Fuso assumido **US/Eastern**. Regra:
  - call **após 16:00 ET** → decisão na **abertura do próximo pregão** (T+1 open);
  - call **antes de 09:30 ET** → abertura do **mesmo dia**;
  - call **durante o pregão** OU **horário ausente/suspeito** (`00:00:00`, nulo) →
    **fallback conservador T+1 open**.
  - A distribuição de horários é validada antes de confiar neles (script 00 plota
    o histograma). Ver ADR-008.
- **Revisão (auditoria da Fase 3, achado 3/3):** "antes da abertura" passou a
  significar "**termina** antes da abertura". As features usam a transcrição
  COMPLETA e o Q&A (fonte da tese) acontece no FIM da call — uma call iniciada
  às 09:00 ainda está acontecendo às 09:30. Regra final: same-day open só se
  `início ≤ open − assumed_call_duration` (75 min conservadores no config);
  na base real, 674 calls (12,4%) iniciadas 08:30–09:29 migraram para T+1.
  A guarda também endureceu: compara o open da decisão com o **fim estimado**
  da call (início + duração), não com o início. `call_datetime` nulo ganhou
  regra própria (`no_timestamp` → sem decisão, contabilizado). Os parâmetros
  no-op `pit.fallback`/`pit.merge_direction` foram REMOVIDOS do config (T+1 e
  backward são fixos por design — parâmetro ignorável seria no-op silencioso).

---

## ADR-003 — Junção texto→preço e guarda anti-look-ahead forte

- **Status:** Aceita
- **Contexto:** Alinhar cada call ao painel de preços sem vazar futuro é o ponto
  mais frágil de sinais baseados em eventos.
- **Decisão:**
  - Junção via `pandas.merge_asof(direction='backward', by='ticker')` —
    **NUNCA** `forward` (que casaria com preço posterior à decisão).
  - **Revisão (auditoria da Fase 3, achado 3/3):** também
    `allow_exact_matches=False`. A decisão acontece no OPEN (09:30), mas
    painéis de mercado (ADV, volume, close) são carimbados no dia e medidos no
    FECHAMENTO — aceitar o match exato do mesmo dia entregaria ~6,5h de
    futuro ao filtro de liquidez da Fase 4. Só a última observação
    ESTRITAMENTE anterior existia no open. O wrapper `merge_asof_backward`
    fixa as duas garantias de forma não-configurável e preserva linhas com
    data nula (evento sem pregão) em vez de quebrar.
  - Guarda `assert_no_lookahead` compara **TIMESTAMPS COMPLETOS**, não datas
    normalizadas. Para uma call após o fechamento, uma decisão no MESMO dia deve
    DISPARAR erro. Há teste que prova que a guarda pega esse caso
    (`tests/test_pit_leakage.py`).
  - **Armadilha evitada (protótipo anterior):** guardas que normalizam para data
    deixam passar "call às 19h → decisão no mesmo dia". Por isso comparamos o
    instante completo.

---

## ADR-004 — Ajuste idiossincrático por pares em janela rolante point-in-time

- **Status:** Aceita
- **Contexto:** Parte do "tom" de uma call é maré do setor, não sinal específico
  da empresa. Removê-lo exige uma referência de pares que não vaze futuro.
- **Alternativas:** subtrair a média do trimestre inteiro do setor — INVÁLIDO,
  pois inclui calls que ocorreram DEPOIS da call em questão (look-ahead).
- **Decisão:** As features `_idio` subtraem a média dos PARES calculada numa
  **janela rolante de 90 dias corridos até o instante da call**, usando SOMENTE
  calls que já aconteceram. Exige `peer_min_calls` pares na janela; senão NaN.
  90 dias ≈ um trimestre de calls, equilibrando amostra suficiente e relevância
  temporal.

---

## ADR-005 — Backtest calendar-time com posições sobrepostas (não quintis por data)

- **Status:** Aceita
- **Contexto:** Eventos (earnings calls) são esparsos no tempo. O entregável é
  uma estratégia OPERÁVEL, não um paper.
- **Alternativas:** carteiras por quintil exigindo 5+ empresas no MESMO dia de
  evento — produz carteiras vazias na maior parte dos dias (defeito conhecido do
  protótipo anterior).
- **Decisão:** Portfólio **calendar-time estilo Jegadeesh-Titman**: holding de H
  pregões com tranches sobrepostas; a cada dia a carteira é a média das tranches
  abertas nos últimos H dias. Duas variantes obrigatórias: (i) long-short
  dollar-neutral e (ii) long-only vs. benchmark. Saída mecânica após H pregões,
  sem stop discricionário. 100% especificada por regras (ver `strategy_spec.md`).

---

## ADR-006 — Heurística de papéis analista × gestão

- **Status:** Aceita
- **Contexto:** O campo `speaker` traz o NOME do orador, não o papel. A tese
  depende de separar falas de analistas das de gestão no Q&A.
- **Decisão:** Regras em cascata:
  1. Detectar início do Q&A por marcadores na fala do `Operator`.
  2. Quem fala ANTES do Q&A (exceto `Operator`) = **gestão**.
  3. No Q&A: `Operator` é neutro (descartado); quem já era gestão continua
     gestão; qualquer orador NOVO que estreia no Q&A = **analista**.
  4. Refinamento: regex extrai o nome anunciado pelo Operator ("...from {nome}
     with {corretora}") como CONFIRMAÇÃO de analista.
  - **Validação obrigatória:** script sorteia 20 calls (seed fixa), exporta CSV
    (`speaker, papel_inferido, primeira_fala_truncada`) para conferência manual e
    calcula cobertura (% de falas classificadas, % de calls com Q&A detectado).
    O relatório vive em `docs/roles_report.md` (gerado pelo script 02) e é
    referenciado pelo `quality_report.md`.
  - **Revisão 1 (2026-07-04, achado CRÍTICO da revisão multi-agente):** a fala
    de abertura padrão do Operator ("...Later, we will conduct a
    question-and-answer session...") contém o marcador como AVISO; sem
    restrição, `qa_start=0`, ninguém "fala antes do Q&A" e TODA a gestão era
    classificada como analista — silenciosamente (a cobertura continuava 100%).
  - **Revisão 2 (mesma data; céticos REPRODUZIRAM dois buracos na 1ª correção):**
    (a) bastava um host de RI — ou uma linha com speaker nulo — falar antes do
    boilerplate para o bug ressurgir; (b) transcrições só-Q&A viravam falso
    negativo (analistas rotulados gestão). Design final: **marcadores em 3
    classes** — FORTES ("first question comes from"...) abrem o Q&A em
    qualquer posição; FRACOS ("question-and-answer session") exigem fala
    NOMEADA anterior de não-Operator; ANTI-PADRÕES de aviso ("later, we will
    conduct", "instructions will follow") vetam a fala inteira. Guardas
    adicionais: listas de marcadores validadas no config (rejeita string
    escalar — que viraria alternância de caracteres casando com tudo — e itens
    não-string); entrada vazia/índice duplicado/`call_id` nulo falham alto com
    mensagem explícita (nulo fundiria calls distintas num pseudo-call).
    Limitação residual DOCUMENTADA: em calls só-Q&A, gestor que estreia no
    Q&A é rotulado analista (regra 3); a validação manual dimensiona o caso.
  - **Revisão 3 (mesma data; céticos derrubaram um descarte NOSSO medindo a
    base real):** grafia divergente do MESMO executivo entre seções não é
    rara — é artefato sistemático (~130 calls, ~950 falas de gestão viravam
    analista): nomes COLADOS no Q&A ("MarcBenioff"), sufixos de cargo,
    iniciais do meio e apelidos (Mike/Michael). Mitigação implementada:
    limpeza de nome ampliada (prefixo "Q -", sufixo " - cargo", split de
    camel-case, NFKD) + **chave de pareamento intra-call `sobrenome|inicial`**
    para continuidade de gestão e confirmação de analistas. Trade-off aceito e
    documentado: colisão (pessoas distintas com mesmo sobrenome+inicial na
    MESMA call) é ordens de grandeza mais rara que o dano medido da chave
    estrita. Segundo descarte derrubado: contrato "chunk ≤ 512" agora é pinado
    com o tokenizer REAL (teste com tokenizer em cache, sem baixar o modelo).

---

## ADR-007 — Mapeamento de labels do FinBERT lido em RUNTIME

- **Status:** Aceita
- **Contexto:** A ordem das classes [neg, neu, pos] de um modelo HuggingFace NÃO
  é garantida; hardcodá-la pode inverter positivo↔negativo silenciosamente.
- **Decisão:** Ler `model.config.id2label` em RUNTIME e mapear por NOME
  ("positive"/"negative"/"neutral"), nunca por índice fixo. Teste unitário com
  frases-sentinela garante `p_pos > p_neg` em texto otimista e o inverso em
  pessimista (`tests/test_scorer_labels.py`).

---

## ADR-008 — Tratamento de horário ausente/suspeito

- **Status:** Aceita
- **Contexto:** Parte das calls do dataset pode ter horário `00:00:00`, nulo ou
  implausível.
- **Decisão:** Horários em `suspicious_times` (config) ou ausentes → fallback
  conservador **T+1 open** (ADR-002). A distribuição real de horários é
  diagnosticada no script 00 (histograma) ANTES de confiarmos no campo.

---

## ADR-009 — Placebo permutado DENTRO da data (cross-section)

- **Status:** Aceita
- **Contexto:** Precisamos de uma distribuição nula honesta para o sinal.
- **Alternativas:** embaralhar o painel inteiro — infla falsos positivos, pois
  destrói a estrutura temporal e cria correlações espúrias (armadilha do
  protótipo anterior).
- **Decisão:** Permutar o sinal **DENTRO de cada data/janela cross-section**,
  preservando a estrutura temporal. O event study usa adicionalmente um placebo
  de **datas falsas** (mesma empresa, data deslocada aleatoriamente) para a
  distribuição nula do CAR.

---

## ADR-010 — Deflated Sharpe Ratio (Bailey & López de Prado, 2014)

- **Status:** Aceita
- **Contexto:** Testar muitas combinações infla o Sharpe do melhor por acaso.
- **Decisão:** Implementar o **DSR de Bailey & López de Prado (2014)** com a
  variância dos Sharpes da **grade REAL de trials** (`n_trials` = nº total de
  células efetivamente testadas pelo script 07, não um número inventado).
  Qualquer aproximação é documentada no docstring da função. Complementado por
  **PBO via CSCV**.

---

## ADR-011 — Custos de transação em 5 bps por perna (configurável)

- **Status:** Aceita
- **Contexto:** O entregável deve ser líquido de custos e crível para o mercado
  americano.
- **Decisão:** Custo default **5 bps por perna** (mercado dos EUA é mais barato
  que os ~15 bps da B3), em `costs.bps_per_side`, plenamente configurável. Todo
  resultado de estratégia é reportado bruto E líquido.

---

## ADR-012 — Período amostral completo; 2015 é só a fronteira do decay

- **Status:** Aceita
- **Contexto:** A pergunta secundária investiga se o sinal decaiu após a
  publicação de Brockman, Li & Price (2015, FAJ).
- **Decisão:** A amostra é **2005–2025 COMPLETA**. 2015 **não** corta a amostra —
  é apenas o ponto de divisão da análise de decay (pré vs. pós-publicação), com
  teste de diferença de IC/CAR/Sharpe entre regimes. Ambos os desfechos (sinal
  sobreviveu ou decaiu) são resultados VÁLIDOS.

---

## ADR-013 — Método FinBERT + Jensen-Shannon, não dicionário Loughran-McDonald

- **Status:** Aceita
- **Contexto e atribuição correta:** a tese analistas×gestão é de **Brockman,
  Li & Price (2015), "Differences in Conference Call Tones: Managers *versus*
  Analysts" (FAJ 71/4)** — eles mediram a diferença de tom gestão×analistas
  com DICIONÁRIO (2004–2007) e mostraram que ela informa retornos. Não
  inventamos a divergência; a herdamos e a MODERNIZAMOS.
- **Decisão:** **FinBERT** (`yiyanghkust/finbert-tone`) gera distribuições
  [neg, neu, pos] por fala; a divergência é a **distância Jensen-Shannon** entre
  a distribuição agregada de analistas e a de gestão, mais a diferença de
  `net_tone` (a versão "assinada", mais próxima da medida original de Brockman).
  **Nossa contribuição sobre Brockman:** (a) modelo contextual vs. dicionário;
  (b) JS das distribuições completas vs. só net-tone; (c) amostra 2005–2025;
  (d) **teste de decay pós-publicação**. Loughran-McDonald pode entrar apenas
  como **baseline comparativo** se sobrar tempo — nunca como método principal.
- **Distinção que a banca pode cobrar:** existe OUTRO ramo da literatura que
  mede o tom da GESTÃO SOZINHA → retornos (Price, Doran, Peterson & Bliss 2012;
  boa parte do trabalho com Loughran-McDonald em prepared remarks). Esse NÃO é
  a nossa tese — a nossa é a DIFERENÇA gestão×analistas (Brockman et al.).

---

## ADR-014 — Robustez de dados: ticker `ON` citado e teste de survivorship real

- **Status:** Aceita (Fase 1)
- **Contexto:** Dois riscos de dados foram tratados na camada de ingestão:
  1. **Bareword YAML:** o ticker `ON` (ON Semiconductor) é interpretado pelo
     YAML 1.1 como o booleano `True` ao carregar — corrompendo a lista de
     universo silenciosamente (o mesmo valeria para `OFF`/`YES`/`NO`).
  2. **Separação analista×gestão precisa do papel, não do nome**, e o teste de
     survivorship precisa do ticker LITERAL, não do canônico.
- **Decisão:**
  - `ON` é citado (`"ON"`) no `config.yaml`, e `config.py` **falha alto**
    (`_ticker_tuple`) se qualquer ticker for parseado como não-string — em vez de
    corromper a lista. Um teste unitário cobre a regra de subtração que expôs o bug.
  - `prepare_calls` preserva `ticker_raw` (pré-alias) além do `ticker` canônico;
    o teste de survivorship do script 00 procura o nome literal (JAVA, SYMC,
    YHOO…) para responder honestamente "essa deslistada aparece nos anos em que
    existiu?".
- **Nota de reprodutibilidade:** o ambiente de desenvolvimento roda **Python
  3.14**, enquanto `requirements.txt` fixa versões testadas em 3.11. As libs
  pesadas (torch/transformers/datasets/yfinance) podem não ter wheels para 3.14;
  recomenda-se um venv **Python 3.11** para o pipeline completo. Os módulos e
  testes offline da Fase 1 rodam em ambas.

---

## ADR-015 — Chunking por sentenças e agregação ponderada no scoring (Fase 2)

- **Status:** Aceita
- **Contexto:** O encoder do FinBERT aceita 512 tokens; falas longas (prepared
  remarks) excedem isso. Como dividir e como recompor a distribuição por fala?
- **Decisão:**
  1. **Chunking por fronteira de sentença** (regex leve), guloso, SEM
     sobreposição; sentença isolada maior que o limite sofre hard-split por
     palavras (caso raro, documentado). Cortes arbitrários no meio de sentença
     mutilariam o contexto sintático de que o modelo depende.
  2. **Contagem de tokens com o tokenizer REAL** do FinBERT no pipeline
     (injetada como função; testes usam contagem de palavras — a lógica de
     empacotamento é idêntica e testável offline).
  3. **Agregação por média ponderada pelo nº de tokens** de cada chunk: um
     chunk de 400 tokens carrega mais texto (e mais evidência de tom) que um de
     40. Alternativa rejeitada: média simples (superpesa chunks curtos).
  4. **Filtro `min_chars_utterance` aplicado NO SCORING, não na inferência de
     papéis**: falas curtas ("Thank you.") são ruído de cortesia para o tom,
     mas ajudam a detectar a estrutura do Q&A — ficam na base, saem do sinal.
  - Ordem canônica em todo o projeto: `[p_neg, p_neu, p_pos]`; a reordenação a
    partir do layout do modelo acontece UMA vez, em `score_texts` (ADR-007).

---

## ADR-016 — Realidade da base: survivorship parcial e o incidente yfinance

- **Status:** Aceita (Fase 1/2, pós-download real)
- **Contexto:** O script 00 rodou contra a base REAL pela primeira vez em
  2026-07-04 e produziu dois fatos que condicionam o projeto.
- **Fato 1 — Survivorship PARCIAL do dataset de transcrições:** das 37
  deslistadas tech da lista de verificação, apenas **5 têm transcrições** no
  `kurry/sp500_earnings_transcripts`: ALTR (2005–2015), XLNX (2006–2021),
  MXIM (2006–2020), LSI (2006–2014) e TWTR (2018–2021). Nomes que saíram cedo
  (Sun, Yahoo, EMC, Broadcom-antiga, Motorola, Palm...) NÃO estão na base.
- **Decisão:** prosseguir com o dataset DECLARANDO a limitação (é a melhor
  fonte aberta disponível; a alternativa seria abandonar 100% open-source):
  1. O viés de sobrevivência da base é **reportado como limitação** no
     relatório final e na lâmina da estratégia — não escondido;
  2. As 5 deslistadas presentes permanecem no universo até a data de saída
     (mitigação parcial, melhor que zero);
  3. Implicação para leitura dos resultados: retornos médios do universo
     tendem a viés OTIMISTA (sobreviventes); o SINAL relativo (long-short
     cross-section) é menos afetado que o nível absoluto, e a comparação
     pré/pós-2015 (decay) usa a mesma base nos dois regimes — o viés não
     favorece um regime sobre o outro.
- **Fato 2 — Fragilidade de versão do yfinance:** o pin original (0.2.50,
  dez/2024) parou de funcionar com a API atual do Yahoo (0 preços e 0 setores,
  com erros obscuros "Expecting value" / YFTzMissingError). Atualizado para
  **1.5.1** (testado: preços e setores OK) e repinado no requirements.txt.
  Lição registrada: dependências de scraping/API não-oficial precisam de
  smoke-test no início de cada sessão de trabalho (o script 00 cumpre esse
  papel ao reportar `n_in_prices=0`).
- **Fato 3 — Survivorship também na CAMADA DE PREÇO (pior que a de
  transcrição):** com yfinance 1.5.1, 110/119 tickers têm preço, mas os 9 sem
  preço incluem TODAS as 5 deslistadas com transcrição (ALTR, LSI, MXIM, TWTR,
  XLNX) e as recém-adquiridas ANSS, ATVI e JNPR — o Yahoo remove o histórico
  de deslistados. FI era falso-negativo de símbolo (Yahoo manteve FISV):
  resolvido via `price_symbol_overrides` no config.
- **Decisão adicional:** o universo de BACKTEST é, por ora, 100% sobrevivente
  — limitação declarada em toda leitura de resultado. **Mitigação planejada
  (Fase 3):** fonte de preço alternativa open-source (Stooq, via
  pandas-datareader) como fallback para deslistados; se recuperar parte dos 9,
  o teste de survivorship melhora de "documentado" para "parcialmente
  mitigado". Registrar o resultado aqui quando implementado.

---

## ADR-017 — Convenções da Fase 3: agregação por tokens e retorno open→close

- **Status:** Aceita
- **Contexto:** Três convenções precisavam ser fixadas antes do backtest; cada
  uma tem alternativas defensáveis e foi escolhida com critério.
- **Decisões:**
  1. **Agregação por call = média ponderada por TOKENS** das distribuições das
     falas (mesma lógica da agregação de chunks, ADR-015). Alternativa
     rejeitada: média simples por fala — superpesaria "Thank you, next
     question" frente a uma resposta de 400 tokens.
  2. **Retorno do evento: entrada no OPEN do pregão de decisão (d0), saída no
     CLOSE de d_{H−1}** — a posição atravessa exatamente H sessões. Alternativa
     (sair no close de d_H) manteria H+1 sessões de exposição para "H pregões";
     a convenção escolhida é a de Jegadeesh-Titman para tranches e casa com o
     portfólio calendar-time da Fase 4. NUNCA se usa o close do dia da call.
  3. **`mgmt_analyst_divergence` exige Q&A detectado E ambos os lados
     pontuados** — senão NaN + flag `qa_detected` para filtragem explícita na
     Fase 4 (nada de papéis incertos contaminando a tese; decorrência da
     revisão da Fase 2).
  4. **Janela `_idio` com limite direito ESTRITO** (`t_j < t_i`): calls do
     mesmo instante não entram na referência de pares — inclusive a própria.
- **Nota de robustez (bug ambiental pego em teste):** a aritmética temporal da
  janela `_idio` usa `datetime64`/`timedelta64` NATIVOS. A resolução de
  timestamps muda entre versões do pandas (ns no 2.x, µs no 3.x) e a versão
  inicial em inteiros crus fazia a janela de 90 dias virar 90.000 dias num dos
  interpretadores — silenciosamente. Rodar a suíte nos DOIS ambientes é
  política do projeto exatamente por isso.

---

## ADR-018 — Fase 4: construção da estratégia e do event study

- **Status:** Aceita
- **Contexto:** As "provas" exigem dezenas de micro-decisões; as não-óbvias:
- **Decisões:**
  1. **Gatilho por percentil PIT**: o sinal do evento é rankeado contra os
     eventos dos últimos `signal_rank_window_days` (90d) ESTRITAMENTE
     anteriores — eventos do mesmo dia não se veem (mesma disciplina da janela
     `_idio`). Warm-up: sem `min_rank_history` (20) eventos, fica de fora.
     Alternativa rejeitada: quantis da amostra inteira (look-ahead óbvio).
  2. **Pesos**: `1/(H × n_cohort)` com teto por nome; excesso vira CAIXA (não
     é redistribuído — redistribuir concentraria onde o teto quis limitar).
  3. **Cohort vazio = caixa a 0%** (config `cash_return`) — conservador.
  4. **Turnover** = pesos que entram + pesos que expiram no dia; custo bps
     por perna sobre esse total (ADR-011).
  5. **Event study em painel FLAT com somas-prefixo**: OLS de janela vira
     O(1)/evento; o placebo de datas falsas (1000 iterações × eventos)
     reestima α/β de verdade em cada data deslocada (±[21, 250] pregões) em
     segundos — sem aproximação.
  6. **Tabela de grupos usa teste t simples/Welch** (a unidade é o evento);
     Newey-West fica para as SÉRIES do portfólio, onde autocorrelação de
     sobreposição existe por construção.
  7. **DSR**: Sharpe POR PERÍODO (não anualizado), `n_trials` = nº real de
     células da grade do script 07, variância dos Sharpes da grade real;
     hipóteses documentadas no docstring (armadilha nº 7).
  8. **Placebo intra-janela** permuta o sinal dentro do TRIMESTRE (a mesma
     cross-section do IC) — no nível do dia, eventos esparsos degenerariam a
     permutação (n=1 na maioria dos dias).
  9. **Harness de alpha sintético** (tests/test_harness_alpha.py): mundo
     artificial com alpha PLANTADO nos H pregões pós-decisão; a máquina
     completa (liquidez → sides → portfólio → métricas → placebo) precisa
     recuperá-lo E não achar nada em ruído puro (asserções com margens e seed
     fixa, armadilha nº 5).

- **Revisão (auditoria multi-agente da Fase 4, 2026-07-07):** o núcleo PIT e a
  matemática (DSR, somas-prefixo, purga/embargo, motor do PBO) foram
  verificados numericamente SEM defeitos; as correções foram na casca:
  1. **Exposição líquida do long-short** — cohorts esparsos são por vezes
     unilaterais (medido: ~52% dos cohorts; |net|/bruta média ~29%); o rótulo
     "dollar-neutral" virou "exposição líquida RESIDUAL", a série
     `net_exposure` é diagnóstico de primeira classe e o script 06 reporta
     também a série HEDGEADA (net_exp × mercado subtraído).
  2. **Midrank no percentil do gatilho** — empates não inclinam o livro para
     long (sinal constante → percentil 0.5 → sem posição); guard
     `top_fraction < 0.5` no config.
  3. **Calendário REAL de pregões** (não `freq='B'`): feriados não viram dias
     de retorno zero fabricados (~3,5% da série; diluíam o Sharpe).
  4. **Turnover de saída ao notional corrente** (w × crescimento da tranche).
  5. **Cohorts contados após casar com preços**; n_used/n_dropped refletem
     eventos NEGOCIADOS (nada descartado em silêncio).
  6. **ADV com janela cheia** (min_periods = janela; limitação do close
     ajustado documentada); **PBO com rank/(N+1)** (convenção Bailey et al.);
     **placebo do script 06 com N permutações**; **scripts 05/06 restritos ao
     bloco DEV** (holdout intocado); **grade do 07 dirigida pelo config**
     (células puladas ENTRAM na tabela com status); **shift mínimo do placebo
     derivado da janela de evento**; parâmetros no-op REMOVIDOS do config
     (cash_return, weight_scheme, signal_rank, event_study.newey_west_lags,
     flags de metrics).
  7. **Testes novos matam os mutantes apontados**: janela do ranking, notional
     de saída, calendário, DSR por derivação independente, NW sob
     autocorrelação, PODER do placebo de datas falsas, alpha≠0 no market
     model, filtro de liquidez PIT. Suíte: 138 testes nos dois interpretadores.

---

## ADR-019 — Análise de decay: testes de diferença por tipo de unidade (Fase 5)

- **Status:** Aceita
- **Contexto:** A pergunta secundária (o sinal decaiu pós-publicação?) exige
  comparar IC, CAR e Sharpe entre pré e pós-2015 com TESTE DE DIFERENÇA. Cada
  métrica tem uma unidade estatística diferente.
- **Decisões:**
  1. **IC** — permutação dos rótulos de regime sobre os **IC por período
     (trimestre)**: a unidade quase-i.i.d. natural. H0: "regime não importa".
  2. **CAR** — permutação dos rótulos de regime sobre o **CAR por evento**.
  3. **Sharpe** — **bootstrap de blocos móveis** dentro de cada regime (bloco =
     H): respeita a autocorrelação induzida por posições sobrepostas, que uma
     permutação i.i.d. destruiria; p bicaudal pela fração das reamostragens no
     lado oposto ao sinal da diferença.
  4. **Amostra COMPLETA** (não o bloco DEV): o decay é resultado final
     descritivo, não seleção de hiperparâmetro — o holdout é revelado aqui,
     junto com o script 09.
  5. **Ambos os desfechos são válidos** (sobreviveu / decaiu) — o teste é
     honesto, não busca confirmar um lado (McLean & Pontiff como referência).
- **Nota de teste (armadilha nº 5):** os testes de "sem decay" usam regimes
  LITERALMENTE idênticos (diferença exata 0 → p alto por construção) em vez de
  ruído com seed — sob H0 real o p é uniforme e uma asserção `p > 0.20` com seed
  fixa seria flaky. Os testes de "decay detectado" usam efeito forte, robusto ao
  ruído de estimação.

- **Revisão (auditoria multi-agente da Fase 5, 2026-07-07):** o núcleo do decay
  passou; três correções aplicadas:
  1. **CRÍTICO (teste, não código):** os testes de decay não fixavam o bootstrap
     de blocos — trocá-lo por i.i.d. (block=1) sobrevivia à suíte, e em dados
     autocorrelacionados isso INVERTE a conclusão (declara decay falso). Teste
     novo com série AR(1) (φ≈0.6): block=5 dá p=0.18 (correto: não significativo)
     vs. i.i.d. p=0.025 (falso positivo) — pina que o `block` importa.
  2. **relative_capital_table** normalizava por `curve.iloc[0]` (= 1+r₀),
     descartando o retorno do dia 0 do retorno total/CAGR; corrigido para a base
     1.0 real.
  3. **Bootstrap circular** (Politis-Romano): inícios em [0, n) com wraparound —
     cobertura uniforme (o não-circular sub-amostrava as bordas ~10x; o `% n`
     era código morto). Impacto estatístico desprezível, mas alinha código e
     intenção documentada.
  Refutados pelos céticos: "report.py sem testes" (já havia `test_report.py`) e
  "testes de decay flaky" (robustos em 300 seeds). Suíte: 155 testes.

---

## ADR-020 — Checkpointing do scoring FinBERT (resiliência a interrupção)

- **Status:** Aceita
- **Contexto:** O scoring das ~307k falas elegíveis leva ~8h em CPU (sem GPU).
  Em três tentativas o processo morreu no meio (última em 55,4%) — o log
  mostrava um buraco de 3h e parada seca SEM traceback: a máquina suspendeu e o
  processo foi morto pelo SO, não por erro de Python. Recomeçar do zero a cada
  interrupção é inviável.
- **Decisão:** o scoring é **checkpointado em fatias**
  (`score_utterances_checkpointed`): as falas elegíveis (filtro determinístico,
  ordem estável) são divididas em fatias de `finbert.checkpoint_shard_utterances`
  (4000); cada fatia é salva em `data/interim/score_shards/shard_XXXXX.parquet`
  assim que concluída. Numa nova execução, fatias existentes são LIDAS do disco
  em vez de reprocessadas — o restart retoma de onde parou. O script 03 é
  idempotente; um teste prova que a 2ª execução não reinvoca o modelo.
- **Operação:** o job roda num loop auto-recuperável (relança até produzir a
  linha `[03]`, cap de 40 tentativas) e a suspensão automática foi desativada no
  modo tomada (`powercfg /change standby-timeout-ac 0`; reverter com um valor em
  minutos). Trade-off aceito: a ordenação por comprimento (batching) passa a ser
  POR FATIA, não global — perda de eficiência desprezível (4000 falas têm
  diversidade de comprimento suficiente) em troca de resiliência.

---

## ADR-021 — Escopo: manter SÓ a tese Brockman (não incorporar o "Tone Distance" do Angelo)

- **Status:** SUPERADA pelo ADR-023 (pivô Angelo). A decisão original de NÃO
  pivotar para o Angelo foi revertida: o projeto adotou a tese "Tone Distance"
  (distância de tom ENTRE OS GESTORES) como tese principal, medida com FinBERT.
  Registro mantido por honestidade — a virada faz parte da história do trabalho.
- **Contexto:** O paper Angelo et al. (2025, Financial Review) — "Tone Distance"
  — mede a variância de tom ENTRE MANAGERS numa call (desacordo gerencial), um
  conceito DIFERENTE da nossa tese principal (gestão×analistas, Brockman 2015).
  O time levantou se deveríamos (a) pivotar a tese para o Angelo, por ser mais
  recente, ou (b) adicioná-lo como feature de enriquecimento.
- **Alternativas consideradas e rejeitadas:**
  1. **Pivotar para o Angelo** — rejeitado: exigiria ID individual confiável de
     cada gestor por NOME (problema de dados bem mais difícil), a poucas semanas
     do prazo, jogando fora a tese Brockman já pronta e auditada. "Mais recente"
     não é critério de nota; o edital premia execução e coerência.
  2. **Adicionar uma feature `mgmt_tone_dispersion`** (dispersão só das falas da
     gestão) à grade — rejeitado pelo TIME: seria uma feature que não
     analisaríamos a fundo, e o edital PUNE "complexidade sem fundamentação".
     Um apêndice meio-usado dilui o foco e abre flanco na arguição.
- **Decisão:** manter o projeto **enxuto e focado na tese Brockman**
  (gestão×analistas), modernizada por FinBERT + Jensen-Shannon + decay. O Angelo
  é citado e diferenciado na documentação (ADR-000), como a literatura da qual
  nos distinguimos — não uma feature nossa. Decisão de ESCOPO, tomada
  conscientemente: profundidade e coerência valem mais que amplitude rasa.

## ADR-022 — Sinal da estratégia: divergência COM SINAL, vendida (direção Brockman)

- **Status:** SUPERADA pelo ADR-023 (pivô Angelo). Esta decisão fixava a direção
  do sinal da tese Brockman (vender a divergência gestão×analistas alta). Com o
  pivô para a tese "Tone Distance" do Angelo, o construto operável mudou por
  completo (distância de tom entre gestores, comprada quando ALTA — ver ADR-024);
  a direção Brockman deixou de valer. Registro mantido por honestidade.
- **Contexto:** Na primeira execução sobre dados reais, o event study encontrou
  o efeito forte (HML = −1,67%, p ≈ 0,000, placebo p = 0,001: divergência ALTA →
  retorno ANORMAL menor), mas o backtest long-short **perdia dinheiro**
  (Sharpe líq. −0,28 no bloco DEV). A investigação mostrou que a estratégia
  estava operando o sinal **na direção errada**, por duas causas:
  1. A feature-gatilho era a **distância de Jensen-Shannon** (`mgmt_analyst_divergence`),
     que é **sem sinal**: mede o TAMANHO do desacordo, não para que lado. Detecta
     informação (ótima no event study), mas não carrega a direção da aposta —
     como sinal de trade, mesmo invertida rende ~0 (Sharpe +0,01).
  2. `assign_sides` assumia por padrão "percentil alto → comprar". Para uma
     feature sem sinal isso é arbitrário; o event study mostrou ser o lado
     perdedor. A própria justificativa econômica da lâmina ("a gestão mantém tom
     otimista mesmo sob perguntas céticas" → excesso de otimismo → correção)
     implica o OPOSTO: **vender** a divergência alta.
- **Alternativas consideradas:**
  1. Flipar o sinal da JS pura "no olho" — rejeitado: seria data-snooping, e a JS
     invertida rende ~0 mesmo assim (é a feature errada para trade).
  2. Manter como estava — rejeitado: contraria a evidência E a própria tese.
- **Decisão:** o gatilho da estratégia passa a ser a **divergência COM SINAL**
  (`mgmt_analyst_divergence_signed` = tom da gestão − tom dos analistas no Q&A),
  operada com **`signal_direction = -1`** (vender quando a gestão está otimista
  demais frente aos analistas). No bloco DEV isso leva o long-short de −0,28 para
  **+0,29 líquido** (market-neutral, exposição residual ~6%); a direção OPOSTA
  (comprar o excesso) rende −0,57, confirmando que o efeito é real e direcional.
- **Por que NÃO é overfitting:** a direção é fixada por (a) a justificativa
  econômica da tese (a priori) e (b) o event study no bloco DEV — não por busca
  de Sharpe máximo. É a **correção de um erro de especificação** (a lâmina dizia
  "comprar alto"; o construto e a direção corretos são os do Brockman). O sinal é
  pré-comprometido e implementado como parâmetro de config auditável
  (`signal_direction`, validado em {−1,+1}), não como transformação escondida. O
  **holdout de 18 meses**, intocado, é o juiz out-of-sample.
- **Honestidade declarada:** +0,29 é POSITIVO e coerente, porém MODESTO — sobre
  ~18 anos, t ≈ 1,3 (não significante isoladamente). O resultado forte continua
  sendo o event study; a estratégia é um alpha market-neutral modesto, cujo
  benchmark justo é zero/caixa (não um índice alavancado em beta como o QQQ). A
  JS pura segue como detector no event study e na grade de robustez.

---

## ADR-023 — Pivô para a tese Angelo (Tone Distance) medida com FinBERT

- **Status:** Aceita. SUPERA ADR-021 e ADR-022.
- **Contexto:** O projeto nasceu sobre a tese de Brockman, Li & Price (2015) —
  divergência de tom entre gestão e analistas na mesma call. A ADR-021 havia
  DECIDIDO conscientemente NÃO pivotar para o "Tone Distance" de Angelo et al.
  (2025, *Financial Review*), por escopo e prazo. Reavaliando com a orientadora,
  o time reverteu essa decisão: a tese do Angelo é mais nova, mais inovadora e
  mais alinhada à orientação. O que a ADR-021 tratou como custo proibitivo
  (identificar cada gestor por nome) revelou-se tratável com a heurística de
  papéis já construída (ADR-006), que separa falas por orador dentro da call.
- **O que é o "Tone Distance" (Angelo et al. 2025, "Tone Distance: Managerial
  Tone Divergence and Market Reaction to Earnings Announcements"):** o sinal é a
  **distância de tom ENTRE OS GESTORES** dentro da MESMA earnings call — o
  desacordo de tom entre os executivos que falam na call — e NÃO a divergência
  gestão×analistas do Brockman. Construção da métrica: o FinBERT dá a cada fala
  uma distribuição [negativo, neutro, positivo]; por gestor, agregam-se
  (ponderadas por tokens, mesma lógica das ADR-015/ADR-017) as coordenadas
  (probabilidade positiva, probabilidade negativa); calcula-se a distância
  euclidiana de cada gestor até a MÉDIA dos gestores da call; a **Tone Distance é
  a média dessas distâncias**. Exige **≥2 gestores** por call (cobertura de
  **97,5%** das calls; mediana de **4 gestores/call**).
- **Alternativas consideradas:**
  1. **Manter a tese Brockman** (o que a ADR-021/ADR-022 prescreviam) — rejeitado:
     tese mais antiga, e o resultado de trade era um alpha market-neutral modesto
     e não-significante isoladamente (ADR-022, t ≈ 1,3). A orientação e o
     potencial de inovação favorecem o pivô.
  2. **Rodar as DUAS teses em paralelo** — rejeitado pela mesma razão da ADR-021:
     o edital pune "complexidade sem fundamentação"; profundidade numa tese vale
     mais que amplitude rasa em duas. O Brockman fica como literatura de contraste
     e a JS gestão×analistas sobrevive apenas como um dos confundidores a limpar
     (ver ADR-024), não como tese.
- **Decisão:** adotar a tese **"Tone Distance" (Angelo et al. 2025)** como tese
  PRINCIPAL do projeto — distância de tom entre os gestores da mesma call. Nossa
  **contribuição/modernização** sobre o Angelo: medimos o tom com **FinBERT**
  (`yiyanghkust/finbert-tone`) no lugar do **dicionário de palavras
  Loughran-McDonald** que o Angelo usa. FinBERT é um classificador contextual
  (entende negação, ironia e contexto que um dicionário de contagem de palavras
  ignora), e reutiliza toda a infraestrutura de scoring já auditada
  (ADR-006/007/015/020). **ADR-021 e ADR-022 ficam SUPERADAS** por esta decisão:
  a ADR-021 porque o projeto de fato pivotou para o Angelo (o oposto do que ela
  decidiu); a ADR-022 porque a direção do sinal Brockman deixou de reger a
  estratégia — o construto operável agora é o do Angelo (ver ADR-024). O
  histórico é PRESERVADO: a virada é registrada honestamente, não apagada.
- **Nota sobre a exigência de IA generativa (edital):** o FinBERT é um
  classificador, não um modelo generativo. A exigência de IA generativa do
  processo é cumprida pelo **PROCESSO de desenvolvimento** — o projeto foi
  construído em pair-programming com o Claude Code (planejamento, código,
  revisões adversariais multi-agente, reprodução independente).

---

## ADR-024 — Sinal LIMPO: controles + residualização estritamente-passada

- **Status:** Aceita.
- **Contexto:** Adotada a Tone Distance (ADR-023), a primeira pergunta é se a
  distância de tom CRUA prevê retorno. A resposta empírica é **NÃO**: no teste
  cru, sem controles, não há nada (t < 0,4). A distância de tom crua é
  **confundida** por variáveis que a acompanham e que têm efeito próprio no
  retorno — nível geral de tom da call, tamanho da empresa, setor, e o tom dos
  analistas. Uma call grande e otimista tende a ter mais gestores e mais dispersão
  mecânica; sem separar isso, a "distância" mede o confundidor, não o desacordo
  genuíno. É exatamente o ponto do método do Angelo: o efeito só aparece com
  CONTROLES.
- **Método (o do Angelo):** regressão da distância de tom sobre os retornos com
  **CONTROLES** + **efeitos fixos de empresa e de trimestre** + **erros agrupados
  por empresa** (clusterizados). Os confundidores controlados são: **nível de tom
  da call, tom dos analistas, distância de tom dos analistas, tamanho e
  comprimento** da call.
- **Sinal operável (nossa construção):** o **SINAL LIMPO** (`tone_distance_clean`)
  é o **resíduo ESTRITAMENTE-PASSADO** da distância de tom regredida sobre esses
  cinco confundidores — isto é, a parte da distância de tom que NÃO é explicada
  por nível de tom, tom dos analistas, distância de tom dos analistas, tamanho e
  comprimento, estimada usando SOMENTE informação disponível ANTES do evento
  (mesma disciplina anti-look-ahead das janelas `_idio` e do gatilho por
  percentil PIT, ADR-004/ADR-017/ADR-018). Residualizar com dados do futuro
  vazaria; por isso "estritamente-passada". Cobertura do sinal limpo: **~63%** das
  calls (exige fala de analista para estimar o tom/distância dos analistas + histórico suficiente
  para a residualização PIT).
- **Direção (a priori, fixada pelo paper):** comprar (long) a distância de tom
  **ALTA**. A leitura econômica é de **prêmio de risco**: mais desacordo entre os
  gestores = mais incerteza = maior retorno exigido. A distância de tom alta CAI
  no anúncio, mas rende MAIS nos ~1–3 meses seguintes. A direção é fixada a priori
  pela Tabela 6 do Angelo — NÃO é escolhida por busca de Sharpe máximo (não é
  data-snooping).
- **Alternativas rejeitadas:**
  1. **Operar a distância de tom CRUA** — rejeitado: não prevê nada (t < 0,4) e,
     pior, VIRA NEGATIVA no holdout (ver números). Sem limpar os confundidores o
     sinal não é robusto.
  2. **Residualizar com a amostra inteira** (in-sample) — rejeitado: usaria o
     futuro na estimação dos coeficientes (look-ahead), inflando o resultado.
- **Números-chave (sinal limpo):**
  - **Event study:** retorno anormal do tercil alto menos o do tercil baixo =
    **+1,5%**, significância forte (p ≈ 0,000); placebo de datas falsas p = 0,001.
  - **Regressão controlada** (efeitos fixos empresa+trimestre, erros agrupados,
    distância de tom padronizada): retorno de **1 mês** = **+0,41% por
    desvio-padrão** (t = 2,39; p = 0,017); retorno de **3 meses** = **+1,01% por
    desvio-padrão** (t = 2,50; p = 0,013). **n = 3584**. No teste CRU, nada
    (t < 0,4).
  - **Backtest market-neutral** (sinal limpo, horizonte 3 meses), bloco de
    desenvolvimento: **Sharpe líquido 0,49**; exposição líquida ~3% (de fato
    neutra); BATE os 20 placebos aleatórios (melhor acaso 0,36).
  - **Período completo com holdout** (relatório final): long-short (market-neutral)
    Sharpe líquido **0,47**, drawdown máximo −7%, retorno anualizado +0,9%, retorno
    total +16%; long-only Sharpe 0,94, drawdown −6%, retorno anualizado +3,2%
    (referências: QQQ Sharpe 0,93 / retorno anualizado 19,5%; SPY Sharpe 0,78).
  - **Holdout** (18 meses lacrados, fora da amostra): a estratégia market-neutral
    fica POSITIVA (~+0,3 em 3 meses, +0,45 em 1 mês) — enquanto o sinal CRU VIRA
    NEGATIVO no holdout. É a prova de que limpar os confundidores dá robustez.
  - **Robustez:** probabilidade de sobreajuste (PBO) 0,26 (aceitável); placebo do
    próprio sinal p = 0,069 (marginal, declarado); o sinal limpo é a CAMPEÃ de
    uma grade de 60 combinações; Sharpe deflacionado (best-de-60) 0,18.
  - **Estabilidade no tempo (pré vs. pós-2015):** o efeito é CONCENTRADO no
    período recente — o poder preditivo vira de negativo para positivo (mudança
    significativa, p = 0,015); Sharpe de −0,16 para +0,78 (p = 0,054). LIMITAÇÃO
    declarada: é um fenômeno da era moderna, não dos 20 anos inteiros.
  - **Verificação independente:** um agente reproduziu a regressão principal do
    zero e bateu exatamente; sem vazamento de futuro nos controles.
- **Honestidade declarada:** magnitude modesta (Sharpe ~0,47 market-neutral) — é
  um alpha descorrelacionado de baixo risco, não bate o índice em nível absoluto;
  efeito concentrado pós-2015; placebo próprio marginal (0,069); cobertura ~63%;
  tamanho é proxy de LIQUIDEZ (não capitalização, por falta de dados de balanço
  abertos); o efeito de anúncio de curtíssimo prazo do Angelo não é observável
  aqui (entramos no dia seguinte, T+1). O resultado FORTE é o event study + a
  regressão controlada; o backtest é um alpha modesto porém robusto ao holdout.

---

## ADR-025 — Revisão de fidelidade à metodologia do Angelo (correções pós-auditoria)

- **Status:** Aceita.
- **Contexto:** Antes de fechar, revisamos a implementação ponto a ponto contra a
  metodologia do artigo (identificação de gestores, cálculo da distância, e demais
  quesitos). A auditoria confirmou que o NÚCLEO é fiel (fórmula da distância
  euclidiana entre gestores, uso da fala inteira de cada gestor, ≥2 gestores,
  direção comprada, regressão controlada com efeitos fixos), e as divergências
  grandes são JUSTIFICADAS (FinBERT no lugar do dicionário; controles reduzidos
  por serem dados abertos; entrada em T+1 impede observar a janela do anúncio).
  Mas encontrou três correções materiais, aplicadas:
- **Correção 1 — identificação de gestores (calls sem Q&A):** quando o marcador de
  Q&A não é detectado (~20% das calls), a heurística rotulava a call INTEIRA como
  "management", incluindo analistas — contaminando a distância de tom. Evidência:
  a contagem de "gestores" chegava a 30 por call (impossível). Correção: a
  distância de tom só é computada em calls com Q&A detectado (onde o rótulo de
  papel é confiável). Efeito: a cauda caiu (mediana 3, máx 25 gestores). O sinal
  OPERÁVEL (limpo) ficou inalterado (Sharpe 0,49/holdout 0,31), porque essas calls
  já não entravam nele (o tom dos analistas exige Q&A) — ou seja, corrigiu o método
  sem inflar resultado. O host de relações com investidores é MANTIDO como gestor:
  o Angelo mede "todos os representantes da empresa que falam", não só executivos.
- **Correção 2 — winsorização 5/95 (script 10):** o Angelo winsoriza todas as
  variáveis nos percentis 5 e 95 (Seção 4.1); nós passamos a fazer o mesmo na
  regressão. O efeito CONTINUA significativo depois de aparar os extremos, ou seja,
  não é fruto de poucos retornos gigantes: **1 mês +0,39% por desvio-padrão
  (t = 2,62; p = 0,009)**; **3 meses +0,76% por desvio-padrão (t = 2,23;
  p = 0,026)**. (A estratégia não precisa de winsorização: o gatilho ranqueia por
  percentil, já robusto a extremos.) A regressão controlada virou script
  reproduzível (`scripts/10_controlled_regression.py`).
- **Correção 3 — ajuste pelo tom dos analistas:** já embutido. O Angelo, na
  robustez (Tabela 7), subtrai o tom dos analistas; nós fazemos o equivalente, mais
  geral — o sinal limpo é o resíduo DEPOIS de remover o tom dos analistas E a
  distância de tom dos analistas (ambos são confundidores da residualização,
  ADR-024). A versão literal (subtrair resposta por pergunta) seria redundante e,
  num ajuste ingênuo por call, um não-efeito (subtrair a mesma constante de todos
  os gestores não muda a dispersão, por invariância a translação).
- **Limitações de fidelidade que PERMANECEM declaradas:** identificação de gestores
  é heurística (o dataset não traz o papel, ao contrário do Capital IQ do Angelo);
  gestor que só fala no Q&A é rotulado analista; controles de balanço ausentes
  (dados fechados); janela do anúncio não observável (T+1). Nenhuma escondida.
