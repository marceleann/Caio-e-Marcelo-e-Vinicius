# data_schemas.md — Esquemas dos arquivos parquet

> **Gerado incrementalmente a partir da Fase 1.** Cada tabela parquet produzida
> pelo pipeline é documentada aqui: caminho, colunas, tipos e significado
> (regra de engenharia nº 11). Placeholder na Fase 0 — preenchido conforme os
> módulos de dados/features/backtest são construídos.

## Convenções
- Formato: Apache Parquet (backend `pyarrow`).
- Datas/timestamps: timezone-aware em `US/Eastern`, salvo indicação contrária.
- Chaves de empresa já com `ticker_aliases` aplicados (ticker canônico).

## Índice de tabelas

| Arquivo | Produzido por | Descrição |
|---------|---------------|-----------|
| `data/raw/calls_all.parquet` | script 01 | Metadados de TODAS as calls do dataset (para cobertura/survivorship). |
| `data/raw/prices.parquet` | script 01 | Painel long de preços do universo + mercado + benchmarks. |
| `data/interim/universe.parquet` | script 01 | Tabela de pertencimento do universo (camadas, blocos, presença). |
| `data/interim/calls.parquet` | script 01 | Metadados das calls do universo aprovadas na qualidade. |
| `data/interim/utterances.parquet` | script 01 | Falas (uma por linha) das calls do universo. |
| `data/outputs/coverage_by_year.parquet` | script 00 | Nº de calls e empresas por ano (dataset completo). |
| `data/outputs/coverage_by_ticker.parquet` | script 00 | Nº de calls e janela de anos por empresa (universo). |
| `data/interim/utterances_roles.parquet` | script 02 | Falas anotadas com seção e papel inferido. |
| `data/outputs/roles_validation_sample.csv` | script 02 | Amostra p/ conferência manual (seed fixa). |
| `data/interim/utterance_scores.parquet` | script 03 | Distribuições FinBERT [neg, neu, pos] por fala. |
| `data/processed/events.parquet` | script 04 | Tabela canônica de eventos: features + decisão PIT + retornos futuros. |
| `data/outputs/event_study_cars.parquet` | script 05 | CAR por evento (market model). |
| `data/outputs/event_study_groups.parquet` | script 05 | CAR médio por tercil da feature + linha HML. |
| `data/outputs/portfolio_returns.parquet` | script 06 | Séries diárias por variante (net, turnover, exposição líquida, hedgeada). |
| `data/outputs/portfolio_summary.parquet` | script 06 | Métricas por variante + placebo de sinal aleatório. |
| `data/outputs/robustness_grid.parquet` | script 07 | Grade horizonte×feature completa (células puladas com status). |
| `data/outputs/robustness_returns.parquet` | script 07 | Séries de retorno de cada célula (insumo do PBO). |
| `data/outputs/decay_comparison.parquet` | script 08 | IC/CAR/Sharpe pré vs pós-2015 + p da diferença. |
| `data/outputs/strategy_report.parquet` | script 09 | Tabela consolidada (variantes + benchmarks) do tearsheet. |
| `data/outputs/tearsheet.png` | script 09 | Curva de capital vs QQQ/SPY + drawdown. |

---

## `calls_all.parquet` — metadados de todas as calls
| Coluna | Tipo | Significado |
|--------|------|-------------|
| `call_id` | str | Chave primária única (`TICKER_ANOQtri`, com sufixo se colidir). |
| `ticker` | str | Ticker CANÔNICO (aliases aplicados). |
| `ticker_raw` | str | Ticker ORIGINAL do dataset (pré-alias); usado no survivorship. |
| `company` | str\|NA | Nome da empresa, se disponível. |
| `call_datetime` | datetime tz (US/Eastern) | Timestamp da call (base da regra T+1). |
| `has_time` | bool | `False` se horário ausente/meia-noite (fallback conservador). |
| `year`, `quarter` | Int64 | Ano/trimestre (explícitos ou derivados do datetime). |

## `utterances.parquet` — falas
| Coluna | Tipo | Significado |
|--------|------|-------------|
| `call_id` | str | FK para `calls`. |
| `utterance_idx` | int | Ordem da fala na call (0-based; preserva a sequência). |
| `speaker` | str\|None | Nome do orador (papel inferido na Fase 2). |
| `text` | str | Texto da fala. |
| `n_chars` | int | Comprimento do texto. |

## `calls.parquet` — metadados do universo (aprovados na qualidade)
Mesmas colunas de `calls_all.parquet` + `n_utterances` (Int64) e `n_chars_total` (Int64).

## `universe.parquet` — pertencimento do universo
| Coluna | Tipo | Significado |
|--------|------|-------------|
| `ticker` | str | Ticker canônico candidato. |
| `in_core` | bool | Pertence ao núcleo sempre-dentro (curado ∪ rede − sensibilidade). |
| `sensitivity_block` | str\|None | Bloco de sensibilidade (`payments`/`non_gics_tech`/`solar_tech`/`ems`). |
| `is_delisted` | bool | Está na lista `delisted_check` (survivorship). |
| `layer` | str | Camada de origem primária (auditoria). |
| `sector`, `industry` | str\|None | Classificação atual (yfinance), quando consultada. |
| `in_dataset` | bool | Tem transcrição no dataset. |
| `in_prices` | bool | Tem série de preço no yfinance. |

## `utterances_roles.parquet` — falas anotadas (script 02)
Colunas de `utterances.parquet` +:
| Coluna | Tipo | Significado |
|--------|------|-------------|
| `section` | str | `remarks` (antes do Q&A) ou `qa`. |
| `role` | str | `management` \| `analyst` \| `operator` \| `unknown` (ADR-006). |
| `operator_confirmed` | bool | Analista também anunciado pelo Operator (refinamento regex). |

## `utterance_scores.parquet` — tom por fala (script 03)
| Coluna | Tipo | Significado |
|--------|------|-------------|
| `call_id`, `utterance_idx` | str, int | FK para a fala. |
| `role`, `section` | str | Copiados da anotação (conveniência de join). |
| `p_neg`, `p_neu`, `p_pos` | float | Distribuição FinBERT na ordem CANÔNICA do projeto. |
| `net_tone` | float | `p_pos − p_neg`. |
| `n_tokens` | int | Tokens da fala (tokenizer real). |
| `n_chunks` | int | Nº de chunks usados (falas longas > 512 tokens). |

## `events.parquet` — tabela canônica de eventos (script 04)
| Coluna | Tipo | Significado |
|--------|------|-------------|
| `call_id`, `ticker`, `call_datetime` | str, str, datetime tz | Identificação do evento. |
| `p_neg`,`p_neu`,`p_pos` | float | Distribuição agregada da call (ponderada por tokens). |
| `net_tone` | float | Feature 1: p_pos − p_neg da call. |
| `delta_tone` | float | Feature 2: variação vs. call anterior da MESMA empresa. |
| `tone_js_prev` | float | Feature 3: distância JS vs. call anterior da mesma empresa. |
| `tone_dispersion` | float | Feature 4: desvio-padrão (populacional) dos net_tones das falas. |
| `mgmt_analyst_divergence` | float | Feature 5 (COMPARAÇÃO/legado — tese antiga de Brockman, mantida como robustez): distância de Jensen-Shannon entre analistas_qa e gestão_qa; NaN sem Q&A. |
| `mgmt_analyst_divergence_signed` | float | Versão assinada: net(gestão_qa) − net(analistas_qa). |
| `qa_remarks_gap` | float | Feature 6: net(gestão remarks) − net(gestão qa). |
| `delta_tone_idio`, `tone_js_prev_idio`, `mgmt_analyst_divergence_idio` | float | Feature 7: versões ajustadas por pares (janela PIT 90d). |
| `qa_detected` | bool | Q&A detectado na call (filtro p/ features dependentes de papel). |
| `n_utt_scored`, `n_utt_mgmt_qa`, `n_utt_analyst_qa` | int | Contagens de falas pontuadas. |
| `intended_date`, `decision_rule` | datetime, str | Data-alvo e regra (same_open/next_open). |
| `decision_date` | datetime (NaT se sem pregão) | Pregão efetivo da decisão. |
| `decision_open_ts` | datetime tz | Instante do open da decisão (usado pela guarda). |
| `entry_open` | float | Preço de entrada (open de d0). |
| `ret_3`, `ret_5`, `ret_10` | float | Retornos futuros: open d0 → close d_{H−1} (ADR-017). |

## `prices.parquet` — painel long de preços
| Coluna | Tipo | Significado |
|--------|------|-------------|
| `date` | datetime (naive, normalizado) | Pregão. |
| `ticker` | str | Ticker (inclui `^GSPC`, `SPY`, `QQQ`). |
| `open`,`high`,`low`,`close` | float | OHLC ajustado (`auto_adjust=True`). |
| `volume` | float | Volume. |
| `ret_cc` | float | Retorno close-to-close diário. |
| `dollar_volume` | float | `close * volume` (base do filtro de liquidez PIT). |
