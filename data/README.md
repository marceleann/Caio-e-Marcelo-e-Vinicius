# Dados do projeto — mapa de replicação

Todos os dados derivados estão versionados aqui para que qualquer pessoa
(orientadora, banca, colega) **replique as análises sem reprocessar nada**:
basta clonar o repo e rodar os scripts de regressão. As fontes primárias são
públicas (HuggingFace, SEC EDGAR, yfinance, Ken French) e os scripts refazem
tudo do zero se necessário.

## O essencial (replicação das análises S&P 500)

| arquivo | o que é | produzido por |
|---|---|---|
| `interim/sp500/tone_distance_sp500.parquet` | Tone Distance (LM) + controles textuais por call | `sp500_phase_a.py` |
| `interim/sp500/speaker_counts.parquet` | contagens LM **por fala** com papel do orador | `sp500_speaker_counts.py` |
| `interim/sp500/td_variants.parquet` | TD ponderada por palavras (T8c1), excl. mediana, adjusted (T7) | `sp500_table78.py` |
| `interim/sp500/events_sp500_car3.parquet` | eventos com CAR exato do paper (CAPM 100d/gap50) | `sp500_car_v3.py` |
| `interim/sp500/events_sp500_paper.parquet` | + fundamentos EDGAR (defs. exatas do Apêndice A) + FF49 | `sp500_fund_v2.py`, `sp500_ff49.py` |
| `interim/sp500/tone_distance_finbert.parquet` | TD com tom FinBERT por sentença (braço GenAI) | `sp500_td_finbert.py` |
| `interim/sentence_scores_shards/` | scores FinBERT sentença a sentença (1 parquet/call; em expansão) | `sp500_score_finbert.py` |
| `raw/sp500/fundamentals_facts_v2.parquet` | fundamentos trimestrais XBRL (extrato do companyfacts.zip) | `sp500_fund_v2.py` |
| `raw/sp500/price_shards/` | preços diários (yfinance, ajustados e brutos) + ^GSPC | `sp500_phase_b.py` |
| `raw/rf_daily.parquet` | taxa livre de risco diária (Ken French) | `sp500_phase_b.py` |
| `raw/calls_all.parquet` | metadados das 33.362 calls (id, ticker, data/hora) — sem texto | `sp500_phase_a.py` |
| `raw/cik_map.parquet`, `raw/cik_sic.parquet`, `raw/Siccodes49.txt` | ticker→CIK→SIC→indústria FF49 | `sp500_ff49.py` |
| `interim/llm_roles_cache.jsonl` | classificação de oradores pelo LLM (robô CLI, Stage 1) | `stage1_llm_classify.py` |
| `outputs/papeis_por_orador.csv` | export legível da classificação de papéis | `11_export_classification.py` |

Para **replicar as regressões** (sem reprocessar texto): rodar diretamente
`sp500_paper_suite.py` (Tabelas 3/4/5 do paper), `sp500_table6.py` (retorno
mensal), `sp500_tdw_deep.py` (TD ponderada — resultado central),
`sp500_backtest.py` e `sp500_diag_controls.py`.

## O que NÃO está no repo (e como obter, só se quiser refazer do zero)

- **Transcrições completas** (texto): dataset público
  [kurry/sp500_earnings_transcripts](https://huggingface.co/datasets/kurry/sp500_earnings_transcripts)
  — os scripts baixam e cacheiam sozinhos na primeira execução.
- **`raw/companyfacts.zip`** (1,3GB, bulk XBRL da SEC): só é preciso para
  re-extrair fundamentos; o extrato já está versionado. URL no `.gitignore`.
- **Dicionário Loughran-McDonald** (`raw/LM_MasterDictionary.csv`): a licença
  da Notre Dame não permite redistribuição — baixar em
  [sraf.nd.edu](https://sraf.nd.edu/loughranmcdonald-master-dictionary/).
  Só é preciso para recontar palavras; as contagens já estão versionadas.
- Arquivos do universo tech antigo com texto integral (>100MB): regeneráveis
  pelos scripts numerados (`01_download_data.py` em diante).
