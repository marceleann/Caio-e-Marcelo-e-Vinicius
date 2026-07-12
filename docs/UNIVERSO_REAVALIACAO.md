# Reavaliação do Universo — "isso é tech?" (2026-07-09)

Feita a pedido do Marcelo: reavaliar se cada empresa é/foi tech e **deixar viável
rodar com e sem** as fronteiriças. **Nada foi removido**: a decisão vira um filtro
via `data/interim/call_blocks.parquet` (etiqueta por call), então "com/sem" é só
selecionar blocos nos Stages 3–4, sem reprocessar nem descartar dados.

Base factual: `sector`/`industry` atuais do Yahoo (`data/raw/sector_cache.parquet`)
+ classificação GICS de conhecimento público. Universo = 116 tickers com call
(5.452 calls, 2005–2025).

## Núcleo (tech inequívoco) — sempre dentro

**87 tickers, 4.153 calls.** Semicondutores (NVDA, AVGO, AMD, INTC, QCOM, TXN,
MU, AMAT, LRCX, KLAC, ADI, MCHP, NXPI, ON, MPWR, SWKS, QRVO, TER…), software
(MSFT, ORCL, CRM, ADBE, NOW, INTU, SNPS, CDNS, ADSK, PANW, CRWD, FTNT, WDAY,
PLTR…), hardware/redes (AAPL, CSCO, DELL, HPQ, HPE, NTAP, STX, WDC, ANET, JNPR,
FFIV, GLW, APH, TEL, MSI…), TI/serviços (IBM, ACN, CTSH, EPAM, IT, CDW, DXC),
internet/mídia interativa (GOOGL, GOOG, META, NFLX, EA, TTWO, ATVI, MTCH). São
tech por qualquer critério — não dependem de confirmação do Yahoo.

## `core_borderline` — NOVO toggle proposto (dentro por padrão)

**7 tickers, 225 calls: GRMN, TDY, FTV, VNT, ROP, LDOS, UIS.**
O Yahoo os marca "Technology", mas economicamente são discutíveis:

| Ticker | calls | O que é | Por que é fronteiriço |
|---|---|---|---|
| ROP  | 60 | Roper | Conglomerado software+industrial diversificado |
| GRMN | 53 | Garmin | Eletrônicos de consumo (GPS/wearables) |
| LDOS | 42 | Leidos | Serviços de TI para **defesa** (Industrials-ish) |
| FTV  | 36 | Fortive | Instrumentação industrial (spin da Danaher) |
| TDY  | 21 | Teledyne | Instrumentação diversificada |
| VNT  |  6 | Vontier | Tecnologia de mobilidade / varejo de combustível |
| UIS  |  7 | Unisys | Serviços de TI legados (pequeno, antigo) |

**Recomendação:** manter no núcleo por padrão (é o que o método declarado do
`config.yaml`/ADR-001 já faz), mas rodar a robustez **sem** eles. CIEN, VIAV
(equip. de rede), IPGP (fotônica/lasers) e IAC (internet) **ficam no núcleo** —
são tech genuíno, não entram neste toggle.

## Blocos de sensibilidade (com/sem) — já existentes, confirmados

| Bloco | tickers | calls | Justificativa factual (por que é discutível como "tech") |
|---|---|---|---|
| `payments` | V, MA, PYPL, FIS, FI, GPN, CPAY | 361 | Redes de pagamento. **GICS moveu de IT → Financials em 2023.** |
| `non_gics_tech` | AMZN, BKNG, EXPE, EBAY, TSLA, UBER, ABNB | 357 | Marketplaces/tech-enabled, mas **Consumer Discretionary** (AMZN nunca foi IT; TSLA é automóvel). |
| `payroll` | ADP, PAYX, BR | 213 | Folha/back-office. Yahoo diz Tech, **GICS diz Industrials.** |
| `solar_tech` | FSLR, ENPH, SEDG | 89 | GICS-IT, mas o **negócio é energia solar**. |
| `ems` | JBL, SANM | 54 | Manufatura eletrônica terceirizada (montagem por contrato, não IP). |

## Recomendação de recortes para os Stages 3–4

1. **Universo primário (defende a tese "US tech firms" do Angelo):**
   `in_strict_tech` = core + core_borderline = **4.378 calls, 94 tickers**.
2. **Núcleo puro (mais conservador):** `in_core_pure` = **4.153 calls, 87 tickers**.
3. **Robustez com/sem cada bloco:** adicionar `payments`, `non_gics_tech`,
   `payroll`, `solar_tech`, `ems` um a um e reportar o efeito no sinal.

Todos os resultados serão reportados sob o recorte primário, com a tabela de
sensibilidade (com/sem) como robustez — atende ao critério de "mitigação de
vieses" (Backtest, 15%) e evita "escolha oportunista de universo" (ponto
negativo do edital).

## Artefato gerado

`data/interim/call_blocks.parquet`: colunas `call_id, ticker, block,
in_strict_tech, in_core_pure`. Determinístico a partir de `universe.parquet` +
o conjunto `core_borderline` acima. Não toca em nenhum dado existente.
