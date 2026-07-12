"""tonediv: biblioteca pura do sinal de Distância de Tom em earnings calls.

O sinal central é a Distância de Tom (Tone Distance, tese Angelo 2025 medida
com FinBERT): o desacordo de tom ENTRE OS GESTORES da mesma call. O nome do
pacote (``tonediv``) preserva a origem histórica do projeto, que PIVOTOU da
tese de divergência analistas×gestão de Brockman, Li & Price (2015) para a de
Angelo (ADR-021/022 superadas, ADR-023).

Este pacote (``src/tonediv``) contém APENAS lógica testável, sem I/O implícito
nem efeitos colaterais no import (regra de engenharia nº 10). A orquestração
(carregar config, salvar parquet) vive em ``scripts/``.

Subpacotes:
    data:      download e parse de transcrições/preços, seleção do universo.
    nlp:       papéis analista×gestão, segmentação, FinBERT, features.
    align:     alinhamento point-in-time texto->preço e guarda anti-look-ahead.
    backtest:  event study, portfólio calendar-time, métricas, validação, decay.
"""

__version__ = "0.1.0"
