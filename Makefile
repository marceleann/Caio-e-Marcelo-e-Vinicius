# =============================================================================
# Makefile — alvos de qualidade + um alvo por etapa do pipeline (regra nº 13).
# Uso: `make <alvo>`. Em Windows, rodar via Git Bash / WSL, ou usar os comandos
# `python -m ...` diretamente (ver README).
# PYTHON pode ser sobrescrito: `make test PYTHON=python3.11`.
# =============================================================================

PYTHON ?= python
CONFIG ?= config.yaml

.PHONY: help install lint format test \
        diagnostics download roles score features controls \
        event-study backtest robustness decay report regression classify all

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

# --- Qualidade ---------------------------------------------------------------
install:  ## Instala dependências pinadas + o pacote em modo editável
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

lint:  ## ruff (lint) + black (checagem de formatação), sem alterar arquivos
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m black --check src scripts tests

format:  ## Aplica black e correções automáticas do ruff
	$(PYTHON) -m black src scripts tests
	$(PYTHON) -m ruff check --fix src scripts tests

test:  ## Roda a suíte pytest
	$(PYTHON) -m pytest

# --- Pipeline (um alvo por script; ver scripts/) -----------------------------
diagnostics:  ## 00: horários, cobertura, qualidade e teste de survivorship
	$(PYTHON) scripts/00_diagnostics.py --config $(CONFIG)

download:  ## 01: baixa transcrições (HF) e preços (yfinance)
	$(PYTHON) scripts/01_download_data.py --config $(CONFIG)

roles:  ## 02: infere papéis analista x gestão (+ CSV de validação manual)
	$(PYTHON) scripts/02_infer_roles.py --config $(CONFIG)

score:  ## 03: aplica FinBERT e gera distribuições de tom por fala
	$(PYTHON) scripts/03_score_tone.py --config $(CONFIG)

features:  ## 04: constrói as features por call (incl. distância de tom) alinhadas a preço
	$(PYTHON) scripts/04_build_features.py --config $(CONFIG)

controls:  ## 04b: controles do Angelo + sinal LIMPO (residualização estritamente-passada)
	$(PYTHON) scripts/04b_angelo_controls.py --config $(CONFIG)

event-study:  ## 05: CAR via market model + placebo de datas falsas
	$(PYTHON) scripts/05_event_study.py --config $(CONFIG)

backtest:  ## 06: portfólio calendar-time (long-short e long-only)
	$(PYTHON) scripts/06_backtest_portfolio.py --config $(CONFIG)

robustness:  ## 07: grade completa de sensibilidade + DSR com n_trials real
	$(PYTHON) scripts/07_robustness_grid.py --config $(CONFIG)

decay:  ## 08: análise de decay pré/pós-2015
	$(PYTHON) scripts/08_decay_analysis.py --config $(CONFIG)

report:  ## 09: tearsheet final (curva de capital, tabelas, drawdown)
	$(PYTHON) scripts/09_strategy_report.py --config $(CONFIG)

regression:  ## 10: regressão controlada da distância de tom (método Angelo, winsorizado)
	$(PYTHON) scripts/10_controlled_regression.py --config $(CONFIG)

classify:  ## 11: exporta a classificação (papéis + distância) de TODAS as calls
	$(PYTHON) scripts/11_export_classification.py --config $(CONFIG)

all: download roles score features controls event-study backtest robustness decay report regression classify  ## Pipeline completo (exceto diagnostics)
