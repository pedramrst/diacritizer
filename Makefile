# High-level commands for setup, training, sweeping, and inference.
# No Docker -- see README.md for why (Vast already provides a CUDA-ready
# image, and this repo's dependency stack is a single Python framework).
#
# Override any variable on the command line, e.g.:
#   make train CONFIG=configs/lr_5e-5.yaml ARGS="--epochs 3 --push_to_hub"

.DEFAULT_GOAL := help
.ONESHELL:
SHELL := /bin/bash

# Picked up if present -- e.g. an uncommented TB_PORT=... in .env persistently
# overrides the default below without typing TB_PORT=... on every invocation.
# "-include" (not "include") so a missing .env before `make setup` isn't an error.
-include .env

VENV        := venv
PYTHON      := $(VENV)/bin/python3
TENSORBOARD := $(VENV)/bin/tensorboard

CONFIG      ?= config.yaml
ARGS        ?=

CONFIGS_DIR ?= configs
OUT_ROOT    ?= sweeps/latest

TB_LOGDIR   ?= runs
TB_PORT     ?= 6006

MODEL       ?= ./canine-fa-diacritizer
TEXT        ?=

REPORT_OUT       ?= report.html
REPORT_METRICS   ?= metrics.json
REPORT_BENCHMARK ?= benchmark.json

.PHONY: help setup train sweep smoke-test diacritize report clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv, install pinned deps, install diacritizer/ as an editable package
	@python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .
	test -f .env || cp .env.example .env
	echo "Done. Edit .env with your real HF tokens before training."

train: ## Train + TensorBoard live alongside it (CONFIG=path.yaml ARGS="--epochs 3 ...")
	@mkdir -p $(TB_LOGDIR)
	$(TENSORBOARD) --logdir $(TB_LOGDIR) --port $(TB_PORT) --host 0.0.0.0 --reload_interval 5 &
	TB_PID=$$!
	trap "kill $$TB_PID 2>/dev/null" EXIT
	echo "[make train] TensorBoard: http://localhost:$(TB_PORT)  (logdir: $(TB_LOGDIR)/)"
	$(PYTHON) scripts/train.py --config $(CONFIG) $(ARGS)

sweep: ## Run scripts/sweep.py over CONFIGS_DIR + TensorBoard alongside (OUT_ROOT=... ARGS="--push_to_hub")
	@mkdir -p $(OUT_ROOT)/runs
	$(TENSORBOARD) --logdir $(OUT_ROOT)/runs --port $(TB_PORT) --host 0.0.0.0 --reload_interval 5 &
	TB_PID=$$!
	trap "kill $$TB_PID 2>/dev/null" EXIT
	echo "[make sweep] TensorBoard: http://localhost:$(TB_PORT)  (logdir: $(OUT_ROOT)/runs)"
	$(PYTHON) scripts/sweep.py --configs $(CONFIGS_DIR) --out_root $(OUT_ROOT) $(if $(ARGS),--extra_args "$(ARGS)")

smoke-test: ## Fast tiny-model sanity check of the full pipeline (no real weights downloaded)
	@$(PYTHON) scripts/smoke_test.py

diacritize: ## Run inference (MODEL=path-or-repo TEXT="...")
	@if [ -z "$(TEXT)" ]; then echo 'Usage: make diacritize MODEL=... TEXT="..."'; exit 1; fi
	$(PYTHON) scripts/diacritize.py --model $(MODEL) --text "$(TEXT)"

report: ## Evaluate + benchmark MODEL, then generate the manager report (REPORT_OUT=report.html)
	@echo "[1/3] evaluating accuracy on the test set ..."
	$(PYTHON) -m diacritizer.metrics --model $(MODEL) --split test --out $(REPORT_METRICS)
	@echo "[2/3] benchmarking speed / memory / device ..."
	$(PYTHON) scripts/benchmark.py --model $(MODEL) --out $(REPORT_BENCHMARK)
	@echo "[3/3] rendering the report ..."
	$(PYTHON) scripts/generate_report.py --metrics $(REPORT_METRICS) --benchmark $(REPORT_BENCHMARK) --out $(REPORT_OUT)
	echo "Report written to $(REPORT_OUT)"

clean: ## Remove __pycache__ and other safe, regenerable caches
	@find . -type d -name "__pycache__" -not -path "./$(VENV)/*" -exec rm -rf {} +
	rm -rf .pytest_cache resume_checkpoint
	echo "Note: checkpoints/, runs/, sweeps/ are left alone -- they're real trained-model" \
	     "output, not caches. Remove those manually if you actually want to."
