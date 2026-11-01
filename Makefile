
# ============================================================
# Low-memory Makefile — Qwen3.5-2B GGUF + Gradio + Docker
# ============================================================

SHELL := /bin/bash

.DEFAULT_GOAL := help

# ------------------------------------------------------------
# Paths and environment
# ------------------------------------------------------------

VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

LLAMA_SERVER ?= $(HOME)/.local/bin/llama-server
LLAMA_MODEL ?= gguf/Qwen3.5-2B-GGUF:Q4_K_M
LLAMA_HOST ?= 127.0.0.1
LLAMA_PORT ?= 8080

# ------------------------------------------------------------
# Memory-conscious llama.cpp configuration
# ------------------------------------------------------------

LLAMA_CONTEXT_SIZE ?= 2048
LLAMA_PARALLEL_SLOTS ?= 1

LLAMA_THREADS ?= 2
LLAMA_THREADS_BATCH ?= 1

LLAMA_BATCH_SIZE ?= 64
LLAMA_UBATCH_SIZE ?= 32

LLAMA_MAX_TOKENS ?= 1024

# ------------------------------------------------------------
# Grading configuration
# ------------------------------------------------------------

LLM_REASONING_ENABLED ?= true
LLM_REASONING_FORMAT ?= deepseek
LLM_REASONING_BUDGET ?= 128

LLM_REVIEW_MAX_ITERATIONS ?= 3

LLM_REVIEW_MAX_TOKENS ?= 384
LLM_RSI_MAX_TOKENS ?= 768
LLM_SOURCE_AUDIT_MAX_TOKENS ?= 512
LLM_UNDERSTANDING_MAX_TOKENS ?= 512
LLM_GLOBAL_RSI_MAX_TOKENS ?= 768
LLM_GLOBAL_SCORE_MAX_TOKENS ?= 512

LLM_SOURCE_MAX_CHARS ?= 4000
LLM_MAX_FINDINGS ?= 6

# ------------------------------------------------------------
# Context compression
# ------------------------------------------------------------

LLM_CONTEXT_COMPRESSION_ENABLED ?= true
LLM_CONTEXT_COMPRESSION_THRESHOLD ?= 1200
LLM_CONTEXT_COMPRESSION_MAX_TOKENS ?= 256

# ------------------------------------------------------------
# Gradio limits
# ------------------------------------------------------------

GRADIO_CONCURRENCY_LIMIT ?= 1
GRADIO_MAX_THREADS ?= 1
GRADIO_QUEUE_SIZE ?= 1

# ------------------------------------------------------------
# Build limits
# ------------------------------------------------------------

BUILD_JOBS ?= 1

# ------------------------------------------------------------
# Docker configuration
# ------------------------------------------------------------

DOCKER_IMAGE ?= grading-sandbox:latest
DOCKERFILE ?= Dockerfile.sandbox

# ------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------

# export OMP_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export BLIS_NUM_THREADS=1
# export VECLIB_MAXIMUM_THREADS=1

# export TOKENIZERS_PARALLELISM=false
# export PIP_NO_CACHE_DIR=1

export CMAKE_BUILD_PARALLEL_LEVEL=$(BUILD_JOBS)
export MAX_JOBS=$(BUILD_JOBS)
export MAKEFLAGS=-j$(BUILD_JOBS)

# ------------------------------------------------------------
# Export grading and model configuration
# ------------------------------------------------------------

export LLAMA_SERVER
export LLAMA_MODEL
export LLAMA_HOST
export LLAMA_PORT

export LLAMA_CONTEXT_SIZE
export LLAMA_PARALLEL_SLOTS
export LLAMA_THREADS
export LLAMA_THREADS_BATCH
export LLAMA_BATCH_SIZE
export LLAMA_UBATCH_SIZE
export LLAMA_MAX_TOKENS

export LLM_REASONING_ENABLED
export LLM_REASONING_FORMAT
export LLM_REASONING_BUDGET

export LLM_REVIEW_MAX_ITERATIONS
export LLM_REVIEW_MAX_TOKENS
export LLM_RSI_MAX_TOKENS
export LLM_SOURCE_AUDIT_MAX_TOKENS
export LLM_UNDERSTANDING_MAX_TOKENS
export LLM_GLOBAL_RSI_MAX_TOKENS
export LLM_GLOBAL_SCORE_MAX_TOKENS

export LLM_SOURCE_MAX_CHARS
export LLM_MAX_FINDINGS

export LLM_CONTEXT_COMPRESSION_ENABLED
export LLM_CONTEXT_COMPRESSION_THRESHOLD
export LLM_CONTEXT_COMPRESSION_MAX_TOKENS

export GRADIO_CONCURRENCY_LIMIT
export GRADIO_MAX_THREADS
export GRADIO_QUEUE_SIZE

# ------------------------------------------------------------
# Help
# ------------------------------------------------------------

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make setup        Install dependencies and build sandbox"
	@echo "  make install      Install Python dependencies"
	@echo "  make sandbox      Build Docker sandbox image"
	@echo "  make llama-check  Validate llama-server version"
	@echo "  make ui           Start Gradio application"
	@echo "  make run          Start application"
	@echo "  make clean        Remove generated caches"
	@echo "  make print-config Show memory configuration"

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

.PHONY: print-config
print-config:
	@echo "========== LOW MEMORY CONFIGURATION =========="
	@echo "Model:                 $(LLAMA_MODEL)"
	@echo "Context:               $(LLAMA_CONTEXT_SIZE)"
	@echo "Parallel slots:        $(LLAMA_PARALLEL_SLOTS)"
	@echo "Threads:               $(LLAMA_THREADS)"
	@echo "Batch threads:         $(LLAMA_THREADS_BATCH)"
	@echo "Batch size:            $(LLAMA_BATCH_SIZE)"
	@echo "Micro batch size:      $(LLAMA_UBATCH_SIZE)"
	@echo "Max tokens:            $(LLAMA_MAX_TOKENS)"
	@echo "Reasoning budget:      $(LLM_REASONING_BUDGET)"
	@echo "Gradio concurrency:    $(GRADIO_CONCURRENCY_LIMIT)"
	@echo "Gradio threads:        $(GRADIO_MAX_THREADS)"
	@echo "Build jobs:            $(BUILD_JOBS)"
	@echo "=============================================="

# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------

.PHONY: install
install:
	@test -x "$(VENV_PYTHON)" || python3 -m venv "$(VENV)"
	@$(VENV_PYTHON) -m pip install --upgrade pip
	@$(VENV_PYTHON) -m pip install -r requirements.txt

# ------------------------------------------------------------
# Docker sandbox
# ------------------------------------------------------------

.PHONY: sandbox
sandbox:
	@docker build \
		--memory=2g \
		--memory-swap=2g \
		--cpu-period=100000 \
		--cpu-quota=100000 \
		-t "$(DOCKER_IMAGE)" \
		-f "$(DOCKERFILE)" .

# ------------------------------------------------------------
# llama.cpp validation
# ------------------------------------------------------------

.PHONY: llama-check
llama-check:
	@test -x "$(LLAMA_SERVER)" || { \
		echo "ERROR: llama-server not found: $(LLAMA_SERVER)"; \
		exit 1; \
	}
	@echo "Checking llama-server..."
	@$(LLAMA_SERVER) --version

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

.PHONY: setup
setup: install sandbox
	@echo "Setup complete."

# ------------------------------------------------------------
# Launch Gradio
# ------------------------------------------------------------

.PHONY: ui
ui: llama-check install
	@if ! docker image inspect "$(DOCKER_IMAGE)" >/dev/null 2>&1; then \
		echo "Sandbox image missing; building it once..."; \
		$(MAKE) sandbox; \
	fi
	@echo "Starting Gradio with low-memory settings..."
	@$(VENV_PYTHON) -m grading_agent.ui.app

.PHONY: run
run: ui

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------

.PHONY: clean
clean:
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "Python caches removed."
