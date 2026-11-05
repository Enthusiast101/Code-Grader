# ============================================================
# C Code Grader — one-command Linux setup and launch
#
#   make            install everything (asks for sudo password once) and open the UI
#   make setup      install everything, do not launch
#   make run        launch the UI (after setup)
#
# Run as your normal user, NOT with sudo. Supports apt (Debian/Ubuntu)
# and dnf (Fedora/RHEL) distributions on x86_64 and arm64.
# ============================================================

SHELL := /bin/bash
.DEFAULT_GOAL := all
.NOTPARALLEL:

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PYTHON_VERSION ?= 3.12
VENV           ?= .venv
VENV_PYTHON    := $(VENV)/bin/python
UV             := $(HOME)/.local/bin/uv

BREW_PREFIX    := /home/linuxbrew/.linuxbrew
BREW           := $(BREW_PREFIX)/bin/brew
LLAMA_SERVER   ?= $(BREW_PREFIX)/bin/llama-server

# ------------------------------------------------------------
# Runtime configuration (read by src/grading_agent/config/settings.py)
# ------------------------------------------------------------

LLAMA_MODEL          ?= gguf/Qwen3.5-2B-GGUF:Q4_K_M
LLAMA_HOST           ?= 127.0.0.1
LLAMA_PORT           ?= 8080
DOCKER_SANDBOX_IMAGE ?= agent-sandbox:latest
GRADIO_HOST          ?= 127.0.0.1
GRADIO_PORT          ?= 7860
UI_URL               := http://$(GRADIO_HOST):$(GRADIO_PORT)

export LLAMA_SERVER LLAMA_MODEL LLAMA_HOST LLAMA_PORT
export DOCKER_SANDBOX_IMAGE GRADIO_HOST GRADIO_PORT
export PATH := $(HOME)/.local/bin:$(BREW_PREFIX)/bin:$(PATH)

.PHONY: all setup run help sudo system-deps python docker ollama llama-server \
        env sandbox model check test clean

all: setup run

setup: sudo system-deps python docker ollama llama-server env sandbox model check
	@echo ""
	@echo "Setup complete. Start the UI any time with: make run"

help:
	@sed -n '2,10p' Makefile

# ------------------------------------------------------------
# sudo: ask for the password once and keep it alive while make runs
# ------------------------------------------------------------

sudo:
	@if [ "$$(id -u)" -eq 0 ]; then \
		echo "ERROR: run 'make' as your normal user, not root/sudo."; exit 1; fi
	@echo "Administrator password is needed to install system packages."
	@sudo -v
	@( while kill -0 $$PPID 2>/dev/null; do sudo -n true; sleep 50; done ) >/dev/null 2>&1 &

# ------------------------------------------------------------
# Base OS packages
# ------------------------------------------------------------

system-deps:
	@echo "==> Installing base system packages"
	@if command -v apt-get >/dev/null; then \
		sudo apt-get update -y && \
		sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
			curl ca-certificates git build-essential procps file acl xdg-utils; \
	elif command -v dnf >/dev/null; then \
		sudo dnf install -y curl ca-certificates git gcc gcc-c++ make procps-ng file acl xdg-utils; \
	else \
		echo "ERROR: unsupported distribution (need apt-get or dnf)."; exit 1; \
	fi

# ------------------------------------------------------------
# Python (via uv, so any distro gets Python $(PYTHON_VERSION))
# ------------------------------------------------------------

python:
	@echo "==> Installing Python $(PYTHON_VERSION) and project dependencies"
	@test -x "$(UV)" || curl -LsSf https://astral.sh/uv/install.sh | sh
	@test -x "$(VENV_PYTHON)" || "$(UV)" venv --python "$(PYTHON_VERSION)" "$(VENV)"
	@"$(UV)" pip install --python "$(VENV_PYTHON)" -r requirements.txt -e ".[all]" pytest

# ------------------------------------------------------------
# Docker (official convenience script)
# ------------------------------------------------------------

docker:
	@echo "==> Installing Docker"
	@command -v docker >/dev/null || curl -fsSL https://get.docker.com | sudo sh
	@sudo systemctl enable --now docker
	@id -nG "$$USER" | grep -qw docker || sudo usermod -aG docker "$$USER"
	@# Group membership applies after the next login; grant socket access now.
	@docker info >/dev/null 2>&1 || sudo setfacl -m "u:$$USER:rw" /var/run/docker.sock

# ------------------------------------------------------------
# Ollama (official install script; runs as a systemd service)
# ------------------------------------------------------------

ollama:
	@echo "==> Installing Ollama"
	@command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh

# ------------------------------------------------------------
# llama-server: prebuilt Homebrew bottle in the default prefix
# ($(BREW_PREFIX)), the official llama.cpp Linux install. No compiling.
# ------------------------------------------------------------

llama-server:
	@echo "==> Installing llama.cpp (llama-server)"
	@test -x "$(BREW)" || NONINTERACTIVE=1 /bin/bash -c \
		"$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
	@grep -qs 'linuxbrew/bin/brew shellenv' "$(HOME)/.bashrc" || \
		echo 'eval "$$($(BREW) shellenv)"' >> "$(HOME)/.bashrc"
	@if "$(BREW)" list --formula llama.cpp >/dev/null 2>&1; then \
		"$(BREW)" upgrade llama.cpp || true; \
	else \
		"$(BREW)" install llama.cpp; \
	fi
	@"$(LLAMA_SERVER)" --version

# ------------------------------------------------------------
# .env so `python -m grading_agent...` works outside make too
# ------------------------------------------------------------

env:
	@touch .env
	@for kv in "LLAMA_SERVER=$(LLAMA_SERVER)" "LLAMA_MODEL=$(LLAMA_MODEL)" \
	           "DOCKER_SANDBOX_IMAGE=$(DOCKER_SANDBOX_IMAGE)"; do \
		grep -q "^$${kv%%=*}=" .env || echo "$$kv" >> .env; \
	done

# ------------------------------------------------------------
# Docker sandbox image used to compile/run submissions
# ------------------------------------------------------------

sandbox:
	@echo "==> Building sandbox image $(DOCKER_SANDBOX_IMAGE)"
	@docker build -t "$(DOCKER_SANDBOX_IMAGE)" -f Dockerfile.sandbox .

# ------------------------------------------------------------
# Download the GGUF model once so the first UI run does not time out
# ------------------------------------------------------------

model:
	@echo "==> Downloading model $(LLAMA_MODEL) (first time ~1.3 GB)"
	@"$(LLAMA_SERVER)" -hf "$(LLAMA_MODEL)" --host 127.0.0.1 --port 18080 --ctx-size 512 \
		>/tmp/code-grader-model.log 2>&1 & pid=$$!; \
	for i in $$(seq 1 1800); do \
		curl -fs -o /dev/null http://127.0.0.1:18080/health && { kill $$pid; wait $$pid 2>/dev/null; echo "Model ready."; exit 0; }; \
		kill -0 $$pid 2>/dev/null || { echo "ERROR: llama-server failed, see /tmp/code-grader-model.log"; exit 1; }; \
		sleep 1; \
	done; kill $$pid; echo "ERROR: model download timed out"; exit 1

# ------------------------------------------------------------
# Verify llama-server passes the app's JSON-schema compatibility check
# ------------------------------------------------------------

check:
	@PYTHONPATH=src "$(VENV_PYTHON)" -c \
		"from grading_agent.llm.server import _verify_json_schema_compatible_server as v; v('$(LLAMA_SERVER)'); print('llama-server OK')"

# ------------------------------------------------------------
# Launch the Gradio UI and open it in the browser
# ------------------------------------------------------------

run:
	@docker info >/dev/null 2>&1 || sudo setfacl -m "u:$$USER:rw" /var/run/docker.sock
	@( for i in $$(seq 1 180); do \
		curl -fs -o /dev/null "$(UI_URL)" && { xdg-open "$(UI_URL)" >/dev/null 2>&1 || true; exit 0; }; \
		sleep 1; done ) &
	@echo "==> Starting UI at $(UI_URL) (Ctrl+C to stop)"
	@PYTHONPATH=src "$(VENV_PYTHON)" -m grading_agent.ui.app

# ------------------------------------------------------------
# Development
# ------------------------------------------------------------

test:
	@PYTHONPATH=src "$(VENV_PYTHON)" -m pytest -q tests

clean:
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "Python caches removed."
