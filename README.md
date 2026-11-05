# C Code Grader

Rule-driven grading for C assignments. It combines Docker-sandboxed analyzers (compiler, cppcheck, Valgrind, ASan/UBSan, tests) with a local LLM (`llama-server`, default model `gguf/Qwen3.5-2B-GGUF:Q4_K_M`) and produces a score out of 100 and an evidence-backed report.

## Quick start (Linux)

Run these as your normal user, **not** with `sudo`:

```bash
git clone <repo-url> && cd Code-Grader
make
```

Enter your password once. `make` then installs and starts everything, and the UI opens at http://127.0.0.1:7860:

| Component | How it is installed |
|---|---|
| Python 3.12 + deps | `uv` → `.venv/` |
| Docker | get.docker.com, user added to `docker` group |
| Ollama | ollama.com install script (systemd service) |
| llama-server | Homebrew bottle, prebuilt (no compiling) → `/home/linuxbrew/.linuxbrew/bin` |
| Sandbox image | `docker build` → `agent-sandbox:latest` |
| Model | downloaded once (~1.3 GB) into the llama.cpp cache |

Supports apt (Debian/Ubuntu) and dnf (Fedora) on x86_64 and arm64. Log out and back in once so the `docker` group membership takes effect permanently.

## Commands

```bash
make              # full setup + launch UI
make setup        # setup only
make run          # launch UI (opens browser)
make check        # verify llama-server is compatible (needs build b12005+)
make sandbox      # rebuild Docker sandbox after editing Dockerfile.sandbox
make test         # run the test suite
make clean        # remove Python caches
```

You can override settings on the command line, for example `make run GRADIO_PORT=8000 LLAMA_MODEL=<hf-repo>:<quant>`.

Without `make`:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m grading_agent.ui.app          # Gradio UI
PYTHONPATH=src python -m grading_agent.main            # CLI, grades ./rules.json
python grade.py --rules path/to/rules.json [--source file.c] [--refresh-suite] [--json-only]
```

## Configuration

Settings are read from environment variables or `.env`, which `make` creates. See [settings.py](src/grading_agent/config/settings.py) for the full list. Common ones:

| Variable | Default |
|---|---|
| `LLAMA_MODEL` | `gguf/Qwen3.5-2B-GGUF:Q4_K_M` |
| `LLAMA_SERVER` | `/home/linuxbrew/.linuxbrew/bin/llama-server` |
| `LLAMA_PORT` / `GRADIO_PORT` | `8080` / `7860` |
| `LLAMA_CONTEXT_SIZE` | `8192` |
| `DOCKER_SANDBOX_IMAGE` | `agent-sandbox:latest` |
| `LLM_REVIEW_MAX_ITERATIONS` | `3` (RSI passes) |
| `LLAMA_CACHE_PROMPT` | `false`, which keeps grades reproducible |

## Using the UI

1. Upload the C submission and a `rules.json` (assignment statement, `problem_contract`, metric weights summing to `1.0`, optional `testset_path` / `reference_solution`).
2. **Generate Test Cases**: the LLM proposes inputs, and the sandbox computes the expected outputs from the reference solution. Review them.
3. **Grade Submission**: grading runs against the saved suite.

Example assignments with rules live in [examples/assignments/](examples/assignments/). The test-case file format is:

```json
[{"id": "basic-1", "input": "1 2\n", "expected": "3\n", "description": "Basic valid input"}]
```

## How grading works

1. Deterministic analyzers collect compiler, static-analysis, test, sanitizer, Valgrind, complexity, readability and similarity evidence.
2. The LLM describes what the program actually does. RSI passes then find defects, each citing evidence, and stop early once the findings converge.
3. Findings are deduplicated and adjudicated. The LLM rates (0–10) only the metrics that no test or analyzer measured, and executed tests and completed analyzers stay authoritative.
4. Requests run at temperature 0 with a fixed seed. Structured JSON stages are retried with bounded corrections. If the LLM fails, the report says so and falls back to an evidence-based score.

## Troubleshooting

- **`Incompatible llama.cpp build`**: the app needs llama.cpp b12005 or newer (PR #31380). Run `brew upgrade llama.cpp`.
- **Docker permission denied**: log out and back in, or run `sudo setfacl -m u:$USER:rw /var/run/docker.sock`.
- **Model download slow on first run**: rerun `make setup` so the model is cached before launching.
