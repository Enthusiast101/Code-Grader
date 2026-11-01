#!/usr/bin/env bash
set -euo pipefail

# Update llama.cpp to a revision containing PR #31380 and build llama-server.
# Optional environment variables:
#   LLAMA_CPP_DIR  Existing clone location (default: ~/llama.cpp)
#   GGML_CUDA      ON for CUDA, OFF for CPU-only (default: OFF)
#   LLAMA_CPP_REF  llama.cpp tag/branch (default: b12005, which contains PR #31380)

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
GGML_CUDA="${GGML_CUDA:-OFF}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-b12005}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
fi

git -C "$LLAMA_CPP_DIR" fetch origin --tags
git -C "$LLAMA_CPP_DIR" checkout "$LLAMA_CPP_REF"
if git -C "$LLAMA_CPP_DIR" show-ref --verify --quiet "refs/heads/$LLAMA_CPP_REF"; then
  git -C "$LLAMA_CPP_DIR" pull --ff-only origin "$LLAMA_CPP_REF"
fi

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA="$GGML_CUDA"
cmake --build "$LLAMA_CPP_DIR/build" --config Release --parallel --target llama-server

mkdir -p "$INSTALL_DIR"
install -m 0755 "$LLAMA_CPP_DIR/build/bin/llama-server" "$INSTALL_DIR/llama-server"
echo "Installed: $INSTALL_DIR/llama-server"
"$INSTALL_DIR/llama-server" --version
