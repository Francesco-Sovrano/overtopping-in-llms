#!/usr/bin/env bash
set -euo pipefail

python3.12 -m venv .env

# shellcheck disable=SC1091
. .env/bin/activate

python -m pip install -U pip
python -m pip install -U setuptools wheel
python -m pip install -U -r requirements.txt

if command -v ollama >/dev/null 2>&1; then
  ollama pull gemma3:27b
  ollama pull qwen3:4b
  ollama pull qwen3:14b
else
  echo "Ollama is not installed; skipping default feature-model downloads."
fi
