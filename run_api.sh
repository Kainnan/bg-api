#!/usr/bin/env bash
# Arranca a API com o Python do .venv (evita uvicorn global sem dependências).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Erro: .venv não existe. Corre primeiro: ./setup_runpod.sh"
  exit 1
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
exec .venv/bin/python -m uvicorn app.main:app --host "$HOST" --port "$PORT"
