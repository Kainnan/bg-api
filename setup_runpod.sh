#!/usr/bin/env bash
# Instalação após git pull — pensado para RunPod (Linux) ou VM similar.
# Uso: chmod +x setup_runpod.sh && ./setup_runpod.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> [1/4] git pull"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git pull --rebase --autostash || git pull
else
  echo "    Aviso: pasta não é um repositório git; a saltar git pull."
fi

echo "==> [2/4] venv (.venv)"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

echo "==> [3/4] pip + requirements.txt"
python -m pip install --upgrade pip wheel
# Instala com índice extra do PyTorch CUDA 12.4 (compatível com drivers 12.x).
# Sem isso, pip pega torch >= 2.10 que exige CUDA 13 e falha em GPUs mais antigas.
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
# Obrigatório para FastAPI com UploadFile/Form (evita erro se algo falhar no freeze)
pip install "python-multipart>=0.0.9"

echo "==> [4/4] concluído"
echo ""
echo "Arrancar a API (usa SEMPRE o venv — não uses o uvicorn do sistema):"
echo "  ./run_api.sh"
echo "  # ou:"
echo "  .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "Na RunPod, expõe a porta 8000 no template / mapeamento HTTP."
