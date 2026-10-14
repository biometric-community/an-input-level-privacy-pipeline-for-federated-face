#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
REPO_ROOT="$ROOT"
while [[ "$REPO_ROOT" != "/" ]]; do
  [[ -d "$REPO_ROOT/.venv" ]] && break
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
source "${TBIOT_SHARED_VENV:-$REPO_ROOT/.venv}/bin/activate"
python -m pip install -q -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
