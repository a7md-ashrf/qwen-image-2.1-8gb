#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./install.sh /path/to/ComfyUI [--force]"
  exit 2
fi

COMFY="$1"
shift || true
python3 qwen21.py install --comfy "$COMFY" "$@"
