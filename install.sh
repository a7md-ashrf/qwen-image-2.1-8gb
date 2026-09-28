#!/usr/bin/env bash
# Bootstrap: find a Python 3.10+, expose the `qwen21` command, then install.
#
#   ./install.sh                          # auto-detect the profile
#   ./install.sh --profile mac-8gb        # force a profile
#   ./install.sh --profile nvidia-8gb --skip-models
#   PYTHON=/usr/bin/python3.12 ./install.sh
set -euo pipefail

cd "$(dirname "$0")"

pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    echo "$PYTHON"
    return
  fi
  for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
        echo "$candidate"
        return
      fi
    fi
  done
  echo ""
}

PY="$(pick_python)"
if [[ -z "$PY" ]]; then
  cat >&2 <<'EOF'
error: no Python 3.10+ found on PATH.

  macOS:   brew install python@3.12
  Ubuntu:  sudo apt install python3.12 python3.12-venv
  Windows: install Python 3.12 from python.org (tick "add to PATH"), then re-run
EOF
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "error: git is required (ComfyUI submodule + ComfyUI-GGUF)." >&2
  exit 1
fi

echo "python: $PY ($("$PY" -c 'import platform;print(platform.python_version())'))"

if [[ "${1:-}" != "--no-cli" ]]; then
  "$PY" -m pip install --user --disable-pip-version-check -e . >/dev/null 2>&1 \
    && echo "installed the 'qwen21' command" \
    || echo "note: could not install the 'qwen21' command; use '$PY -m qwen21 ...' instead"
fi
if [[ "${1:-}" == "--no-cli" ]]; then
  shift
fi

exec "$PY" -m qwen21 install "$@"
