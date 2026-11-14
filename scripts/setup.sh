#!/bin/bash
# Prepare a machine for Resonance with one command:
#
#   bash scripts/setup.sh
#
# It asks for a HuggingFace token (unless one is already stored or HF_TOKEN is set), installs aria2 with apt,
# installs Resonance and its dependencies with `pip install .`, stores the token, runs the preflight checks
# and downloads every model the default pipeline uses.
# The install is not editable: run this script (or `pip install .`) again after pulling new code.
# PYTHON picks the interpreter; the default is /venv/main/bin/python when it exists (vast.ai images), else python3.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-$([ -x /venv/main/bin/python ] && echo /venv/main/bin/python || command -v python3)}
TOKEN_FILE=${HF_HOME:-$HOME/.cache/huggingface}/token
echo "Using $PYTHON ($($PYTHON --version 2>&1))"

if [ -z "${HF_TOKEN:-}" ] && [ ! -s "$TOKEN_FILE" ]; then
    echo "The pipeline downloads gated models (pyannote/segmentation-3.0). Accept their terms on huggingface.co"
    read -r -s -p "and paste a HuggingFace token: " HF_TOKEN
    echo
fi

if ! command -v aria2c > /dev/null; then
    if command -v apt-get > /dev/null; then
        echo "Installing aria2"
        $([ "$(id -u)" -eq 0 ] || echo sudo) apt-get install -y aria2 > /dev/null
    else
        echo "aria2c is missing and apt is not available: install aria2 yourself (e.g. brew install aria2)"
    fi
fi

echo "Installing Resonance and its dependencies"
"$PYTHON" -m pip install --quiet .

if [ -n "${HF_TOKEN:-}" ] && [ ! -s "$TOKEN_FILE" ]; then
    # Written by Python from the environment, so the token never appears on a command line.
    HF_TOKEN="$HF_TOKEN" TOKEN_FILE="$TOKEN_FILE" "$PYTHON" - <<'PY'
import os, pathlib
path = pathlib.Path(os.environ["TOKEN_FILE"])
path.parent.mkdir(parents=True, exist_ok=True)
path.touch(mode=0o600)
path.write_text(os.environ["HF_TOKEN"].strip())
PY
    echo "Stored the HuggingFace token in $TOKEN_FILE"
fi

"$PYTHON" scripts/preflight.py --download-models && echo "Setup complete." || echo "Setup finished, but the checks above need attention."
