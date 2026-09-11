#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
MODELS_DIR="${MODELS_DIR:-/home/roni/models}"
DEST="${MUSE_MODEL:-$MODELS_DIR/Muse-Glimmer-30B}"
mkdir -p "$MODELS_DIR"
echo "[teela] downloading meta-models/Muse-Glimmer-30B -> $DEST"
hf download meta-models/Muse-Glimmer-30B --local-dir "$DEST"
test -f "$DEST/model.safetensors.index.json"
test -f "$DEST/chat_template.jinja"
du -sh "$DEST"
echo "[teela] muse weights ready"
