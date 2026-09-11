#!/usr/bin/env bash
set -euo pipefail
CONTAINER_NAME="${CONTAINER_NAME:-teela-vllm}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_docker.sh
source "$HERE/_docker.sh"

stop_one() {
  local name="$1"
  if dk ps -a --format '{{.Names}}' | grep -qx "$name"; then
    echo "[teela] stopping $name"
    dk rm -f "$name" >/dev/null
    echo "[teela] stopped $name"
  fi
}

stop_one "$CONTAINER_NAME"
stop_one "${FAST_CONTAINER_NAME:-teela-vllm-fast}"
stop_one "${FAST_LEGACY_NAME:-teela-llama-9b}"
