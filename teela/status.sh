#!/usr/bin/env bash
set -euo pipefail
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-teela-vllm}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_docker.sh
source "$HERE/_docker.sh"

echo "=== container ==="
dk ps -a --filter "name=^/${CONTAINER_NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' || true
echo
echo "=== /v1/models ==="
if curl -fsS "http://${HOST}:${PORT}/v1/models"; then
  echo
else
  echo "API not responding on http://${HOST}:${PORT}/v1/models"
  echo
  echo "=== recent logs ==="
  dk logs --tail 40 "$CONTAINER_NAME" 2>/dev/null || true
  exit 1
fi
