#!/usr/bin/env bash
set -euo pipefail
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
BASE="http://${HOST}:${PORT}/v1"

MODEL="$(curl -fsS "${BASE}/models" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["data"][0]["id"])')"
echo "served model: $MODEL"

curl -fsS "${BASE}/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(python3 - <<PY
import json
print(json.dumps({
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
  "max_tokens": 128,
  "temperature": 0,
}))
PY
)"
echo
