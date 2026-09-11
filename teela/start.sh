#!/usr/bin/env bash
# Thin wrapper around start.sh.real (Intel llm-scaler vLLM on 2× Arc Pro B60).
set -euo pipefail
exec bash "$(cd "$(dirname "$0")" && pwd)/start.sh.real" "$@"
