#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(id -u)" -eq 0 ]]; then
  echo "Do not run grok-deskd as root. Use the same user that installed Grok Build (grok login)." >&2
  echo "The CLI is ~/.local/bin/grok or ~/.grok/bin/grok for that user — not /root/.local/bin/grok." >&2
  echo "If you must, set GROK_BIN=/path/to/grok and GROK_HOME=/home/<user>/.grok" >&2
  exit 1
fi
if [[ -z "${GROK_BIN:-}" ]]; then
  if command -v grok >/dev/null 2>&1; then
    GROK_BIN="$(command -v grok)"
  elif [[ -x "${HOME}/.grok/bin/grok" ]]; then
    GROK_BIN="${HOME}/.grok/bin/grok"
  elif [[ -x "${HOME}/.local/bin/grok" ]]; then
    GROK_BIN="${HOME}/.local/bin/grok"
  else
    GROK_BIN="${HOME}/.local/bin/grok"
  fi
fi
export GROK_BIN
export GROK_DESK_PORT="${GROK_DESK_PORT:-8742}"
export GROK_DESKS="${GROK_DESKS:-$HOME/grok-desks}"
export GROK_DESK_SANDBOX="${GROK_DESK_SANDBOX:-off}"
# Local model upstream (llama.cpp, vLLM, or a key proxy in front of them).
# Renamed from GROK_DESK_VLLM*; the old names are still honored by deskd.
export GROK_DESK_LLM="${GROK_DESK_LLM:-${GROK_DESK_VLLM:-http://127.0.0.1:8081}}"
export GROK_DESK_MODEL="${GROK_DESK_MODEL:-${GROK_DESK_VLLM_MODEL:-Qwen3.8-27B}}"
export GROK_DESK_MAX_LEN="${GROK_DESK_MAX_LEN:-${GROK_DESK_VLLM_MAX_LEN:-262144}}"
export GROK_DESK_MAX_TOKENS="${GROK_DESK_MAX_TOKENS:-${GROK_DESK_VLLM_MAX_TOKENS:-32768}}"

runtime="${XDG_RUNTIME_DIR:-/tmp}/grok-desk"
mkdir -p "$runtime"
pidfile="$runtime/deskd.pid"

deskd_already_running() {
  if [[ -f "$pidfile" ]]; then
    old="$(tr -d ' \n' < "$pidfile" 2>/dev/null || true)"
    if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
      cmd="$(tr '\0' ' ' < "/proc/${old}/cmdline" 2>/dev/null || true)"
      if [[ "$cmd" == *deskd.py* ]]; then
        echo "grok-deskd already running (pid ${old})"
        return 0
      fi
    fi
  fi
  if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -qE ":${GROK_DESK_PORT}[[:space:]]"; then
    echo "grok-deskd port ${GROK_DESK_PORT} already in use"
    return 0
  fi
  return 1
}

if deskd_already_running; then
  exit 0
fi

# Nested under a Grok Build agent, detach so the parent cannot SIGTERM our grok children.
if [[ -n "${GROK_AGENT:-}" && -z "${GROK_DESK_FOREGROUND:-}" ]] && command -v setsid >/dev/null 2>&1; then
  echo "grok-deskd detaching on http://127.0.0.1:${GROK_DESK_PORT}/  (log $runtime/deskd.log)"
  nohup setsid python3 deskd/deskd.py </dev/null >"$runtime/deskd.log" 2>&1 &
  echo "pid $!"
  exit 0
fi
exec python3 deskd/deskd.py
