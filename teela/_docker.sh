# sourced by teela scripts — run docker as root, this user, or via sg.
dk() {
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    docker "$@"
    return
  fi
  if command -v sg >/dev/null 2>&1 && getent group docker | grep -qw "${USER:-roni}"; then
    local tmp rc
    tmp="$(mktemp "${TMPDIR:-/tmp}/teela-dk.XXXXXX")"
    {
      printf '#!/bin/bash\nexec docker'
      local a
      for a in "$@"; do
        printf ' %q' "$a"
      done
      printf '\n'
    } >"$tmp"
    chmod +x "$tmp"
    sg docker "$tmp"
    rc=$?
    rm -f "$tmp"
    return "$rc"
  fi
  sudo docker "$@"
}
