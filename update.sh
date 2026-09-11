#!/usr/bin/env bash
# Reset this checkout to GitHub main. Does not touch ~/.grok or cluster tokens.
set -euo pipefail
cd "$(dirname "$0")"
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git checkout. Clone once:"
  echo "  git clone https://github.com/colornosteela-cloud/Grok-desk.git"
  exit 1
fi
git remote set-url origin https://github.com/colornosteela-cloud/Grok-desk.git
git fetch origin
git checkout -B main origin/main
git reset --hard origin/main
echo "Updated to $(git log -1 --oneline)"
echo "Restart deskd if it is running: stop it, then ./start.sh"
echo "Cluster token stays in ~/.grok/desk.json — do not delete that."
