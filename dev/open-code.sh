#!/usr/bin/env bash
set -euo pipefail
# Start an agent if none, then add your GitHub key (prompts once)
if ! ssh-add -l >/dev/null 2>&1; then
  eval "$(ssh-agent -s)" >/dev/null
  ssh-add "$HOME/.ssh/id_ed25519_github" </dev/tty
fi
# open VS Code with this environment (SSH_AUTH_SOCK inherited)
exec code .
