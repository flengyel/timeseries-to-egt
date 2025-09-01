#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${1:-timeseries-to-egt}"
VISIBILITY="${2:-public}"   # or private

if [[ ! -f "pyproject.toml" ]]; then
  echo "Run from project root (pyproject.toml not found)." >&2
  exit 1
fi

# Recommend LF line endings on Unix
git config core.autocrlf input || true

git init
git add .
git commit -m "Initial commit: ts2eg package, tests, demo, notebooks, math background"

# If gh CLI available, prefer it
if command -v gh >/dev/null 2>&1; then
  set +e
  gh repo create "$REPO_NAME" --source . --${VISIBILITY} --remote origin --push
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "Pushed to GitHub via gh CLI."
    exit 0
  else
    echo "gh repo create failed (maybe repo exists). Falling back to manual steps."
  fi
fi

echo "Manual steps:"
echo "1) Create an empty repo on GitHub named $REPO_NAME (Public/Private)."
echo "2) Then run:"
echo "   git remote add origin https://github.com/YOURUSER/${REPO_NAME}.git"
echo "   git branch -M main"
echo "   git push -u origin main"
