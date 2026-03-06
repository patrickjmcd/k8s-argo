#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

chmod +x scripts/security-commit-agent.sh
chmod +x .githooks/pre-commit

git config core.hooksPath .githooks

echo "security-agent: installed"
echo "security-agent: git hooks path set to .githooks"

