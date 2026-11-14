#!/bin/bash
# Run the same checks as .github/workflows/ci.yml locally, on Python 3.11 and 3.12.
# Needs uv. Usage: scripts/ci.sh
set -euo pipefail
cd "$(dirname "$0")/.."
for py in 3.11 3.12; do
  echo "== Python $py"
  uv run -q --isolated --no-project -p "$py" --with "pytest>=8" --with "ruff>=0.6" \
    bash -c 'ruff check . && ruff format --check . && python -m pytest -q'
done
echo "== all green"
