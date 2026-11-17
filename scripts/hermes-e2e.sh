#!/bin/bash
# End-to-end check in a real Hermes, against a local fake Anthropic server.
# No network, no API key, no spend; a throwaway HERMES_HOME.
# Usage: scripts/hermes-e2e.sh [path-to-hermes-python]
# Env: E2E_MODEL=sonnet|opus, E2E_EFFORT=<advisor.effort> (default: unset)
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
python="${1:-$repo/../hermes-agent-ref/.venv/bin/python}"
port="${E2E_PORT:-18765}"
model="${E2E_MODEL:-sonnet}"   # sonnet or opus
effort="${E2E_EFFORT:-}"       # empty, or low | medium | high | xhigh | max
scratch="$(mktemp -d -t advisor-call-e2e.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
mkdir -p "$scratch/plugins"
ln -s "$repo" "$scratch/plugins/advisor-call"
cat > "$scratch/config.yaml" <<YAML
approvals:
  mode: "off"   # no human here; the plugin's own approve directive is checked directly
plugins:
  enabled: [advisor-call]
  entries:
    advisor-call:
      budget: {cycle_start_day: 3}
      advisor: {model: $model${effort:+, effort: $effort}}
      approval: {refuse_when_approvals_off: false}
auxiliary:
  advisor_call_$model:
    base_url: http://127.0.0.1:$port
YAML
cd "$scratch"
HERMES_HOME="$scratch" ANTHROPIC_API_KEY="local-placeholder" \
  "$python" -I "$repo/tests/e2e/hermes_e2e.py" "$port" "$model" "$effort"
