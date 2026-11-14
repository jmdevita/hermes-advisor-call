#!/bin/bash
# Load this checkout into a real Hermes, in a throwaway HERMES_HOME, and run
# Hermes' own plugin checks. No network, no API key, no real config touched.
# Usage: scripts/hermes-smoke.sh [path-to-hermes-binary]
# Default binary: ../hermes-agent-ref/.venv/bin/hermes (set up by the cloud
# environment's setup script).
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
hermes="${1:-$repo/../hermes-agent-ref/.venv/bin/hermes}"
scratch="$(mktemp -d -t advisor-call-smoke.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
mkdir -p "$scratch/plugins"
ln -s "$repo" "$scratch/plugins/advisor-call"
export HERMES_HOME="$scratch"
v="$("$hermes" --version)"; echo "${v%%$'\n'*}"
"$hermes" plugins enable advisor-call >/dev/null
"$hermes" plugins validate "$repo"
"$hermes" plugins doctor "$repo"
"$hermes" plugins list 2>/dev/null | grep -E 'advisor-call|Name' || { echo "advisor-call not listed"; exit 1; }
