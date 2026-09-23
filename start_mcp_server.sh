#!/bin/bash
# MCP server entry point. Resolves its own location, so it works wherever the
# repo lives — no hardcoded paths. Assumes `uv sync` has been run.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export KMP_DUPLICATE_LIB_OK=TRUE
# Load models from the local HuggingFace cache only (no network check on
# every start; a timeout there broke the weekly reindex in Sep 2026).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ ! -x "$SCRIPT_DIR/.venv/bin/semantic-image-search" ]; then
    echo "venv missing or incomplete. Run: uv sync" >&2
    exit 1
fi

exec "$SCRIPT_DIR/.venv/bin/semantic-image-search"
