#!/bin/bash
# MCP server entry point. Resolves its own location, so it works wherever the
# repo lives — no hardcoded paths. Assumes `uv sync` has been run.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export KMP_DUPLICATE_LIB_OK=TRUE

if [ ! -x "$SCRIPT_DIR/.venv/bin/semantic-image-search" ]; then
    echo "venv missing or incomplete. Run: uv sync" >&2
    exit 1
fi

exec "$SCRIPT_DIR/.venv/bin/semantic-image-search"
