#!/bin/bash
# Unattended incremental reindex of the photo library.
#
# Runs reindex_missing.py, which indexes only new/changed files (detected from
# size + mtime on the online-only placeholders, so it does NOT re-download the
# whole library) and rebuilds the FAISS index. Meant to be driven on a schedule
# by the launchd agent com.himalayantrust.photo-reindex.plist so the index stays
# current as new photos are added.
#
# NOTE on disk: this does not evict between files. New additions are downloaded
# and left local until you next "Free Up Space" on the library. That is fine for
# normal trickle. For a large new drop (tens of GB), run the attended
# index_library.py instead so eviction keeps peak disk in check.
set -euo pipefail

# faiss and torch can each pull in their own OpenMP runtime; on macOS that
# aborts with "OMP: Error #15" unless we allow the duplicate. Harmless here.
export KMP_DUPLICATE_LIB_OK=TRUE

# Load the CLIP model from the local HuggingFace cache only. Without this,
# transformers calls huggingface.co on every run to check for updates, and a
# network timeout there killed the 13 Sep 2026 run and every run after it.
# The model is already cached; nothing here needs the network except OneDrive.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO="/Users/casey-hemingway/Code/HimalayanTrust/semantic-image-search-mcp"
cd "$REPO"
mkdir -p "$REPO/logs"
LOG="$REPO/logs/auto_reindex.log"

{
  echo "=== reindex run: $(date '+%Y-%m-%d %H:%M:%S') ==="
  "$REPO/.venv/bin/python" reindex_missing.py
  echo "=== finished: $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo
} >> "$LOG" 2>&1
