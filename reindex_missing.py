#!/usr/bin/env python3
"""Incrementally index new/modified photos and rebuild the FAISS index.

Runs index_archive(force=False), so only images whose filepath is absent from
the DB (or whose file hash has changed) are embedded. Existing entries are left
untouched. Safe to re-run: a fully-indexed library reports 0 new images.
"""

import asyncio
from pathlib import Path

from src.config import Config
from src.indexer import ImageIndexer


async def main() -> None:
    config = Config.from_yaml(Path("config.yml"))
    config.validate()
    indexer = ImageIndexer(config)
    stats = await indexer.index_archive(force=False)
    print("\n=== Incremental reindex summary ===")
    print(f"  newly indexed: {stats['indexed']}")
    print(f"  errors:        {stats['errors']}")
    print(f"  duration:      {stats['duration_seconds']:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
