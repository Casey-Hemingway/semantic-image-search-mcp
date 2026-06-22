"""Prune orphaned index entries whose source image no longer exists on disk.

Deletes rows from `images` (and their `image_tags` links) where the stored
filepath is missing, removes the matching thumbnail, vacuums the DB, then
rebuilds the FAISS index so it matches the live photo library exactly.

Backups (data/metadata.db.bak, data/embeddings.faiss.bak) should exist before
running. Safe to re-run: a clean library reports 0 orphans.
"""

import asyncio
import os
import sqlite3
from pathlib import Path

from src.config import Config
from src.indexer import ImageIndexer

DB_PATH = "data/metadata.db"
THUMB_DIR = "data/thumbnails"


def prune() -> int:
    db = sqlite3.connect(DB_PATH)
    cur = db.cursor()

    cur.execute("SELECT image_id, filepath FROM images")
    rows = cur.fetchall()
    orphans = [iid for iid, fp in rows if not (fp and os.path.exists(fp))]
    print(f"orphans to delete: {len(orphans)}")
    if not orphans:
        db.close()
        return 0

    cur.executemany("DELETE FROM image_tags WHERE image_id=?", [(i,) for i in orphans])
    cur.executemany("DELETE FROM images WHERE image_id=?", [(i,) for i in orphans])
    db.commit()

    orphan_set = set(orphans)
    removed_thumbs = 0
    for fn in os.listdir(THUMB_DIR):
        if os.path.splitext(fn)[0] in orphan_set:
            os.remove(os.path.join(THUMB_DIR, fn))
            removed_thumbs += 1
    print(f"thumbnails removed: {removed_thumbs}")

    cur.execute("SELECT count(*) FROM images")
    print(f"rows remaining: {cur.fetchone()[0]}")
    cur.execute("VACUUM")
    db.close()
    print("vacuum done")
    return len(orphans)


async def rebuild() -> None:
    config = Config.from_yaml(Path("config.yml"))
    indexer = ImageIndexer(config)
    # Reuse the private rebuild that reads all embeddings from the DB.
    await indexer._rebuild_faiss_index()


if __name__ == "__main__":
    deleted = prune()
    if deleted:
        asyncio.run(rebuild())
    print("DONE")
