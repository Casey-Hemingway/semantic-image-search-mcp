#!/usr/bin/env python3
"""Re-embed an existing index with a different CLIP model, from local thumbnails.

Built for the Sep 2026 upgrade from ViT-B/32 to ViT-L/14. The index already
holds a 512px thumbnail of every photo, and CLIP looks at 224px, so embedding
from the thumbnails loses nothing the model would use and needs no OneDrive
downloads. Only rows with no thumbnail fall back to the original file.

The source data dir is never modified. The destination starts as an APFS clone
of the source (instant, no extra disk until rows change), then every row gets a
new embedding, an aesthetic score (when --aesthetic is given) and long_edge_px.

Resumable: rows whose embedding already has the new model's dimension are
skipped, and each batch is committed as it finishes.

Examples:
    # Pilot: 500 random photos, thumbnails only, no FAISS rebuild
    python reembed_from_thumbnails.py --src data.clip-b32.bak --dst data-pilot \\
        --limit 500 --thumbnails-only

    # Full run
    python reembed_from_thumbnails.py --src data --dst data-l14
"""

import argparse
import asyncio
import random
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.config import Config
from src.indexer import ClipEmbedder, ImageIndexer
from src.metadata import init_database
from src.quality import AestheticScorer

DEFAULT_MODEL = "openai/clip-vit-large-patch14"
DEFAULT_AESTHETIC = "models/sac+logos+ava1-l14-linearMSE.pth"


def clone_data_dir(src: Path, dst: Path) -> None:
    # cp -c makes APFS clones: copy-on-write, so the 1.2 GB of thumbnails
    # costs nothing until something writes to them. Only the ~2,400 thumbnails
    # that need turning upright get rewritten, and only in the destination.
    subprocess.run(["cp", "-c", "-R", str(src), str(dst)], check=True)
    for suffix in ("-wal", "-shm"):
        stale = dst / f"metadata.db{suffix}"
        if stale.exists() and not (src / f"metadata.db{suffix}").exists():
            stale.unlink()


def upright_thumbnail(thumb: Path, orientation: int) -> None:
    """Rotate a thumbnail saved before generate_thumbnail honoured EXIF.

    Old thumbnails were written without the source's EXIF orientation applied
    and without EXIF, so the stored orientation column is the only record.
    Rewrites the file in place (in the destination's clone, never the source).
    """
    if orientation in (None, 0, 1):
        return
    with Image.open(thumb) as im:
        exif = im.getexif()
        exif[0x0112] = orientation
        im.info["exif"] = exif.tobytes()
        fixed = ImageOps.exif_transpose(im)
        fixed.save(thumb, "JPEG", quality=85, optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--aesthetic", type=Path, default=Path(DEFAULT_AESTHETIC))
    ap.add_argument("--no-aesthetic", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, help="Random sample of N rows (pilot)")
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument(
        "--thumbnails-only",
        action="store_true",
        help="Skip rows with no thumbnail instead of reading the original",
    )
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if not (src / "metadata.db").exists():
        print(f"No metadata.db in {src}", file=sys.stderr)
        return 1
    if not dst.exists():
        print(f"Cloning {src} -> {dst}")
        clone_data_dir(src, dst)

    db_path = dst / "metadata.db"
    asyncio.run(init_database(db_path))  # adds long_edge_px / aesthetic_score

    embedder = ClipEmbedder(args.model, "auto")
    dim = embedder.model.config.projection_dim
    scorer = None if args.no_aesthetic else AestheticScorer(args.aesthetic)

    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT image_id, filepath, width, height, length(embedding_vector), orientation "
        "FROM images"
    ).fetchall()
    todo = [r for r in rows if r[4] != dim * 4]
    if args.limit:
        random.Random(args.seed).shuffle(todo)
        todo = todo[: args.limit]
    print(f"{len(rows)} rows, {len(rows) - len(todo) if not args.limit else 0} already at {dim}-d, {len(todo)} to embed")

    thumbs = dst / "thumbnails"
    done = failed = from_original = rotated = 0
    start = time.time()
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        paths, ids, sizes = [], [], []
        for image_id, filepath, width, height, _, orientation in batch:
            thumb = thumbs / f"{image_id}.jpg"
            if thumb.exists():
                upright_thumbnail(thumb, orientation)
                rotated += orientation not in (None, 0, 1)
                paths.append(thumb)
            elif args.thumbnails_only:
                continue
            else:
                paths.append(Path(filepath))
                from_original += 1
            ids.append(image_id)
            sizes.append(max(width or 0, height or 0) or None)
        if not paths:
            continue
        try:
            emb = embedder.embed_images(paths)
            scores = scorer.score(emb) if scorer else [None] * len(ids)
        except Exception as e:  # one bad file must not sink the run
            print(f"batch at {i} failed ({e}); retrying one by one")
            emb, scores, keep = [], [], []
            for j, p in enumerate(paths):
                try:
                    e1 = embedder.embed_images([p])
                    emb.append(e1[0])
                    scores.append(scorer.score(e1)[0] if scorer else None)
                    keep.append(j)
                except Exception as e2:
                    print(f"  failed {ids[j]}: {e2}")
                    con.execute(
                        "UPDATE images SET embedding_vector = NULL WHERE image_id = ?",
                        (ids[j],),
                    )
                    failed += 1
            ids = [ids[j] for j in keep]
            sizes = [sizes[j] for j in keep]
            emb = np.array(emb)
        con.executemany(
            "UPDATE images SET embedding_vector = ?, aesthetic_score = ?, long_edge_px = ? "
            "WHERE image_id = ?",
            [
                (e.astype(np.float32).tobytes(), None if s is None else float(s), le, iid)
                for e, s, le, iid in zip(emb, scores, sizes, ids)
            ],
        )
        con.commit()
        done += len(ids)
        if (i // args.batch_size) % 25 == 0:
            rate = done / max(time.time() - start, 1e-6)
            left = (len(todo) - i - len(batch)) / max(rate, 1e-6)
            print(f"  {done}/{len(todo)}  {rate:.1f} img/s  ~{left / 60:.0f} min left", flush=True)

    elapsed = time.time() - start
    con.execute(
        "INSERT OR REPLACE INTO index_metadata (key, value) VALUES ('embedding_model', ?)",
        (args.model,),
    )
    con.commit()

    # Rows the run could not embed at the new dimension would break the FAISS
    # build, so a full run clears any leftover old-dimension vectors.
    if not args.limit:
        stale = con.execute(
            "SELECT count(*) FROM images WHERE embedding_vector IS NOT NULL "
            "AND length(embedding_vector) != ?",
            (dim * 4,),
        ).fetchone()[0]
        if stale:
            print(f"Clearing {stale} rows still at the old dimension")
            con.execute(
                "UPDATE images SET embedding_vector = NULL WHERE length(embedding_vector) != ?",
                (dim * 4,),
            )
            con.commit()
    con.close()

    print(f"\nEmbedded {done} ({from_original} from originals, {rotated} thumbnails "
          f"turned upright), failed {failed}, "
          f"{elapsed:.0f}s, {done / max(elapsed, 1e-6):.1f} img/s")

    if not args.limit:
        cfg = Config(archive_path=Path("/"), data_dir=dst)
        asyncio.run(ImageIndexer(cfg)._rebuild_faiss_index())
    return 0


if __name__ == "__main__":
    sys.exit(main())
