#!/usr/bin/env python3
"""Scan the index for photos stored sideways or upside down, and fix the confident ones.

Stage 1 (scan, resumable): for every photo, score its thumbnail at 90/180/270
degrees with detect_rotation.RotationDetector. The 0-degree score comes from
the stored embedding, which was computed from the same thumbnail. Scores go
into the rotation_scan table.

Stage 2 (apply): photos where a rotation beats upright by at least
--apply-margin get their thumbnail turned upright in place, and are then
re-embedded and re-scored (aesthetic and taste). The fix is recorded in
images.rotation_fix (degrees CCW), and the FAISS index is rebuilt. The
OneDrive originals are never touched: the fix lives in the index, the
thumbnails, and rotation_fix for exporters to apply.

Photos in the uncertain band (--review-margin up to --apply-margin) are
written to a CSV for a human to check rather than rotated.

Margins were set on EXIF ground truth, 23 Sep 2026 (150 rotated, 286
upright): margin 0.03 caught 32/150 with 0 false alarms; 0.01 caught 111/150
with 6/286 false alarms, which is too many to apply unattended across a
library of 25k.

Usage:
    python scan_rotation.py --data data-l14 [--apply-margin 0.03] [--review-margin 0.01]
"""

import argparse
import asyncio
import csv
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from detect_rotation import ROTATIONS, RotationDetector
from src.config import Config
from src.indexer import ClipEmbedder, ImageIndexer
from src.quality import AestheticScorer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data-l14"))
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--aesthetic", type=Path, default=Path("models/sac+logos+ava1-l14-linearMSE.pth"))
    ap.add_argument("--taste", type=Path, default=Path("models/taste-linear.npz"))
    ap.add_argument("--apply-margin", type=float, default=0.03)
    ap.add_argument("--review-margin", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-apply", action="store_true")
    args = ap.parse_args()

    db_path = args.data / "metadata.db"
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS rotation_scan (image_id TEXT PRIMARY KEY, "
        "s0 REAL, s90 REAL, s180 REAL, s270 REAL, scanned_at TEXT)"
    )
    cols = {r[1] for r in con.execute("PRAGMA table_info(images)")}
    if "rotation_fix" not in cols:
        con.execute("ALTER TABLE images ADD COLUMN rotation_fix INTEGER")
    con.commit()

    det = RotationDetector(ClipEmbedder(args.model, "auto"))
    todo = con.execute(
        "SELECT i.image_id, i.embedding_vector FROM images i LEFT JOIN rotation_scan r "
        "ON r.image_id = i.image_id WHERE r.image_id IS NULL AND i.embedding_vector IS NOT NULL "
        "AND i.rotation_fix IS NULL"
    ).fetchall()
    thumbs = args.data / "thumbnails"
    todo = [(i, e) for i, e in todo if (thumbs / f"{i}.jpg").exists()]
    print(f"stage 1: {len(todo)} photos to scan", flush=True)

    start = time.time()
    for k in range(0, len(todo), args.batch):
        batch = todo[k : k + args.batch]
        paths = [thumbs / f"{i}.jpg" for i, _ in batch]
        s = det.scores_rotated(paths)  # [n, 3] for 90/180/270
        s0 = np.array([np.frombuffer(e, dtype=np.float32) @ det.direction for _, e in batch])
        con.executemany(
            "INSERT OR REPLACE INTO rotation_scan VALUES (?,?,?,?,?,datetime('now'))",
            [(i, float(a), float(b), float(c), float(d)) for (i, _), a, (b, c, d) in zip(batch, s0, s)],
        )
        con.commit()
        done = k + len(batch)
        if (k // args.batch) % 50 == 0:
            rate = done / max(time.time() - start, 1e-6)
            print(f"  {done}/{len(todo)}  {rate:.1f} img/s  ~{(len(todo) - done) / max(rate, 1e-6) / 60:.0f} min left", flush=True)

    rows = con.execute("SELECT image_id, s0, s90, s180, s270 FROM rotation_scan").fetchall()
    apply, review = [], []
    for iid, *sc in rows:
        sc = np.array(sc)
        best = int(sc.argmax())
        gain = sc[best] - sc[0]
        if best and gain >= args.apply_margin:
            apply.append((iid, ROTATIONS[best], gain))
        elif best and gain >= args.review_margin:
            review.append((iid, ROTATIONS[best], gain))
    print(f"stage 2: {len(apply)} to fix (margin >= {args.apply_margin}), {len(review)} for review", flush=True)

    with open(args.data / "rotation-review.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "suggested_rotation_ccw", "margin", "filepath"])
        fp = dict(con.execute("SELECT image_id, filepath FROM images"))
        for iid, r, g in sorted(review, key=lambda x: -x[2]):
            w.writerow([iid, r, f"{g:.4f}", fp.get(iid, "")])

    if args.no_apply or not apply:
        return 0

    scorer = AestheticScorer(args.aesthetic)
    taste = np.load(args.taste) if args.taste.exists() else None
    for k in range(0, len(apply), args.batch):
        batch = apply[k : k + args.batch]
        for iid, r, _ in batch:
            p = thumbs / f"{iid}.jpg"
            with Image.open(p) as im:
                fixed = im.convert("RGB").rotate(r, expand=True)
            fixed.save(p, "JPEG", quality=85, optimize=True)
        emb = det.e.embed_images([thumbs / f"{iid}.jpg" for iid, _, _ in batch])
        aes = scorer.score(emb)
        tz = ((emb @ taste["w"]) - taste["mu"]) / taste["sd"] if taste is not None else [None] * len(batch)
        con.executemany(
            "UPDATE images SET embedding_vector = ?, aesthetic_score = ?, taste_score = ?, rotation_fix = ? "
            "WHERE image_id = ?",
            [
                (e.astype(np.float32).tobytes(), float(a), None if t is None else float(t), r, iid)
                for e, a, t, (iid, r, _) in zip(emb, aes, tz, batch)
            ],
        )
        con.execute(
            f"DELETE FROM rotation_scan WHERE image_id IN ({','.join('?' * len(batch))})",
            [iid for iid, _, _ in batch],
        )
        con.commit()
    con.close()
    asyncio.run(ImageIndexer(Config(archive_path=Path("/"), data_dir=args.data))._rebuild_faiss_index())
    print(f"fixed {len(apply)} photos; review list at {args.data / 'rotation-review.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
