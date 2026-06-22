#!/usr/bin/env python3
"""Bulk-index the photo library one folder at a time, evicting each folder's
downloaded originals before moving to the next.

Why folder-by-folder: the library lives online-only (OneDrive Files On-Demand)
and is far larger than free local disk. Indexing reads (materialises) each
photo; if we indexed the whole library in one pass we would download all of it
at once and run out of space. Instead we index one folder, reclaim its space
("Free Up Space" in Finder), then move on, so peak local disk stays at roughly
one folder's worth.

The index itself (embeddings + thumbnails) is tiny and persists in data/. Only
the full-res originals are transient.

Usage:
    # index the folders listed in an allow-list, pausing to free space between each
    python index_library.py --folders folders.allow.txt

    # index one or more specific folders (relative to the library root)
    python index_library.py --only "05 Nepal Trips" --only "03 HT Event Images"

    # you have plenty of disk and don't want to evict between folders
    python index_library.py --folders folders.allow.txt --no-evict

Allow-list format: one folder path per line, relative to the library root.
Blank lines and lines starting with # are ignored. Paths may be nested, e.g.
    05 Nepal Trips
    08 Historical Photographs - Ed Hillary & Expeditions/1953 Everest
"""
import argparse
import asyncio
from pathlib import Path

from src.config import Config
from src.indexer import ImageIndexer
from src.utils import find_images, format_file_size, format_duration

GB = 1024 ** 3


def folder_logical_size(folder: Path) -> int:
    """Sum logical sizes of indexable images under folder.

    Uses stat() only, which reads placeholder metadata and does NOT download
    the originals. Lets us see how much a folder would pull down before we
    commit to indexing it.
    """
    total = 0
    for p in find_images(folder):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def load_allowlist(path: Path) -> list[str]:
    names = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--folders", help="allow-list file: one folder path (relative to library root) per line")
    ap.add_argument("--only", action="append", metavar="FOLDER",
                    help="index just this folder, relative to library root (repeatable)")
    ap.add_argument("--max-gb", type=float, default=50.0,
                    help="warn and pause before any single folder larger than this (default 50)")
    ap.add_argument("--no-evict", action="store_true",
                    help="do not pause to free space between folders")
    args = ap.parse_args()

    config = Config.from_yaml(Path(args.config))
    config.validate()
    root = config.archive_path
    print(f"Library root: {root}")

    if args.only:
        folders = args.only
    elif args.folders:
        folders = load_allowlist(Path(args.folders))
    else:
        folders = sorted(p.name for p in root.iterdir() if p.is_dir())

    indexer = ImageIndexer(config)
    totals = {"indexed": 0, "errors": 0}

    for name in folders:
        folder = (root / name)
        if not folder.is_dir():
            print(f"!! skip (not a folder under library root): {name}")
            continue

        size = folder_logical_size(folder)
        print(f"\n===== {name}  ({format_file_size(size)}) =====")

        if size > args.max_gb * GB:
            print(f"!! WARNING: this folder is {format_file_size(size)}, over the "
                  f"{args.max_gb:g} GB safety limit. Indexing it will download that "
                  f"much before you can free the space.")
            print("   Consider Ctrl-C and listing its subfolders individually in the "
                  "allow-list instead.")
            resp = input("   Type 'yes' to index it anyway, anything else to skip: ").strip().lower()
            if resp != "yes":
                print("   skipped.")
                continue

        stats = await indexer.index_archive(force=False, subpath=folder, rebuild=False)
        totals["indexed"] += stats["indexed"]
        totals["errors"] += stats["errors"]
        print(f"  indexed {stats['indexed']}, errors {stats['errors']}, "
              f"{format_duration(stats['duration_seconds'])}")

        if not args.no_evict:
            input(f"  -> In Finder, right-click '{name}' and choose 'Free Up Space', "
                  f"then press Enter to continue...")

    print("\nRebuilding FAISS index from all stored embeddings...")
    await indexer._rebuild_faiss_index()
    print(f"\nDONE. total indexed {totals['indexed']}, errors {totals['errors']}")


if __name__ == "__main__":
    asyncio.run(main())
