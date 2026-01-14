#!/usr/bin/env python3
"""Test script to index sample images and verify functionality."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.indexer import ImageIndexer
from src.searcher import ImageSearcher


async def main():
    """Test indexing and search functionality."""
    print("=" * 60)
    print("Semantic Image Search - Test Indexing")
    print("=" * 60)

    # Load configuration
    print("\n1. Loading configuration...")
    config_path = Path("config.yml")

    if not config_path.exists():
        print("ERROR: config.yml not found")
        print("Please create config.yml from config.yml.example")
        return 1

    try:
        config = Config.from_yaml(config_path)
        config.validate()
        print(f"   ✓ Configuration loaded")
        print(f"   Archive path: {config.archive_path}")
        print(f"   Data directory: {config.data_dir}")
    except Exception as e:
        print(f"   ERROR: Failed to load configuration: {e}")
        return 1

    # Run indexing
    print("\n2. Indexing images...")
    indexer = ImageIndexer(config)

    try:
        stats = await indexer.index_archive(force=True)
        print(f"\n   ✓ Indexing complete!")
        print(f"   Total images found: {stats['total_images']}")
        print(f"   Successfully indexed: {stats['indexed']}")
        print(f"   Errors: {stats['errors']}")
        print(f"   Duration: {stats['duration_seconds']:.1f}s")

        if stats['indexed'] == 0:
            print("\n   WARNING: No images were indexed.")
            print(f"   Make sure {config.archive_path} contains image files (.jpg, .png, etc.)")
            return 1

    except Exception as e:
        print(f"\n   ERROR: Indexing failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Test search
    print("\n3. Testing search functionality...")
    searcher = ImageSearcher(config)

    test_queries = [
        "person in a room",
        "outdoor scene",
        "landscape",
    ]

    for query in test_queries:
        print(f"\n   Query: '{query}'")
        try:
            results = await searcher.search(query, limit=3)
            print(f"   Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"      {i}. {result['filename']} (similarity: {result['similarity']:.3f})")
        except Exception as e:
            print(f"   ERROR: Search failed: {e}")

    # Get archive stats
    print("\n4. Archive statistics...")
    try:
        stats = await searcher.get_archive_stats()
        print(f"   Total images: {stats['total_images']}")
        print(f"   Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
        print(f"   Total size: {stats['total_size_gb']} GB")
        if stats['top_cameras']:
            print(f"   Top camera: {stats['top_cameras'][0]['model']} ({stats['top_cameras'][0]['count']} images)")
    except Exception as e:
        print(f"   ERROR: Failed to get stats: {e}")

    print("\n" + "=" * 60)
    print("✓ Test complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Add this server to Claude Desktop config:")
    print(f"   Command: python3")
    print(f"   Args: [\"{Path(__file__).parent.resolve() / 'run_server.py'}\"]")
    print("2. Restart Claude Desktop")
    print("3. Try searching your photos through Claude!")
    print()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
