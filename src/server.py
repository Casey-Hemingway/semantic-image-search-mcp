"""MCP server for semantic image search."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .config import Config
from .indexer import ImageIndexer
from .searcher import ImageSearcher


# Initialize MCP server
mcp = FastMCP("semantic-image-search")

# Global state (initialized on first use)
config: Optional[Config] = None
indexer: Optional[ImageIndexer] = None
searcher: Optional[ImageSearcher] = None


def _ensure_initialized():
    """Lazy initialization of global components.

    Raises:
        FileNotFoundError: If config.yml not found
        Exception: If initialization fails
    """
    global config, indexer, searcher

    if config is None:
        # Use absolute path relative to this file's location
        config_path = Path(__file__).parent.parent / "config.yml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.yml not found at {config_path}. Please create one based on config.yml.example"
            )

        config = Config.from_yaml(config_path)
        config.validate()

        indexer = ImageIndexer(config)
        searcher = ImageSearcher(config)

        logging.info(f"Initialized with archive: {config.archive_path}")


@mcp.tool()
async def search_images(
    query: str,
    limit: int = 10,
    date_from: str = None,
    date_to: str = None,
    folder_pattern: str = None,
) -> dict:
    """Search images using natural language queries with optional filters.

    This tool performs semantic search across your photo archive using AI-powered
    image understanding. It finds images based on their visual content, not just
    filenames or tags.

    Args:
        query: Natural language search query describing what you're looking for.
               Examples: "person in a room", "mountain landscape", "children playing",
               "sunset over water", "food on a table"
        limit: Maximum number of results to return (default: 10, max: 100)
        date_from: Filter results to images taken on or after this date (ISO format: YYYY-MM-DD)
        date_to: Filter results to images taken on or before this date (ISO format: YYYY-MM-DD)
        folder_pattern: Filter by folder path pattern (e.g., "2024" or "vacation")

    Returns:
        Dictionary containing:
        - success: Whether the search succeeded
        - query: The search query used
        - count: Number of results found
        - results: List of matching images with metadata and similarity scores
    """
    try:
        _ensure_initialized()

        results = await searcher.search(
            query=query,
            limit=limit,
            date_from=date_from,
            date_to=date_to,
            folder_pattern=folder_pattern,
        )

        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        logging.error(f"Search error: {e}", exc_info=True)
        return {"success": False, "error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def get_image_info(image_id: str) -> dict:
    """Get detailed metadata for a specific image.

    Retrieves comprehensive information about an image including EXIF data,
    camera settings, location, and file details.

    Args:
        image_id: Unique identifier for the image (obtained from search results)

    Returns:
        Dictionary containing:
        - success: Whether the operation succeeded
        - image: Complete metadata including:
          - filepath: Full path to the image file
          - filename: Name of the file
          - folder: Folder containing the image
          - width/height: Image dimensions
          - date_taken: When the photo was taken (if available)
          - camera_make/model: Camera information
          - lens_model: Lens used
          - focal_length: Focal length in mm
          - aperture: F-stop value
          - iso: ISO sensitivity
          - shutter_speed: Exposure time
          - gps_latitude/longitude: Location coordinates (if available)
          - tags: Associated tags
    """
    try:
        _ensure_initialized()

        info = await searcher.get_image_info(image_id)

        return {"success": True, "image": info}
    except Exception as e:
        logging.error(f"Get image info error: {e}", exc_info=True)
        return {"success": False, "error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def reindex_archive(force: bool = False) -> dict:
    """Re-index the photo archive for new or modified images.

    Scans the archive for new or changed images and updates the search index.
    This should be run after adding new photos to the archive.

    Args:
        force: If True, re-index ALL images (slow). If False, only index
               new or modified images (default, recommended)

    Returns:
        Dictionary containing:
        - success: Whether indexing succeeded
        - total_images: Total number of images found
        - indexed: Number of images indexed
        - errors: Number of errors encountered
        - duration_seconds: Time taken to complete indexing
    """
    try:
        _ensure_initialized()

        stats = await indexer.index_archive(force=force)

        return {"success": True, **stats}
    except Exception as e:
        logging.error(f"Reindex error: {e}", exc_info=True)
        return {"success": False, "error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def get_archive_stats() -> dict:
    """Get statistics about the indexed photo archive.

    Provides an overview of your photo collection including total count,
    date range, storage size, and distribution by camera and folder.

    Returns:
        Dictionary containing:
        - success: Whether the operation succeeded
        - total_images: Total number of indexed images
        - date_range: Earliest and latest photo dates
        - total_size_bytes: Total storage used in bytes
        - total_size_gb: Total storage used in gigabytes
        - top_cameras: Most common camera models
        - top_folders: Folders with the most images
    """
    try:
        _ensure_initialized()

        stats = await searcher.get_archive_stats()

        return {"success": True, **stats}
    except Exception as e:
        logging.error(f"Get stats error: {e}", exc_info=True)
        return {"success": False, "error": str(e), "error_type": type(e).__name__}


def setup_logging():
    """Configure logging for the MCP server."""
    # Log to file only (not stdout/stderr to avoid interfering with MCP stdio)
    # Use absolute path to avoid permission issues
    log_path = Path(__file__).parent.parent / "mcp-server.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
        ],
    )

    # Reduce noise from transformers library
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def main():
    """Main entry point for MCP server."""
    setup_logging()

    logging.info("Starting MCP Semantic Image Search Server")
    logging.info(f"Python version: {sys.version}")

    try:
        # Run the MCP server
        mcp.run(transport="stdio")
    except Exception as e:
        logging.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
