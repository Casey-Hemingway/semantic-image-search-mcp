"""Utility functions for semantic image search."""

from pathlib import Path
from typing import Iterator
import torch
from PIL import Image, ImageOps


def get_optimal_device(device_preference: str = "auto") -> torch.device:
    """Auto-detect best available device for PyTorch.

    Args:
        device_preference: Device preference ("auto", "mps", "cuda", "cpu")

    Returns:
        torch.device object for the selected device
    """
    if device_preference == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("✓ Using Apple Silicon MPS (Metal Performance Shaders)")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name()}")
            return torch.device("cuda")
        else:
            print("✓ Using CPU (consider Apple Silicon or NVIDIA GPU for faster inference)")
            return torch.device("cpu")
    else:
        device = torch.device(device_preference)
        print(f"✓ Using device: {device}")
        return device


def generate_thumbnail(
    image_path: Path, output_path: Path, max_size: int = 512, quality: int = 85
) -> None:
    """Generate thumbnail for image.

    Args:
        image_path: Path to source image
        output_path: Path to save thumbnail
        max_size: Maximum dimension (maintains aspect ratio)
        quality: JPEG quality (0-100)

    Raises:
        Exception: If image cannot be processed
    """
    with Image.open(image_path) as img:
        # Apply the EXIF orientation. Without this, ~2,400 library photos shot
        # rotated got sideways or flipped thumbnails, which were then embedded
        # and scored sideways too. The saved JPEG carries no EXIF, so the
        # thumbnail itself must be upright.
        img = ImageOps.exif_transpose(img)

        # Convert RGBA to RGB if necessary
        if img.mode in ("RGBA", "LA", "P"):
            # Create white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = background

        # Preserve aspect ratio
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save thumbnail
        img.save(output_path, "JPEG", quality=quality, optimize=True)


def batch_iterator(items: list, batch_size: int) -> Iterator[list]:
    """Yield batches of items.

    Args:
        items: List of items to batch
        batch_size: Size of each batch

    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def find_images(root: Path, extensions: set = None) -> list[Path]:
    """Recursively find all image files in directory.

    Args:
        root: Root directory to search
        extensions: Set of file extensions to include (default: common image formats)

    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp", ".WEBP"}

    image_paths = []
    for ext in extensions:
        # Use rglob to recursively find files
        image_paths.extend(root.rglob(f"*{ext}"))

    # Sort by modification time (newest first)
    image_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return image_paths


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB", "245 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2m 30s", "45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
