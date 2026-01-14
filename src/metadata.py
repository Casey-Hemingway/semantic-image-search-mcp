"""Metadata extraction and database management for image search."""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import hashlib
import re
import aiosqlite
import exifread
from PIL import Image


@dataclass
class ImageMetadata:
    """Metadata for a single image."""

    image_id: str
    filepath: str
    filename: str
    folder: str
    file_hash: str
    file_size: int
    width: int
    height: int
    date_taken: Optional[str] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    focal_length: Optional[float] = None
    aperture: Optional[float] = None
    iso: Optional[int] = None
    shutter_speed: Optional[str] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    orientation: int = 1
    file_modified: Optional[str] = None
    file_created: Optional[str] = None
    indexed_at: Optional[str] = None


# Database schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    image_id TEXT PRIMARY KEY,
    filepath TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    folder TEXT,
    file_hash TEXT,
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    date_taken TEXT,
    camera_make TEXT,
    camera_model TEXT,
    lens_model TEXT,
    focal_length REAL,
    aperture REAL,
    iso INTEGER,
    shutter_speed TEXT,
    gps_latitude REAL,
    gps_longitude REAL,
    orientation INTEGER,
    file_modified TEXT,
    file_created TEXT,
    indexed_at TEXT,
    embedding_vector BLOB
);

CREATE INDEX IF NOT EXISTS idx_date_taken ON images(date_taken);
CREATE INDEX IF NOT EXISTS idx_folder ON images(folder);
CREATE INDEX IF NOT EXISTS idx_filename ON images(filename);
CREATE INDEX IF NOT EXISTS idx_file_hash ON images(file_hash);
CREATE INDEX IF NOT EXISTS idx_camera_model ON images(camera_model);

CREATE TABLE IF NOT EXISTS tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS image_tags (
    image_id TEXT,
    tag_id INTEGER,
    FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE,
    PRIMARY KEY (image_id, tag_id)
);

CREATE TABLE IF NOT EXISTS index_metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Enable Write-Ahead Logging for better concurrency
PRAGMA journal_mode=WAL;
"""


async def init_database(db_path: Path) -> None:
    """Initialize the database with schema.

    Args:
        db_path: Path to SQLite database file
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA)
        await db.commit()

        # Store schema version
        await db.execute(
            "INSERT OR REPLACE INTO index_metadata (key, value) VALUES (?, ?)",
            ("schema_version", "1.0"),
        )
        await db.commit()


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of first 8KB of file for change detection.

    Args:
        path: Path to file

    Returns:
        Hexadecimal hash string
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read(8192)).hexdigest()


def generate_image_id(filepath: Path) -> str:
    """Generate unique ID for image based on filepath.

    Args:
        filepath: Path to image file

    Returns:
        Unique identifier string
    """
    return hashlib.sha256(str(filepath).encode()).hexdigest()[:16]


def parse_filename_metadata(filename: str) -> dict:
    """Parse metadata from filename if it follows a known pattern.

    Handles patterns like: "SK Monitoring 2024 (22.04.24) -1466.jpg"

    Args:
        filename: Filename to parse

    Returns:
        Dictionary with parsed metadata (empty if no pattern matches)
    """
    # Pattern: "Event Name (DD.MM.YY) -NUMBER"
    pattern = r"(.+?)\s*\((\d{2})\.(\d{2})\.(\d{2})\)\s*-(\d+)"
    match = re.match(pattern, filename)

    if match:
        day = int(match.group(2))
        month = int(match.group(3))
        year = 2000 + int(match.group(4))

        return {
            "event_name": match.group(1).strip(),
            "day": day,
            "month": month,
            "year": year,
            "sequence": int(match.group(5)),
        }

    return {}


def parse_exif_datetime(exif_dict: dict, tag_name: str) -> Optional[str]:
    """Parse EXIF datetime tag to ISO format.

    Args:
        exif_dict: Dictionary of EXIF tags
        tag_name: Name of datetime tag to parse

    Returns:
        ISO formatted datetime string or None if not available
    """
    if tag_name not in exif_dict:
        return None

    try:
        date_str = str(exif_dict[tag_name])
        # EXIF format: "YYYY:MM:DD HH:MM:SS"
        dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
        return dt.isoformat()
    except (ValueError, TypeError):
        return None


def parse_exif_rational(exif_dict: dict, tag_name: str) -> Optional[float]:
    """Parse EXIF rational number tag.

    Args:
        exif_dict: Dictionary of EXIF tags
        tag_name: Name of tag to parse

    Returns:
        Float value or None if not available
    """
    if tag_name not in exif_dict:
        return None

    try:
        value = exif_dict[tag_name]
        if hasattr(value, "values") and len(value.values) > 0:
            rational = value.values[0]
            if hasattr(rational, "num") and hasattr(rational, "den") and rational.den != 0:
                return float(rational.num) / float(rational.den)
        return float(value)
    except (ValueError, TypeError, AttributeError, ZeroDivisionError):
        return None


def parse_exif_gps(exif_dict: dict) -> tuple[Optional[float], Optional[float]]:
    """Parse GPS coordinates from EXIF data.

    Args:
        exif_dict: Dictionary of EXIF tags

    Returns:
        Tuple of (latitude, longitude) or (None, None) if not available
    """
    try:
        if "GPS GPSLatitude" not in exif_dict or "GPS GPSLongitude" not in exif_dict:
            return None, None

        lat_ref = str(exif_dict.get("GPS GPSLatitudeRef", "N"))
        lon_ref = str(exif_dict.get("GPS GPSLongitudeRef", "E"))

        lat_values = exif_dict["GPS GPSLatitude"].values
        lon_values = exif_dict["GPS GPSLongitude"].values

        # Convert to decimal degrees
        lat = float(lat_values[0].num) / float(lat_values[0].den)
        lat += (float(lat_values[1].num) / float(lat_values[1].den)) / 60
        lat += (float(lat_values[2].num) / float(lat_values[2].den)) / 3600

        lon = float(lon_values[0].num) / float(lon_values[0].den)
        lon += (float(lon_values[1].num) / float(lon_values[1].den)) / 60
        lon += (float(lon_values[2].num) / float(lon_values[2].den)) / 3600

        if lat_ref == "S":
            lat = -lat
        if lon_ref == "W":
            lon = -lon

        return lat, lon

    except (ValueError, TypeError, AttributeError, ZeroDivisionError, KeyError, IndexError):
        return None, None


def extract_metadata(image_path: Path, archive_root: Path) -> ImageMetadata:
    """Extract comprehensive metadata from image file.

    Args:
        image_path: Path to image file
        archive_root: Root path of archive (for computing relative folder)

    Returns:
        ImageMetadata object with all extracted information

    Raises:
        Exception: If image cannot be opened or processed
    """
    # Generate unique ID
    image_id = generate_image_id(image_path)

    # Compute file hash
    file_hash = compute_file_hash(image_path)

    # Get file stats
    stats = image_path.stat()
    file_size = stats.st_size
    file_modified = datetime.fromtimestamp(stats.st_mtime).isoformat()
    file_created = datetime.fromtimestamp(stats.st_ctime).isoformat()

    # Compute relative folder path
    try:
        folder = str(image_path.parent.relative_to(archive_root))
    except ValueError:
        # If not relative to archive_root, use parent folder name
        folder = image_path.parent.name

    # Initialize metadata
    metadata = ImageMetadata(
        image_id=image_id,
        filepath=str(image_path),
        filename=image_path.name,
        folder=folder,
        file_hash=file_hash,
        file_size=file_size,
        width=0,
        height=0,
        file_modified=file_modified,
        file_created=file_created,
        indexed_at=datetime.now().isoformat(),
    )

    # Extract image dimensions
    try:
        with Image.open(image_path) as img:
            metadata.width, metadata.height = img.size
            metadata.orientation = img.getexif().get(0x0112, 1) if hasattr(img, "getexif") else 1
    except Exception:
        # If PIL fails, we'll still have basic metadata
        pass

    # Extract EXIF data
    try:
        with open(image_path, "rb") as f:
            exif_tags = exifread.process_file(f, details=False)

            # Date taken
            metadata.date_taken = parse_exif_datetime(exif_tags, "EXIF DateTimeOriginal") or parse_exif_datetime(
                exif_tags, "Image DateTime"
            )

            # Camera info
            metadata.camera_make = str(exif_tags["Image Make"]) if "Image Make" in exif_tags else None
            metadata.camera_model = (
                str(exif_tags["Image Model"]) if "Image Model" in exif_tags else None
            )
            metadata.lens_model = (
                str(exif_tags["EXIF LensModel"]) if "EXIF LensModel" in exif_tags else None
            )

            # Camera settings
            metadata.focal_length = parse_exif_rational(exif_tags, "EXIF FocalLength")
            metadata.aperture = parse_exif_rational(exif_tags, "EXIF FNumber")

            if "EXIF ISOSpeedRatings" in exif_tags:
                try:
                    metadata.iso = int(str(exif_tags["EXIF ISOSpeedRatings"]))
                except (ValueError, TypeError):
                    pass

            if "EXIF ExposureTime" in exif_tags:
                metadata.shutter_speed = str(exif_tags["EXIF ExposureTime"])

            # GPS coordinates
            metadata.gps_latitude, metadata.gps_longitude = parse_exif_gps(exif_tags)

    except Exception:
        # EXIF extraction failed, continue with basic metadata
        pass

    return metadata


async def store_image_metadata(
    db: aiosqlite.Connection, metadata: ImageMetadata, embedding: Optional[bytes] = None
) -> None:
    """Store image metadata in database.

    Args:
        db: Database connection
        metadata: Image metadata to store
        embedding: Optional embedding vector as bytes
    """
    data = asdict(metadata)
    data["embedding_vector"] = embedding

    columns = ", ".join(data.keys())
    placeholders = ", ".join("?" * len(data))

    await db.execute(
        f"INSERT OR REPLACE INTO images ({columns}) VALUES ({placeholders})",
        tuple(data.values()),
    )
