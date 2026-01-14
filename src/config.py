"""Configuration management for semantic image search MCP server."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ClipConfig:
    """Configuration for CLIP model."""

    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"  # auto, mps, cuda, cpu
    batch_size: int = 32


@dataclass
class SearchConfig:
    """Configuration for search functionality."""

    default_limit: int = 10
    max_limit: int = 100
    similarity_threshold: float = 0.0


@dataclass
class ThumbnailConfig:
    """Configuration for thumbnail generation."""

    enabled: bool = True
    max_size: int = 512
    quality: int = 85


@dataclass
class Config:
    """Main configuration for the semantic image search system."""

    archive_path: Path
    data_dir: Path
    clip: ClipConfig = field(default_factory=ClipConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    thumbnails: ThumbnailConfig = field(default_factory=ThumbnailConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Config object with validated settings

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required fields are missing or invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError("Configuration file is empty")

        # Validate required fields
        if "archive_path" not in data:
            raise ValueError("Missing required field: archive_path")

        # Convert paths
        archive_path = Path(data["archive_path"]).expanduser().resolve()
        if not archive_path.exists():
            raise ValueError(f"Archive path does not exist: {archive_path}")

        data_dir = Path(data.get("data_dir", "./data")).expanduser().resolve()
        data_dir.mkdir(parents=True, exist_ok=True)

        # Parse nested config objects
        clip_config = ClipConfig(**data.get("clip", {}))
        search_config = SearchConfig(**data.get("search", {}))
        thumbnail_config = ThumbnailConfig(**data.get("thumbnails", {}))

        return cls(
            archive_path=archive_path,
            data_dir=data_dir,
            clip=clip_config,
            search=search_config,
            thumbnails=thumbnail_config,
        )

    @property
    def db_path(self) -> Path:
        """Get path to SQLite database."""
        return self.data_dir / "metadata.db"

    @property
    def index_path(self) -> Path:
        """Get path to FAISS index."""
        return self.data_dir / "embeddings.faiss"

    @property
    def thumbnails_dir(self) -> Path:
        """Get path to thumbnails directory."""
        thumbnails_dir = self.data_dir / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        return thumbnails_dir

    def validate(self) -> None:
        """Validate configuration settings.

        Raises:
            ValueError: If settings are invalid
        """
        if not self.archive_path.is_dir():
            raise ValueError(f"Archive path is not a directory: {self.archive_path}")

        if self.search.default_limit < 1:
            raise ValueError("search.default_limit must be >= 1")

        if self.search.max_limit < self.search.default_limit:
            raise ValueError("search.max_limit must be >= search.default_limit")

        if self.clip.batch_size < 1:
            raise ValueError("clip.batch_size must be >= 1")

        if self.thumbnails.max_size < 64:
            raise ValueError("thumbnails.max_size must be >= 64")

        if not (0 <= self.thumbnails.quality <= 100):
            raise ValueError("thumbnails.quality must be between 0 and 100")
