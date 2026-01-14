"""Semantic image search with CLIP and FAISS."""

from pathlib import Path
from typing import Optional
import numpy as np
import faiss
import aiosqlite

from .config import Config
from .indexer import ClipEmbedder


class ImageSearcher:
    """Handles semantic image search using CLIP embeddings and FAISS."""

    def __init__(self, config: Config):
        """Initialize searcher.

        Args:
            config: Configuration object
        """
        self.config = config
        self.embedder: Optional[ClipEmbedder] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.image_id_mapping: list[str] = []

    async def search(
        self,
        query: str,
        limit: int = 10,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        folder_pattern: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[dict]:
        """Search images using natural language query with filters.

        Args:
            query: Natural language search query
            limit: Maximum number of results
            date_from: Filter by date taken (ISO format)
            date_to: Filter by date taken (ISO format)
            folder_pattern: Filter by folder path pattern
            tags: Filter by tags

        Returns:
            List of matching images with metadata and similarity scores
        """
        # Ensure components are loaded
        await self._ensure_loaded()

        # Validate limit
        if limit > self.config.search.max_limit:
            limit = self.config.search.max_limit

        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)

        # Search FAISS index
        # Over-fetch to allow for filtering
        k = min(limit * 10, len(self.image_id_mapping))
        if k == 0:
            return []

        distances, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype(np.float32), k
        )

        # Convert L2 distances to similarity scores
        # For normalized vectors: similarity = 1 - (L2_distance^2 / 2)
        similarities = 1 - (distances[0] ** 2 / 2)

        # Get candidate image IDs
        candidate_ids = [self.image_id_mapping[i] for i in indices[0]]

        # Apply filters and fetch metadata
        results = await self._filter_and_fetch(
            candidate_ids, similarities, date_from, date_to, folder_pattern, tags, limit
        )

        return results

    async def get_image_info(self, image_id: str) -> dict:
        """Get detailed metadata for a specific image.

        Args:
            image_id: Unique identifier for the image

        Returns:
            Complete metadata for the image

        Raises:
            ValueError: If image not found
        """
        async with aiosqlite.connect(self.config.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM images WHERE image_id = ?", (image_id,)
            )
            row = await cursor.fetchone()

            if row is None:
                raise ValueError(f"Image not found: {image_id}")

            # Fetch associated tags
            cursor = await db.execute(
                """
                SELECT t.tag_name FROM tags t
                JOIN image_tags it ON t.tag_id = it.tag_id
                WHERE it.image_id = ?
            """,
                (image_id,),
            )
            tag_rows = await cursor.fetchall()
            tags = [r["tag_name"] for r in tag_rows]

            # Build result
            result = {
                "image_id": row["image_id"],
                "filepath": row["filepath"],
                "filename": row["filename"],
                "folder": row["folder"],
                "file_size": row["file_size"],
                "width": row["width"],
                "height": row["height"],
                "date_taken": row["date_taken"],
                "camera_make": row["camera_make"],
                "camera_model": row["camera_model"],
                "lens_model": row["lens_model"],
                "focal_length": row["focal_length"],
                "aperture": row["aperture"],
                "iso": row["iso"],
                "shutter_speed": row["shutter_speed"],
                "gps_latitude": row["gps_latitude"],
                "gps_longitude": row["gps_longitude"],
                "orientation": row["orientation"],
                "file_modified": row["file_modified"],
                "file_created": row["file_created"],
                "indexed_at": row["indexed_at"],
                "tags": tags,
            }

            # Add thumbnail path if available
            if self.config.thumbnails.enabled:
                thumbnail_path = self.config.thumbnails_dir / f"{image_id}.jpg"
                if thumbnail_path.exists():
                    result["thumbnail_path"] = str(thumbnail_path)

            return result

    async def get_archive_stats(self) -> dict:
        """Get statistics about the indexed archive.

        Returns:
            Dictionary with archive statistics
        """
        async with aiosqlite.connect(self.config.db_path) as db:
            # Total images
            cursor = await db.execute("SELECT COUNT(*) FROM images")
            total = (await cursor.fetchone())[0]

            # Date range
            cursor = await db.execute(
                "SELECT MIN(date_taken), MAX(date_taken) FROM images WHERE date_taken IS NOT NULL"
            )
            date_range = await cursor.fetchone()

            # Storage size
            cursor = await db.execute("SELECT SUM(file_size) FROM images")
            total_size = (await cursor.fetchone())[0] or 0

            # Camera models
            cursor = await db.execute(
                """
                SELECT camera_model, COUNT(*) as count
                FROM images
                WHERE camera_model IS NOT NULL
                GROUP BY camera_model
                ORDER BY count DESC
                LIMIT 5
            """
            )
            camera_stats = await cursor.fetchall()

            # Folder distribution
            cursor = await db.execute(
                """
                SELECT folder, COUNT(*) as count
                FROM images
                GROUP BY folder
                ORDER BY count DESC
                LIMIT 10
            """
            )
            folder_stats = await cursor.fetchall()

            return {
                "total_images": total,
                "date_range": {
                    "earliest": date_range[0],
                    "latest": date_range[1],
                },
                "total_size_bytes": total_size,
                "total_size_gb": round(total_size / (1024**3), 2),
                "top_cameras": [{"model": c[0], "count": c[1]} for c in camera_stats],
                "top_folders": [{"folder": f[0], "count": f[1]} for f in folder_stats],
            }

    async def _ensure_loaded(self) -> None:
        """Lazy load CLIP model and FAISS index."""
        # Load CLIP embedder
        if self.embedder is None:
            self.embedder = ClipEmbedder(
                self.config.clip.model_name, self.config.clip.device
            )

        # Load FAISS index
        if self.faiss_index is None:
            if not self.config.index_path.exists():
                raise FileNotFoundError(
                    "FAISS index not found. Please run indexing first."
                )

            self.faiss_index = faiss.read_index(str(self.config.index_path))

            # Load image ID mapping (must match order in FAISS index)
            async with aiosqlite.connect(self.config.db_path) as db:
                cursor = await db.execute("SELECT image_id FROM images ORDER BY image_id")
                rows = await cursor.fetchall()
                self.image_id_mapping = [row[0] for row in rows]

            print(f"✓ Loaded FAISS index with {len(self.image_id_mapping)} images")

    async def _filter_and_fetch(
        self,
        candidate_ids: list[str],
        similarities: np.ndarray,
        date_from: Optional[str],
        date_to: Optional[str],
        folder_pattern: Optional[str],
        tags: Optional[list[str]],
        limit: int,
    ) -> list[dict]:
        """Apply filters and fetch full metadata.

        Args:
            candidate_ids: List of candidate image IDs from FAISS search
            similarities: Similarity scores for candidates
            date_from: Filter by date taken (ISO format)
            date_to: Filter by date taken (ISO format)
            folder_pattern: Filter by folder path pattern
            tags: Filter by tags
            limit: Maximum number of results

        Returns:
            List of filtered results with metadata
        """
        # Build SQL query with filters
        query = "SELECT * FROM images WHERE image_id IN ({})".format(
            ",".join("?" * len(candidate_ids))
        )
        conditions = []
        params = list(candidate_ids)

        if date_from:
            conditions.append("date_taken >= ?")
            params.append(date_from)

        if date_to:
            conditions.append("date_taken <= ?")
            params.append(date_to)

        if folder_pattern:
            conditions.append("folder LIKE ?")
            params.append(f"%{folder_pattern}%")

        if conditions:
            query += " AND " + " AND ".join(conditions)

        # Handle tag filtering separately (requires JOIN)
        if tags:
            query = f"""
            SELECT DISTINCT i.* FROM images i
            JOIN image_tags it ON i.image_id = it.image_id
            JOIN tags t ON it.tag_id = t.tag_id
            WHERE i.image_id IN ({','.join('?' * len(candidate_ids))})
            AND t.tag_name IN ({','.join('?' * len(tags))})
            """
            params = list(candidate_ids) + tags

            if date_from:
                query += " AND i.date_taken >= ?"
                params.append(date_from)

            if date_to:
                query += " AND i.date_taken <= ?"
                params.append(date_to)

            if folder_pattern:
                query += " AND i.folder LIKE ?"
                params.append(f"%{folder_pattern}%")

        # Fetch results
        async with aiosqlite.connect(self.config.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        # Match with similarity scores
        id_to_similarity = dict(zip(candidate_ids, similarities))
        results = []

        for row in rows:
            image_id = row["image_id"]
            similarity = float(id_to_similarity[image_id])

            # Apply similarity threshold
            if similarity < self.config.search.similarity_threshold:
                continue

            result = {
                "image_id": image_id,
                "filepath": row["filepath"],
                "similarity": similarity,
                "filename": row["filename"],
                "folder": row["folder"],
                "date_taken": row["date_taken"],
                "camera_model": row["camera_model"],
                "width": row["width"],
                "height": row["height"],
            }

            # Add thumbnail path if available
            if self.config.thumbnails.enabled:
                thumbnail_path = self.config.thumbnails_dir / f"{image_id}.jpg"
                if thumbnail_path.exists():
                    result["thumbnail_path"] = str(thumbnail_path)

            results.append(result)

        # Sort by similarity (descending) and limit
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]
