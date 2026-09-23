"""Semantic image search with CLIP and FAISS."""

import re
from pathlib import Path
from typing import Optional
import numpy as np
import faiss
import aiosqlite

from .config import Config
from .indexer import ClipEmbedder
from .sources import SOURCE_CLASSES


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
        min_quality: Optional[float] = None,
        min_long_edge: Optional[int] = None,
        quality_weight: float = 0.0,
        dedupe: bool = True,
        sources: Optional[list[str]] = None,
        archive: str = "auto",
        taste_weight: float = 1.0,
    ) -> list[dict]:
        """Search images using natural language query with filters.

        Args:
            query: Natural language search query
            limit: Maximum number of results
            date_from: Filter by date taken (ISO format)
            date_to: Filter by date taken (ISO format)
            folder_pattern: Filter by folder path pattern
            tags: Filter by tags
            min_quality: Drop images whose aesthetic score is below this
            min_long_edge: Drop images whose original's long edge is below this (px)
            quality_weight: 0 ranks by similarity alone. Above 0, blends in the
                aesthetic score (see _rank_score)
            dedupe: Collapse copies of the same photo (re-exports, rescans,
                re-crops) into one result, listing the others in duplicate_ids
            sources: Only return these source classes (pro, archive, amateur,
                unknown). None applies the default source ranking instead.
            archive: "auto" (archive film scans rank freely for historical
                queries, otherwise at most ARCHIVE_CAP per page), "include"
                (rank freely), "exclude", or "only"

            taste_weight: How much the learned HT taste score reorders close
                matches. 1.0 (default) is gentle; 0 ranks by relevance alone.

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
        # Over-fetch to allow for filtering. Quality filters and re-ranking can
        # discard most of the nearest neighbours, so fetch much deeper for them.
        # An exact flat index over ~26k vectors makes this cheap.
        depth = 10
        if (min_quality is not None or min_long_edge is not None or quality_weight > 0
                or sources or archive in ("only", "exclude")):
            depth = 50
        k = min(limit * depth, len(self.image_id_mapping))
        if k == 0:
            return []

        distances, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype(np.float32), k
        )

        # Convert L2 distances to cosine similarity. IndexFlatL2 returns SQUARED
        # L2 distances, so for normalised vectors cosine = 1 - d / 2. (This used
        # to square d again. Ranking was unaffected, since the map is monotonic,
        # but reported scores were wrong, and with ViT-L/14's lower cosines they
        # went negative and similarity_threshold 0.0 silently dropped results.)
        similarities = 1 - distances[0] / 2

        # Get candidate image IDs
        candidate_ids = [self.image_id_mapping[i] for i in indices[0]]

        # Apply filters and fetch metadata
        results = await self._filter_and_fetch(
            candidate_ids,
            similarities,
            date_from,
            date_to,
            folder_pattern,
            tags,
            limit,
            min_quality=min_quality,
            min_long_edge=min_long_edge,
            quality_weight=quality_weight,
            dedupe=dedupe,
            sources=sources,
            archive_mode=_archive_mode(archive, query),
            taste_weight=taste_weight,
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
                # Must select exactly the rows _rebuild_faiss_index embedded, in
                # the same order, or FAISS positions map to the wrong image.
                cursor = await db.execute(
                    "SELECT image_id FROM images WHERE embedding_vector IS NOT NULL ORDER BY image_id"
                )
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
        min_quality: Optional[float] = None,
        min_long_edge: Optional[int] = None,
        quality_weight: float = 0.0,
        dedupe: bool = True,
        sources: Optional[list[str]] = None,
        archive_mode: str = "capped",
        taste_weight: float = 1.0,
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

        allowed = list(sources) if sources else None
        if archive_mode == "only":
            allowed = ["archive"]
        elif archive_mode == "exclude":
            allowed = [c for c in (allowed or SOURCE_CLASSES) if c != "archive"]
        if allowed:
            conditions.append(f"source_class IN ({','.join('?' * len(allowed))})")
            params.extend(allowed)

        # Unscored or unmeasured images fail a quality floor rather than pass it.
        if min_quality is not None:
            conditions.append("aesthetic_score >= ?")
            params.append(min_quality)

        if min_long_edge is not None:
            conditions.append("long_edge_px >= ?")
            params.append(min_long_edge)

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

            if allowed:
                query += f" AND i.source_class IN ({','.join('?' * len(allowed))})"
                params.extend(allowed)

            if min_quality is not None:
                query += " AND i.aesthetic_score >= ?"
                params.append(min_quality)

            if min_long_edge is not None:
                query += " AND i.long_edge_px >= ?"
                params.append(min_long_edge)

        # Fetch results
        async with aiosqlite.connect(self.config.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        # Match with similarity scores
        id_to_similarity = dict(zip(candidate_ids, similarities))
        results = []
        vectors = {}

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
                # Older data dirs predate these columns; keep them searchable so
                # rolling back to one never breaks the server.
                "long_edge_px": row["long_edge_px"] if "long_edge_px" in row.keys() else None,
                "aesthetic_score": None,
            }
            aesthetic = row["aesthetic_score"] if "aesthetic_score" in row.keys() else None
            if aesthetic is not None:
                result["aesthetic_score"] = round(aesthetic, 2)
            source = row["source_class"] if "source_class" in row.keys() else None
            result["source_class"] = source
            prior = 0.0 if sources else SOURCE_PRIOR.get(source, 0.0)
            if source == "archive" and archive_mode in ("include", "only"):
                prior = 0.0
            taste = row["taste_score"] if "taste_score" in row.keys() else None
            result["taste_score"] = round(taste, 2) if taste is not None else None
            taste_term = TASTE_SCALE * taste_weight * taste if taste is not None else 0.0
            result["rank_score"] = (
                _rank_score(similarity, aesthetic, quality_weight) + prior + taste_term
            )
            if row["embedding_vector"] is not None:
                vectors[image_id] = np.frombuffer(row["embedding_vector"], dtype=np.float32)

            # Add thumbnail path if available
            if self.config.thumbnails.enabled:
                thumbnail_path = self.config.thumbnails_dir / f"{image_id}.jpg"
                if thumbnail_path.exists():
                    result["thumbnail_path"] = str(thumbnail_path)

            results.append(result)

        # Sort by rank (similarity alone when quality_weight is 0) and limit
        results.sort(key=lambda x: x["rank_score"], reverse=True)
        if dedupe:
            results = _collapse_duplicates(results, vectors, len(results))
        if archive_mode == "capped" and not sources:
            results = _cap_archive(results, ARCHIVE_CAP)
        return results[:limit]


# PLACEHOLDERS, not calibrated: the LAION score tracked Casey's labels at only
# rho 0.14 (23 Sep 2026), so quality_weight defaults to 0 until a better score exists.
QUALITY_CENTRE = 5.0
QUALITY_SCALE = 0.02


def _rank_score(
    similarity: float, aesthetic_score: Optional[float], quality_weight: float
) -> float:
    """Blend CLIP similarity with the aesthetic score.

    CLIP text-image similarities for good matches sit in a narrow band (roughly
    0.2 to 0.35), while aesthetic scores run about 1 to 10. Each aesthetic point
    above QUALITY_CENTRE is worth QUALITY_SCALE of similarity at weight 1.0, so
    quality reorders close matches without dragging in off-topic photos.
    Unscored images are treated as exactly average.
    """
    if quality_weight <= 0 or aesthetic_score is None:
        return similarity
    return similarity + quality_weight * QUALITY_SCALE * (aesthetic_score - QUALITY_CENTRE)


# Cosine at or above which two library photos are treated as the same picture.
# Measured on ViT-L/14, 23 Sep 2026: byte-identical copies score 1.000 and
# rescans or re-exports 0.996-0.998, while the closest DIFFERENT photos in the
# test queries (same classroom, same peak) topped out around 0.92.
DUPLICATE_COSINE = 0.97


def _collapse_duplicates(results: list[dict], vectors: dict, limit: int) -> list[dict]:
    """Keep the best-ranked copy of each photo; note the others on it."""
    kept: list[dict] = []
    kept_vecs: list[np.ndarray] = []
    for r in results:
        v = vectors.get(r["image_id"])
        if v is not None and kept_vecs:
            sims = np.array(kept_vecs) @ v
            j = int(sims.argmax())
            if sims[j] >= DUPLICATE_COSINE:
                twin = kept[[i for i, k in enumerate(kept) if k["_has_vec"]][j]]
                twin.setdefault("duplicate_ids", []).append(r["image_id"])
                continue
        r["_has_vec"] = v is not None
        kept.append(r)
        if v is not None:
            kept_vecs.append(v)
        if len(kept) >= limit:
            break
    for r in kept:
        r.pop("_has_vec", None)
    return kept


# Default source ranking, in cosine-similarity units. Top matches for a query
# typically sit within about 0.01-0.03 of each other, so a pro photo of the
# same subject outranks an amateur one, while an amateur photo that is the only
# good match for a query still appears. Set from Casey's sourcing hierarchy,
# 23 Sep 2026 (see src/sources.py).
SOURCE_PRIOR = {"pro": 0.0, "unknown": -0.01, "archive": -0.01, "amateur": -0.03}

# Learned HT taste score (train_taste.py; z-scored across the library). One
# standard deviation is worth this much cosine similarity at taste_weight 1.0:
# enough to reorder close matches, never enough to lift an off-topic photo.
# Trained on Casey's round-2 pairwise picks, 23 Sep 2026: held-out rho 0.51
# on his 30 tier labels (LAION: 0.14).
TASTE_SCALE = 0.005

# Archive film scans are "magical when used sparingly": outside historical
# queries, at most this many per results page.
ARCHIVE_CAP = 2

_HISTORICAL = re.compile(
    r"\b(hillary|sir ed|expedition|historic|history|archive|archival|vintage|old photo|"
    r"founding|founder|195\d|196\d|197\d|1950s|1960s|1970s|1980s|black and white|b&w)\b",
    re.IGNORECASE,
)


def _archive_mode(archive: str, query: str) -> str:
    if archive in ("include", "exclude", "only"):
        return archive
    return "include" if _HISTORICAL.search(query or "") else "capped"


def _cap_archive(results: list[dict], cap: int) -> list[dict]:
    kept, seen = [], 0
    for r in results:
        if r.get("source_class") == "archive":
            if seen >= cap:
                continue
            seen += 1
        kept.append(r)
    return kept
