"""Image indexing with CLIP embeddings and FAISS vector search."""

from pathlib import Path
from typing import Optional
from datetime import datetime
import time
import numpy as np
import torch
import faiss
import aiosqlite
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

from .config import Config
from .quality import AestheticScorer
from .metadata import (
    extract_metadata,
    store_image_metadata,
    init_database,
)
from .utils import (
    get_optimal_device,
    generate_thumbnail,
    batch_iterator,
    find_images,
    format_duration,
)


class ClipEmbedder:
    """CLIP model wrapper for generating image and text embeddings."""

    def __init__(self, model_name: str, device: str = "auto"):
        """Initialize CLIP model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use for inference ("auto", "mps", "cuda", "cpu")
        """
        self.device = get_optimal_device(device)
        print(f"Loading CLIP model: {model_name}...")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode

        print(f"✓ CLIP model loaded successfully")

    @torch.no_grad()
    def embed_images(self, image_paths: list[Path]) -> np.ndarray:
        """Generate embeddings for a batch of images.

        Args:
            image_paths: List of paths to images

        Returns:
            NumPy array of normalized embeddings (shape: [batch_size, embedding_dim])
        """
        # Load images
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Warning: Failed to load image {path}: {e}")
                # Use blank image as fallback
                images.append(Image.new("RGB", (224, 224), (0, 0, 0)))

        # Process images
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embeddings
        embeddings = self.model.get_image_features(**inputs)
        embeddings = embeddings.cpu().numpy()

        # Normalize embeddings (required for cosine similarity)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text query.

        Args:
            text: Text query

        Returns:
            NumPy array of normalized embedding (shape: [embedding_dim])
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        embedding = self.model.get_text_features(**inputs)
        embedding = embedding.cpu().numpy()

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        return embedding[0]


class ImageIndexer:
    """Handles indexing of images with CLIP embeddings and FAISS."""

    def __init__(self, config: Config):
        """Initialize indexer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.embedder: Optional[ClipEmbedder] = None
        self.scorer: Optional[AestheticScorer] = None

    def _ensure_scorer(self) -> None:
        if self.scorer is None and self.config.quality.aesthetic_model_path:
            self.scorer = AestheticScorer(Path(self.config.quality.aesthetic_model_path))

    async def index_archive(
        self,
        force: bool = False,
        subpath: Optional[Path] = None,
        rebuild: bool = True,
    ) -> dict:
        """Index all images in the archive.

        Args:
            force: If True, re-index all images. If False, only index new/modified images.
            subpath: If given, only scan this folder (must live under archive_path).
                Relative-folder metadata stays rooted at archive_path, so indexing
                one folder at a time produces identical records to a full scan. This
                lets a driver index then evict the library folder-by-folder, keeping
                peak local disk to roughly one folder rather than the whole library.
            rebuild: If True, rebuild the FAISS index at the end. Set False when
                indexing many folders in a loop and rebuilding once afterwards.

        Returns:
            Dictionary with indexing statistics
        """
        start_time = time.time()

        # Initialize embedder
        if self.embedder is None:
            self.embedder = ClipEmbedder(
                self.config.clip.model_name, self.config.clip.device
            )
        self._ensure_scorer()

        # Initialize database
        await init_database(self.config.db_path)

        # Find all images
        scan_root = subpath if subpath is not None else self.config.archive_path
        print(f"Scanning archive: {scan_root}")
        image_paths = find_images(scan_root)
        print(f"Found {len(image_paths)} images")

        if not image_paths:
            return {
                "total_images": 0,
                "indexed": 0,
                "skipped": 0,
                "errors": 0,
                "duration_seconds": 0,
            }

        # Filter to new/modified images unless force=True
        if not force:
            image_paths = await self._filter_unindexed(image_paths)
            print(f"New/modified images to index: {len(image_paths)}")

        if not image_paths:
            print("No new images to index")
            return {
                "total_images": len(image_paths),
                "indexed": 0,
                "skipped": 0,
                "errors": 0,
                "duration_seconds": time.time() - start_time,
            }

        # Process images in batches
        batch_size = self.config.clip.batch_size
        indexed = 0
        errors = 0
        error_details = []

        print(f"Indexing images (batch size: {batch_size})...")

        with tqdm(total=len(image_paths), desc="Indexing", unit="img") as pbar:
            for batch in batch_iterator(image_paths, batch_size):
                try:
                    batch_indexed = await self._index_batch(batch)
                    indexed += batch_indexed
                except Exception as e:
                    errors += len(batch)
                    error_details.append(str(e))
                    print(f"\nError processing batch: {e}")

                pbar.update(len(batch))

        # Rebuild FAISS index
        if rebuild:
            print("Building FAISS index...")
            await self._rebuild_faiss_index()

        duration = time.time() - start_time

        print(f"\n✓ Indexing complete!")
        print(f"  Indexed: {indexed} images")
        print(f"  Errors: {errors} images")
        print(f"  Duration: {format_duration(duration)}")

        return {
            "total_images": len(image_paths),
            "indexed": indexed,
            "skipped": 0,
            "errors": errors,
            "error_details": error_details[:10],  # Limit error details
            "duration_seconds": duration,
        }

    async def _filter_unindexed(self, paths: list[Path]) -> list[Path]:
        """Filter to only new or modified images, WITHOUT reading file contents.

        Change detection uses file size + modification time, both of which are
        available from the filesystem placeholder metadata of an online-only
        (OneDrive Files On-Demand) file. This matters a lot: the previous
        implementation hashed the first 8 KB of every file, and reading even one
        byte of a placeholder forces the cloud provider to download the whole
        file. That turned every incremental reindex into a full re-download of
        the entire library. We now only touch (and therefore download) files
        that are genuinely new or whose size/mtime changed; the content hash is
        still computed later, inside extract_metadata, for exactly those files
        we are about to embed anyway.

        Args:
            paths: List of image paths

        Returns:
            List of paths that need to be indexed
        """
        # Pull stored size + mtime for everything already indexed, in one query.
        stored: dict[str, tuple] = {}
        async with aiosqlite.connect(self.config.db_path) as db:
            async with db.execute(
                "SELECT filepath, file_size, file_modified FROM images"
            ) as cursor:
                async for row in cursor:
                    stored[row[0]] = (row[1], row[2])

        unindexed = []
        for path in paths:
            try:
                st = path.stat()  # metadata only - does NOT materialise a placeholder
            except OSError:
                continue
            current = (st.st_size, datetime.fromtimestamp(st.st_mtime).isoformat())
            prev = stored.get(str(path))
            if prev is None or prev[0] != current[0] or prev[1] != current[1]:
                unindexed.append(path)

        return unindexed

    async def _index_batch(self, batch: list[Path]) -> int:
        """Index a batch of images.

        Args:
            batch: List of image paths

        Returns:
            Number of successfully indexed images
        """
        # Extract metadata
        metadata_list = []
        valid_paths = []

        for path in batch:
            try:
                metadata = extract_metadata(path, self.config.archive_path)
                metadata_list.append(metadata)
                valid_paths.append(path)
            except Exception as e:
                print(f"\nWarning: Failed to extract metadata from {path}: {e}")
                continue

        if not valid_paths:
            return 0

        # Generate embeddings
        embeddings = self.embedder.embed_images(valid_paths)
        if self.scorer is not None:
            for metadata, score in zip(metadata_list, self.scorer.score(embeddings)):
                metadata.aesthetic_score = float(score)

        # Generate thumbnails and store in database
        async with aiosqlite.connect(self.config.db_path) as db:
            for i, (metadata, embedding, path) in enumerate(
                zip(metadata_list, embeddings, valid_paths)
            ):
                try:
                    # Generate thumbnail
                    if self.config.thumbnails.enabled:
                        thumbnail_path = (
                            self.config.thumbnails_dir / f"{metadata.image_id}.jpg"
                        )
                        try:
                            generate_thumbnail(
                                path,
                                thumbnail_path,
                                self.config.thumbnails.max_size,
                                self.config.thumbnails.quality,
                            )
                        except Exception as e:
                            print(f"\nWarning: Failed to generate thumbnail for {path}: {e}")

                    # Convert embedding to bytes
                    embedding_bytes = embedding.astype(np.float32).tobytes()

                    # Store metadata and embedding
                    await store_image_metadata(db, metadata, embedding_bytes)

                except Exception as e:
                    print(f"\nWarning: Failed to store metadata for {path}: {e}")

            await db.commit()

        return len(valid_paths)

    async def _rebuild_faiss_index(self) -> None:
        """Rebuild FAISS index from all embeddings in database."""
        # Load all embeddings from database
        async with aiosqlite.connect(self.config.db_path) as db:
            cursor = await db.execute(
                "SELECT image_id, embedding_vector FROM images WHERE embedding_vector IS NOT NULL ORDER BY image_id"
            )
            rows = await cursor.fetchall()

        if not rows:
            print("Warning: No embeddings found in database")
            return

        # Convert to numpy array. The searcher maps FAISS positions back to
        # image_ids with the same query, so both must select identical rows.
        embeddings = np.array(
            [np.frombuffer(row[1], dtype=np.float32) for row in rows]
        )

        # Create FAISS index
        dimension = embeddings.shape[1]
        num_vectors = len(embeddings)

        # Use IndexFlatL2 for small/medium datasets, IndexIVFFlat for very large.
        # IndexFlatL2 is exact and needs no training; at this dimension it searches
        # hundreds of thousands of vectors in milliseconds, so it comfortably covers
        # the whole HT library (~26k images). The IVF path requires a train() step
        # that hits a faiss/OpenMP segfault on macOS, so only use it when the dataset
        # is genuinely large enough to justify the approximation.
        if num_vectors < 200000:
            index = faiss.IndexFlatL2(dimension)
        else:
            # Use IVF for larger datasets
            nlist = min(100, num_vectors // 100)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

            # Train index
            print(f"Training FAISS index with {num_vectors} vectors...")
            index.train(embeddings)

        # Add vectors to index
        index.add(embeddings)

        # Save index
        faiss.write_index(index, str(self.config.index_path))
        print(f"✓ FAISS index saved: {num_vectors} vectors, dimension {dimension}")


async def get_existing_image_ids(db_path: Path) -> set[str]:
    """Get set of image IDs already in database.

    Args:
        db_path: Path to database

    Returns:
        Set of image IDs
    """
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT image_id FROM images")
        rows = await cursor.fetchall()
        return {row[0] for row in rows}
