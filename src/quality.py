"""Per-photo aesthetic quality score.

Uses the LAION "improved aesthetic predictor": a small MLP trained on top of
CLIP ViT-L/14 image embeddings to predict a 1-10 human aesthetic rating
(weights: sac+logos+ava1-l14-linearMSE.pth, from
github.com/christophschuhmann/improved-aesthetic-predictor).

It only works on ViT-L/14 embeddings (768-d, L2-normalised), which is exactly
what the indexer stores once the model is ViT-L/14, so scoring costs one tiny
matrix multiply per photo on top of the embedding we already compute.

The score is a heuristic trained on general web aesthetics. It ranks and
filters; it never deletes or hides a photo.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

AESTHETIC_EMBED_DIM = 768


class _AestheticMLP(nn.Module):
    """Architecture must match the published checkpoint's state-dict keys."""

    def __init__(self, input_size: int = AESTHETIC_EMBED_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AestheticScorer:
    """Scores L2-normalised ViT-L/14 image embeddings on a ~1-10 scale."""

    def __init__(self, weights_path: Path):
        weights_path = Path(weights_path).expanduser()
        if not weights_path.exists():
            raise FileNotFoundError(f"Aesthetic model weights not found: {weights_path}")
        self.model = _AestheticMLP()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return one score per row. Embeddings must be 768-d and normalised."""
        if embeddings.ndim != 2 or embeddings.shape[1] != AESTHETIC_EMBED_DIM:
            raise ValueError(
                f"Aesthetic scorer needs {AESTHETIC_EMBED_DIM}-d ViT-L/14 embeddings, "
                f"got shape {embeddings.shape}"
            )
        x = torch.from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32))
        return self.model(x).squeeze(1).numpy()
