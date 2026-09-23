#!/usr/bin/env python3
"""Find photos stored sideways or upside down, with no EXIF tag saying so.

For a candidate photo, embed its thumbnail at 0/90/180/270 degrees with
ViT-L/14 and score each against text prompts for "upright" versus "rotated".
The rotation whose embedding reads most upright wins; a photo is flagged only
when that beats the stored orientation by a clear margin.

Validated against a free ground truth: photos whose EXIF orientation says
rotate, whose pre-fix thumbnails (data.clip-b32.bak/thumbnails) are known to
be sideways by exactly that amount.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.indexer import ClipEmbedder

UPRIGHT = [
    "an upright photograph",
    "a photo taken the right way up",
    "a normal photograph with the sky at the top",
]
ROTATED = [
    "a photograph rotated sideways",
    "a photo turned 90 degrees on its side",
    "an upside-down photograph",
]
ROTATIONS = (0, 90, 180, 270)  # degrees counter-clockwise, as PIL rotate()


class RotationDetector:
    def __init__(self, embedder: ClipEmbedder):
        self.e = embedder
        up = np.array([embedder.embed_text(t) for t in UPRIGHT]).mean(0)
        rot = np.array([embedder.embed_text(t) for t in ROTATED]).mean(0)
        self.direction = up / np.linalg.norm(up) - rot / np.linalg.norm(rot)

    @torch.no_grad()
    def scores(self, paths: list[Path]) -> np.ndarray:
        """Return [n, 4] uprightness scores for the four rotations."""
        imgs = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            imgs.extend(im.rotate(r, expand=True) for r in ROTATIONS)
        inputs = self.e.processor(images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.e.device) for k, v in inputs.items()}
        emb = self.e.model.get_image_features(**inputs).cpu().numpy()
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return (emb @ self.direction).reshape(len(paths), 4)


    @torch.no_grad()
    def scores_rotated(self, paths: list[Path]) -> np.ndarray:
        """Return [n, 3] uprightness scores at 90/180/270 only (0 comes from the
        stored embedding, which was computed from the same thumbnail)."""
        imgs = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            imgs.extend(im.rotate(r, expand=True) for r in ROTATIONS[1:])
        inputs = self.e.processor(images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.e.device) for k, v in inputs.items()}
        emb = self.e.model.get_image_features(**inputs).cpu().numpy()
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return (emb @ self.direction).reshape(len(paths), 3)


def verdict(score_row: np.ndarray, margin: float) -> int:
    """Rotation (degrees CCW) to apply, or 0 when not clearly rotated."""
    best = int(score_row.argmax())
    return ROTATIONS[best] if best and score_row[best] - score_row[0] >= margin else 0


if __name__ == "__main__":
    sys.exit("import this module; scan_rotation.py runs the library scan")
