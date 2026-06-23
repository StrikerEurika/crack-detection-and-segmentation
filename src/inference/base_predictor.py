from __future__ import annotations
from abc import ABC, abstractmethod

import cv2
import numpy as np


class BasePredictor(ABC):
    """Base predictor with integrated tiled inference.

    Subclasses implement `predict_tile` for a single tile.  The tiling,
    blending, and stitching logic lives here so subclasses stay focused on
    model I/O.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device

    @abstractmethod
    def predict_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        """Run inference on a single RGB tile, return probability map [0,1]."""
        pass

    def predict_full_image(
        self,
        image_rgb: np.ndarray,
        tile_size: int = 512,
        overlap: int = 64,
        batch_size: int = 4,
        blend_mode: str = "cosine",
    ) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        step = tile_size - overlap

        output = np.zeros((h, w), dtype=np.float64)
        weight_map = np.zeros((h, w), dtype=np.float64)
        blend_window = self._make_blend_window(tile_size, blend_mode)

        tiles = self._crop_tiles(image_rgb, tile_size, step)

        for y1, y2, x1, x2 in tiles:
            tile_h, tile_w = y2 - y1, x2 - y1
            tile = image_rgb[y1:y2, x1:x2]

            prob = self.predict_tile(tile)
            if prob.shape != (tile_h, tile_w):
                prob = cv2.resize(prob, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

            window = blend_window[:tile_h, :tile_w]
            output[y1:y2, x1:x2] += prob * window
            weight_map[y1:y2, x1:x2] += window

        weight_map = np.maximum(weight_map, 1e-8)
        return (output / weight_map).astype(np.float32)

    @staticmethod
    def _crop_tiles(
        image_rgb: np.ndarray, tile_size: int, step: int
    ) -> list[tuple[int, int, int, int]]:
        h, w = image_rgb.shape[:2]
        coords = []
        for y in range(0, h, step):
            y1 = y
            y2 = min(y + tile_size, h)
            if y2 - y1 < tile_size and y != 0:
                y1 = max(0, h - tile_size)
                y2 = h
            for x in range(0, w, step):
                x1 = x
                x2 = min(x + tile_size, w)
                if x2 - x1 < tile_size and x != 0:
                    x1 = max(0, w - tile_size)
                    x2 = w
                coords.append((y1, y2, x1, x2))
            if y2 == h:
                break
        return coords

    @staticmethod
    def _make_blend_window(size: int, mode: str) -> np.ndarray:
        if mode == "cosine":
            t = np.linspace(0, np.pi, size, dtype=np.float64)
            win_1d = 0.5 * (1.0 - np.cos(t))
            return np.outer(win_1d, win_1d)
        center = size / 2.0
        axis = np.minimum(np.arange(size), size - 1 - np.arange(size)) / center
        return np.clip(np.outer(axis, axis), 0.05, 1.0)
