from __future__ import annotations
import cv2
import numpy as np


class ShapeFilter:
    """Removes noise components from binary masks."""

    @staticmethod
    def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        if min_area <= 1:
            return mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out
