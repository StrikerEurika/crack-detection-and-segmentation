from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from src.preprocessing.shape_filter import ShapeFilter


class PostProcessor:
    """Post-processing for crack segmentation predictions."""

    @staticmethod
    def binarize_probability_map(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (prob_map >= threshold).astype(np.uint8) * 255

    @staticmethod
    def remove_noise(binary_mask: np.ndarray, min_area: int = 20) -> np.ndarray:
        return ShapeFilter.remove_small_components(binary_mask, min_area)

    @staticmethod
    def skeletonize(binary_mask: np.ndarray) -> np.ndarray:
        return (skeletonize(binary_mask > 0).astype(np.uint8)) * 255

    @staticmethod
    def estimate_dimensions(binary_mask: np.ndarray, skeleton: np.ndarray) -> dict:
        crack_pixels = int(np.sum(binary_mask > 0))
        skeleton_pixels = int(np.sum(skeleton > 0))

        if skeleton_pixels == 0:
            return {"length_pixels": 0, "average_width_pixels": 0.0, "crack_area_pixels": crack_pixels}

        dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        avg_width = float(2.0 * np.mean(dist[skeleton > 0]))

        return {
            "length_pixels": skeleton_pixels,
            "average_width_pixels": round(avg_width, 2),
            "crack_area_pixels": crack_pixels,
        }
