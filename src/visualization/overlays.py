from __future__ import annotations

import cv2
import numpy as np


class Visualizer:
    """Visualization utilities for crack detection results."""

    @staticmethod
    def draw_mask_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        color: tuple = (255, 0, 0),
        alpha: float = 0.4,
    ) -> np.ndarray:
        bin_mask = mask > 127
        overlay = image.copy()
        overlay[bin_mask] = color
        return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

    @staticmethod
    def draw_error_analysis_overlay(
        image: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        gt_bin = gt_mask > 127
        pred_bin = pred_mask > 127

        tp = np.logical_and(gt_bin, pred_bin)
        fp = np.logical_and(~gt_bin, pred_bin)
        fn = np.logical_and(gt_bin, ~pred_bin)

        color_layer = image.copy()
        color_layer[tp] = [0, 255, 0]
        color_layer[fp] = [255, 0, 0]
        color_layer[fn] = [0, 0, 255]

        changed = tp | fp | fn
        output = image.copy()
        output[changed] = cv2.addWeighted(color_layer, alpha, image, 1.0 - alpha, 0)[changed]
        return output
