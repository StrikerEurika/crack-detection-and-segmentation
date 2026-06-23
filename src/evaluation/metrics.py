from __future__ import annotations

import cv2
import numpy as np


class CrackMetrics:
    """Pixel-level and centerline-buffered evaluation metrics for crack segmentation."""

    @staticmethod
    def compute_pixel_metrics(pred_mask: np.ndarray, target_mask: np.ndarray) -> dict:
        p = pred_mask > 127
        t = target_mask > 127

        intersection = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        pred_sum = p.sum()
        target_sum = t.sum()

        precision = float(intersection / pred_sum) if pred_sum > 0 else 0.0
        recall = float(intersection / target_sum) if target_sum > 0 else 0.0
        f1 = float(2 * intersection / (pred_sum + target_sum)) if (pred_sum + target_sum) > 0 else 0.0
        iou = float(intersection / union) if union > 0 else 0.0

        return {
            "pixel_precision": round(precision, 4),
            "pixel_recall": round(recall, 4),
            "pixel_f1_dice": round(f1, 4),
            "pixel_iou": round(iou, 4),
        }

    @staticmethod
    def compute_buffered_metrics(
        pred_skeleton: np.ndarray, target_skeleton: np.ndarray, tolerance: float = 3.0
    ) -> dict:
        p_coords = np.argwhere(pred_skeleton > 0)
        t_coords = np.argwhere(target_skeleton > 0)

        if len(p_coords) == 0 and len(t_coords) == 0:
            return {"buffered_precision": 1.0, "buffered_recall": 1.0, "buffered_f1": 1.0}
        if len(p_coords) == 0 or len(t_coords) == 0:
            return {"buffered_precision": 0.0, "buffered_recall": 0.0, "buffered_f1": 0.0}

        target_dist = cv2.distanceTransform(
            (target_skeleton == 0).astype(np.uint8), cv2.DIST_L2, 5
        )
        pred_dist = cv2.distanceTransform(
            (pred_skeleton == 0).astype(np.uint8), cv2.DIST_L2, 5
        )

        p_dists = target_dist[pred_skeleton > 0]
        precision = float(np.sum(p_dists <= tolerance) / len(p_dists))

        t_dists = pred_dist[target_skeleton > 0]
        recall = float(np.sum(t_dists <= tolerance) / len(t_dists))

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "buffered_precision": round(precision, 4),
            "buffered_recall": round(recall, 4),
            "buffered_f1": round(f1, 4),
        }
