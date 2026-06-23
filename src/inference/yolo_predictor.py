from __future__ import annotations

import cv2
import numpy as np

from src.inference.base_predictor import BasePredictor
from src.preprocessing.shape_filter import ShapeFilter


class YOLOPredictor(BasePredictor):
    """YOLO segmentation predictor using ultralytics."""

    def __init__(self, model_path: str, device: str = "cpu", conf: float = 0.18, iou: float = 0.45, min_component_area: int = 20):
        super().__init__(model_path, device)
        self.conf = conf
        self.iou = iou
        self.min_component_area = min_component_area
        self.model = self._load_model()

    def _load_model(self):
        from ultralytics import YOLO
        return YOLO(self.model_path)

    def predict_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        results = self.model.predict(tile_rgb, conf=self.conf, iou=self.iou, verbose=False)
        h, w = tile_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        for result in results:
            if result.masks is None:
                continue
            for seg in result.masks.data:
                seg_np = seg.cpu().numpy().astype(np.float32)
                if seg_np.shape != (h, w):
                    seg_np = cv2.resize(seg_np, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = np.maximum(mask, seg_np)

        mask_bin = (mask >= 0.5).astype(np.uint8) * 255
        mask_bin = ShapeFilter.remove_small_components(mask_bin, self.min_component_area)
        return mask_bin.astype(np.float32) / 255.0
