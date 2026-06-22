import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from src.inference.base_predictor import BasePredictor
from src.preprocessing.marker_suppression import MarkerSuppressor

class YoloPredictor(BasePredictor):
    def __init__(
        self,
        model, # YOLO model instance from ultralytics
        conf: float = 0.18,
        iou: float = 0.45,
        min_component_area: int = 10,
        disable_color_marker_suppression: bool = False,
        marker_saturation_threshold: int = 85,
        marker_value_threshold: int = 55,
        frame_side_coverage_threshold: float = 0.35,
        roundness_threshold: float = 0.58,
    ):
        """YOLO-specific predictor.
        
        Args:
            model: YOLO model instance.
            conf: YOLO confidence threshold.
            iou: YOLO NMS IoU threshold.
            min_component_area: Discard components smaller than this area.
            disable_color_marker_suppression: Disable saturation-based marker suppression.
            marker_saturation_threshold: HSV saturation threshold for markers.
            marker_value_threshold: HSV value threshold for markers.
            frame_side_coverage_threshold: Border coverage threshold for frame-like markers.
            roundness_threshold: Circularity threshold for round markers.
        """
        self.model = model
        self.conf = conf
        self.iou = iou
        self.min_component_area = min_component_area
        self.disable_color_marker_suppression = disable_color_marker_suppression
        self.marker_saturation_threshold = marker_saturation_threshold
        self.marker_value_threshold = marker_value_threshold
        self.frame_side_coverage_threshold = frame_side_coverage_threshold
        self.roundness_threshold = roundness_threshold

    def filter_instance_mask(self, mask_prob: np.ndarray, tile_rgb: np.ndarray) -> np.ndarray:
        """Applies marker and noise filters on a prediction mask tile."""
        return MarkerSuppressor.filter_mask(
            mask_prob,
            tile_rgb,
            min_component_area=self.min_component_area,
            disable_color_suppression=self.disable_color_marker_suppression,
            saturation_threshold=self.marker_saturation_threshold,
            value_threshold=self.marker_value_threshold,
            frame_side_coverage_threshold=self.frame_side_coverage_threshold,
            roundness_threshold=self.roundness_threshold,
        )

    def predict_tile_batch(self, tiles: List[np.ndarray], batch_size: int = 4) -> List[np.ndarray]:
        """Predicts a batch of tile images using YOLO and filters/cleans the predicted masks."""
        preds = []
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i+batch_size]
            results = self.model(batch_tiles, conf=self.conf, iou=self.iou, verbose=False)
            
            for tile_rgb, result in zip(batch_tiles, results):
                tile_h, tile_w = tile_rgb.shape[:2]
                tile_prob = np.zeros((tile_h, tile_w), dtype=np.float32)
                
                if result.masks is not None:
                    for mask_tensor in result.masks.data:
                        mask_prob = mask_tensor.cpu().numpy().astype(np.float32)
                        mh, mw = mask_prob.shape
                        if mh != tile_h or mw != tile_w:
                            mask_prob = cv2.resize(mask_prob, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
                        
                        cleaned = self.filter_instance_mask(mask_prob, tile_rgb)
                        tile_prob = np.maximum(tile_prob, cleaned)
                        
                preds.append(tile_prob)
                
        return preds
