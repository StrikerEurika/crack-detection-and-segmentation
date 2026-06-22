import torch
import numpy as np
from typing import List, Optional
from src.inference.base_predictor import BasePredictor
from src.preprocessing.marker_suppression import MarkerSuppressor

class UnetPredictor(BasePredictor):
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
        inpaint_markers: bool = False,
        marker_saturation_threshold: int = 85,
        marker_value_threshold: int = 55,
        inpaint_radius: int = 5,
    ):
        """U-Net specific predictor.
        
        Args:
            model: PyTorch U-Net model instance.
            device: Device to run inference on.
            mean: Normalization channel means.
            std: Normalization channel standard deviations.
            inpaint_markers: Whether to pre-inpaint colored markers.
            marker_saturation_threshold: HSV saturation threshold for markers.
            marker_value_threshold: HSV value threshold for markers.
            inpaint_radius: Inpainting radius.
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.inpaint_markers = inpaint_markers
        self.marker_saturation_threshold = marker_saturation_threshold
        self.marker_value_threshold = marker_value_threshold
        self.inpaint_radius = inpaint_radius

    def preprocess_tile(self, tile: np.ndarray) -> torch.Tensor:
        """Preprocesses a single tile (inpainting if requested, normalizing, converting to tensor)."""
        tile_prep = tile
        if self.inpaint_markers:
            tile_prep = MarkerSuppressor.preprocess_tile(
                tile,
                inpaint=True,
                saturation_threshold=self.marker_saturation_threshold,
                value_threshold=self.marker_value_threshold,
                inpaint_radius=self.inpaint_radius,
            )

        tile_float = tile_prep.astype(np.float32) / 255.0
        tile_normalized = (tile_float - self.mean) / self.std
        tile_tensor = torch.from_numpy(tile_normalized.transpose(2, 0, 1)).unsqueeze(0).float()
        return tile_tensor.to(self.device)

    def predict_tile_batch(self, tiles: List[np.ndarray], batch_size: int = 4) -> List[np.ndarray]:
        """Predicts a batch of tile images using U-Net."""
        preds = []
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i+batch_size]
            batch_tensors = [self.preprocess_tile(t) for t in batch_tiles]
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
                
            for prob in probs:
                preds.append(np.squeeze(prob, axis=0)) # Shape (H, W)
                
        return preds
