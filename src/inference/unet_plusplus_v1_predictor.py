import torch
import numpy as np
from typing import List, Optional
from src.inference.base_predictor import BasePredictor

class UnetPlusPlusV1Predictor(BasePredictor):
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """U-Net++ specific predictor for v1 dataset (without standard normalization).
        
        Args:
            model: PyTorch U-Net++ model instance.
            device: Device to run inference on.
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def preprocess_tile(self, tile: np.ndarray) -> torch.Tensor:
        """Preprocesses a single tile.
        
        Note: The model was trained without standard ImageNet normalization.
        It expects image values in range [0, 255] transposed to (C, H, W).
        """
        # tile is (H, W, C), convert to float32 (keeping range 0-255)
        tile_float = tile.astype(np.float32)
        # Permute to (C, H, W) and add batch dimension
        tile_tensor = torch.from_numpy(tile_float.transpose(2, 0, 1)).unsqueeze(0)
        return tile_tensor.to(self.device)

    def predict_tile_batch(self, tiles: List[np.ndarray], batch_size: int = 4) -> List[np.ndarray]:
        """Predicts a batch of tile images using U-Net++."""
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
