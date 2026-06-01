import torch
import numpy as np
from typing import Dict, Any, List, Optional
from src.tiling.tile_generator import TileGenerator
from src.tiling.tile_merger import TileMerger

class CrackPredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225)
    ):
        """Initializes the CrackPredictor.
        
        Args:
            model: The trained PyTorch crack segmentation model.
            device: Device to run inference on ('cuda' or 'cpu').
            mean: Mean for ImageNet normalization.
            std: Std dev for ImageNet normalization.
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def preprocess_tile(self, tile: np.ndarray) -> torch.Tensor:
        """Preprocesses a single tile image (H, W, 3) for the model."""
        # Convert to float32 [0.0, 1.0]
        tile_float = tile.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        tile_normalized = (tile_float - self.mean) / self.std
        
        # HWC -> CHW
        tile_tensor = torch.from_numpy(tile_normalized.transpose(2, 0, 1)).unsqueeze(0).float()
        return tile_tensor.to(self.device)

    def predict_tiles_batch(self, tiles: List[np.ndarray], batch_size: int = 4) -> List[np.ndarray]:
        """Predicts a batch of tile images.
        
        Returns:
            A list of probability maps of shape (H, W).
        """
        preds = []
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i+batch_size]
            batch_tensors = [self.preprocess_tile(t) for t in batch_tiles]
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
                
            for prob in probs:
                preds.append(np.squeeze(prob, axis=0)) # (H, W)
                
        return preds

    def predict_full_image(
        self,
        image: np.ndarray,
        tile_size: int = 512,
        overlap: int = 64,
        batch_size: int = 4,
        blend_mode: str = "cosine"
    ) -> np.ndarray:
        """Performs full-image inference by tiling, batched inference, and merging.
        
        Args:
            image: Full size image as numpy array (H, W, 3).
            tile_size: Size of tile patch.
            overlap: Overlap in pixels between adjacent tiles.
            batch_size: Batch size for model inference.
            blend_mode: Blending window type for merging ("average" or "cosine").
            
        Returns:
            Full-resolution probability map (H, W).
        """
        height, width = image.shape[:2]
        
        # 1. Generate coordinates
        generator = TileGenerator(tile_size=tile_size, overlap=overlap)
        coords = generator.get_tile_coordinates(height, width)
        
        # 2. Crop tiles
        cropped = generator.crop_tiles(image, coords)
        tiles_list = [tile for tile, _ in cropped]
        coords_list = [coord for _, coord in cropped]
        
        # 3. Predict tiles in batches
        tile_preds = self.predict_tiles_batch(tiles_list, batch_size=batch_size)
        
        # 4. Merge predictions
        merger = TileMerger(height, width, channels=1, blend_mode=blend_mode)
        for tile_pred, coord in zip(tile_preds, coords_list):
            merger.add_tile(tile_pred, coord)
            
        return merger.get_merged_map()
