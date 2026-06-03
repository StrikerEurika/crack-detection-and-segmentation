import numpy as np
from typing import List, Dict, Any, Tuple
from src.tiling.tile_generator import TileGenerator
from src.tiling.tile_merger import TileMerger

class BasePredictor:
    def predict_tile_batch(self, tiles: List[np.ndarray], batch_size: int = 4) -> List[np.ndarray]:
        """Runs model inference on a batch of tiles.
        
        Args:
            tiles: List of tile images (H, W, C) as numpy arrays.
            batch_size: Batch size for model inference.
            
        Returns:
            A list of probability maps of shape (H, W) or (H, W, 1).
        """
        raise NotImplementedError("Subclasses must implement predict_tile_batch")
        
    def predict_full_image(
        self,
        image: np.ndarray,
        tile_size: int = 512,
        overlap: int = 64,
        batch_size: int = 4,
        blend_mode: str = "cosine"
    ) -> np.ndarray:
        """Performs full-image inference by tiling, batched tile inference, and merging.
        
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
        
        # 3. Predict tiles in batches using the subclass's predict_tile_batch method
        tile_preds = self.predict_tile_batch(tiles_list, batch_size=batch_size)
        
        # 4. Merge predictions
        merger = TileMerger(height, width, channels=1, blend_mode=blend_mode)
        for tile_pred, coord in zip(tile_preds, coords_list):
            merger.add_tile(tile_pred, coord)
            
        return merger.get_merged_map()
