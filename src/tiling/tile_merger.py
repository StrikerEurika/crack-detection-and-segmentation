from typing import Dict, List, Any
import numpy as np

class TileMerger:
    def __init__(self, height: int, width: int, channels: int = 1, blend_mode: str = "cosine"):
        """Initializes the TileMerger.
        
        Args:
            height: Height of the reconstructed full image.
            width: Width of the reconstructed full image.
            channels: Number of output channels (typically 1 for segmentation masks).
            blend_mode: "average" (simple averaging) or "cosine" (smooth cosine blending window).
        """
        self.height = height
        self.width = width
        self.channels = channels
        self.blend_mode = blend_mode
        
        # Canvas to accumulate predicted probabilities
        self.canvas = np.zeros((height, width, channels), dtype=np.float32)
        # Canvas to accumulate blending weights
        self.weight_canvas = np.zeros((height, width, channels), dtype=np.float32)
        
        # Precomputed tile window weights
        self._tile_weights = {}

    def _get_window_weights(self, tile_height: int, tile_width: int) -> np.ndarray:
        """Computes and caches blending weights for a tile of given size."""
        key = (tile_height, tile_width)
        if key in self._tile_weights:
            return self._tile_weights[key]
            
        if self.blend_mode == "average":
            weights = np.ones((tile_height, tile_width, 1), dtype=np.float32)
        elif self.blend_mode == "cosine":
            # Create a 2D cosine window
            y_window = np.sin(np.linspace(0, np.pi, tile_height)).astype(np.float32)
            x_window = np.sin(np.linspace(0, np.pi, tile_width)).astype(np.float32)
            # Make sure weight is not exactly 0 at the border to avoid numerical instability
            y_window = np.clip(y_window, 0.01, 1.0)
            x_window = np.clip(x_window, 0.01, 1.0)
            
            w_2d = np.outer(y_window, x_window)
            weights = np.expand_dims(w_2d, axis=-1)
        else:
            raise ValueError(f"Unknown blend mode: {self.blend_mode}")
            
        self._tile_weights[key] = weights
        return weights

    def add_tile(self, tile_pred: np.ndarray, coords: Dict[str, int]):
        """Adds a tile prediction to the merger canvas.
        
        Args:
            tile_pred: The prediction map of shape (tile_height, tile_width) or (tile_height, tile_width, channels).
            coords: Coordinate dictionary with 'ymin', 'xmin', 'ymax', 'xmax'.
        """
        ymin, xmin = coords["ymin"], coords["xmin"]
        ymax, xmax = coords["ymax"], coords["xmax"]
        
        # Ensure 3D shape (H, W, C)
        if len(tile_pred.shape) == 2:
            tile_pred = np.expand_dims(tile_pred, axis=-1)
            
        th, tw = ymax - ymin, xmax - xmin
        weights = self._get_window_weights(th, tw)
        
        # Accumulate weighted prediction and weight
        self.canvas[ymin:ymax, xmin:xmax] += tile_pred * weights
        self.weight_canvas[ymin:ymax, xmin:xmax] += weights

    def get_merged_map(self) -> np.ndarray:
        """Computes the final merged probability map by normalizing by accumulated weights.
        
        Returns:
            A numpy array of shape (height, width) or (height, width, channels) with values normalized.
        """
        # Avoid division by zero
        safe_weights = np.where(self.weight_canvas == 0, 1.0, self.weight_canvas)
        merged = self.canvas / safe_weights
        
        # Squeeze channels if it's a single channel output
        if self.channels == 1:
            merged = np.squeeze(merged, axis=-1)
            
        return merged
