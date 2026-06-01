from typing import List, Dict, Any, Tuple
import numpy as np

class TileGenerator:
    def __init__(self, tile_size: int = 512, overlap: int = 64):
        """Initializes the TileGenerator.
        
        Args:
            tile_size: The height and width of each tile.
            overlap: The overlap in pixels between adjacent tiles.
        """
        if overlap >= tile_size:
            raise ValueError("Overlap must be strictly less than tile_size")
        self.tile_size = tile_size
        self.overlap = overlap

    def get_tile_coordinates(self, height: int, width: int) -> List[Dict[str, int]]:
        """Calculates overlapping tile coordinates for an image of shape (height, width).
        
        If the tile size does not divide the dimensions perfectly, the last tile is aligned
        with the bottom/right border, ensuring all tiles have exactly the size (tile_size x tile_size).
        
        Returns:
            A list of coordinate dictionaries containing 'ymin', 'xmin', 'ymax', 'xmax'.
        """
        coords = []
        step = self.tile_size - self.overlap
        
        # Y-coordinates
        y_starts = []
        y = 0
        while y < height:
            if y + self.tile_size >= height:
                # Clamp last tile to bottom edge
                y_starts.append(max(0, height - self.tile_size))
                break
            y_starts.append(y)
            y += step
            
        # X-coordinates
        x_starts = []
        x = 0
        while x < width:
            if x + self.tile_size >= width:
                # Clamp last tile to right edge
                x_starts.append(max(0, width - self.tile_size))
                break
            x_starts.append(x)
            x += step
            
        # Combine
        tile_id = 0
        for y_start in y_starts:
            for x_start in x_starts:
                coords.append({
                    "tile_id": tile_id,
                    "ymin": y_start,
                    "xmin": x_start,
                    "ymax": y_start + self.tile_size,
                    "xmax": x_start + self.tile_size
                })
                tile_id += 1
                
        return coords

    def crop_tiles(self, image: np.ndarray, coords: List[Dict[str, int]]) -> List[Tuple[np.ndarray, Dict[str, int]]]:
        """Crops an image into tiles based on the provided list of coordinates.
        
        Returns:
            A list of tuples, each containing (cropped_tile, coord_dict).
        """
        tiles = []
        for coord in coords:
            ymin, xmin = coord["ymin"], coord["xmin"]
            ymax, xmax = coord["ymax"], coord["xmax"]
            
            # Slice image
            tile = image[ymin:ymax, xmin:xmax]
            tiles.append((tile, coord))
            
        return tiles
