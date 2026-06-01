import os
import cv2
import numpy as np
import random
from pathlib import Path
from src.ingestion.image_loader import ImageLoader
from src.tiling.tile_generator import TileGenerator

def tile_split(
    raw_dir: Path,
    output_dir: Path,
    split: str,
    tile_size: int = 512,
    overlap: int = 64,
    keep_negative_prob: float = 0.1,
    seed: int = 42
):
    """Tiles the images and masks in a given dataset split."""
    random.seed(seed)
    
    split_img_dir = raw_dir / split / "images"
    split_mask_dir = raw_dir / split / "masks"
    
    if not split_img_dir.exists() or not split_mask_dir.exists():
        print(f"Skipping split '{split}' (directories not found).")
        return
        
    out_img_dir = output_dir / split / "images"
    out_mask_dir = output_dir / split / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    
    img_files = sorted(list(split_img_dir.glob("*.png")) + list(split_img_dir.glob("*.jpg")) + list(split_img_dir.glob("*.tiff")))
    
    print(f"Tiling split '{split}' ({len(img_files)} images)...")
    
    tiler = TileGenerator(tile_size=tile_size, overlap=overlap)
    
    total_pos = 0
    total_neg = 0
    total_saved = 0
    
    for img_path in img_files:
        mask_path = split_mask_dir / img_path.name
        if not mask_path.exists():
            print(f"Warning: Mask not found for {img_path.name}, skipping.")
            continue
            
        # Load image and mask
        image = ImageLoader.load(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        height, width = image.shape[:2]
        coords = tiler.get_tile_coordinates(height, width)
        
        # Crop tiles
        image_tiles = tiler.crop_tiles(image, coords)
        mask_tiles = tiler.crop_tiles(mask, coords)
        
        base_name = img_path.stem
        
        for i, ((img_tile, _), (mask_tile, coord)) in enumerate(zip(image_tiles, mask_tiles)):
            # Check if tile contains crack (any pixel > 127 in mask)
            has_crack = np.any(mask_tile > 127)
            
            save_tile = False
            if has_crack:
                total_pos += 1
                save_tile = True
            else:
                total_neg += 1
                # Sample negative tiles with probability keep_negative_prob
                if random.random() < keep_negative_prob:
                    save_tile = True
                    
            if save_tile:
                tile_name = f"{base_name}_tile_{i:04d}.png"
                
                # Convert RGB back to BGR for cv2.imwrite
                img_tile_bgr = cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(str(out_img_dir / tile_name), img_tile_bgr)
                cv2.imwrite(str(out_mask_dir / tile_name), mask_tile)
                total_saved += 1
                
    print(f"Split '{split}' completed: Saved {total_saved} tiles (Positive: {total_pos}, Negative: {total_neg}, Kept Empty: {total_saved - total_pos})")

def main():
    raw_dir = Path("data/raw")
    output_dir = Path("data/tiles")
    
    # Tile train, val, and test splits
    tile_split(raw_dir, output_dir, "train", tile_size=512, overlap=64, keep_negative_prob=0.1)
    tile_split(raw_dir, output_dir, "val", tile_size=512, overlap=64, keep_negative_prob=0.2) # keep a bit more validation background
    tile_split(raw_dir, output_dir, "test", tile_size=512, overlap=64, keep_negative_prob=1.0) # keep all test tiles for perfect reconstruction comparison

if __name__ == "__main__":
    main()
