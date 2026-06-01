import os
from pathlib import Path
from typing import Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image

class ImageLoader:
    @staticmethod
    def load(path: str) -> np.ndarray:
        """Loads an image from the given path and returns it as a numpy array.
        
        Supports PNG, JPG, JPEG, and TIFF.
        Handles high bit-depth (e.g., 16-bit) and preserves it.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Image not found at path: {path}")
            
        suffix = path_obj.suffix.lower()
        
        # For TIFF files, especially high bit depth, we can use OpenCV with IMREAD_UNCHANGED or PIL
        if suffix in ['.tif', '.tiff']:
            # Try reading with IMREAD_UNCHANGED to preserve 16-bit depth
            image = cv2.imread(str(path_obj), cv2.IMREAD_UNCHANGED)
            if image is None:
                # Fallback to PIL
                with Image.open(path_obj) as img:
                    image = np.array(img)
        elif suffix in ['.raw', '.dng']:
            # RAW support (optional rawpy import if needed)
            try:
                import rawpy
                with rawpy.imread(str(path_obj)) as raw:
                    image = raw.postprocess()
            except ImportError:
                raise ImportError(
                    f"To load RAW images ({suffix}), please install rawpy. "
                    "Run 'uv add rawpy'."
                )
        else:
            # Standard formats (JPG, PNG, etc.)
            image = cv2.imread(str(path_obj), cv2.IMREAD_COLOR)
            if image is not None:
                # OpenCV reads BGR, convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Fallback to PIL
                with Image.open(path_obj) as img:
                    image = np.array(img.convert('RGB'))
                    
        if image is None:
            raise ValueError(f"Could not load image at: {path}")
            
        return image

    @staticmethod
    def get_metadata(path: str) -> Dict[str, Any]:
        """Extracts basic metadata from the image file without loading the full image into memory if possible."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Image not found at path: {path}")
            
        suffix = path_obj.suffix.lower()
        
        # Default metadata values
        width, height, channels, bit_depth = 0, 0, 3, 8
        
        # Use PIL to read metadata without fully loading pixel data
        try:
            with Image.open(path_obj) as img:
                width, height = img.size
                mode = img.mode
                if mode in ['L', 'P']:
                    channels = 1
                elif mode in ['RGB', 'YCbCr']:
                    channels = 3
                elif mode in ['RGBA', 'CMYK']:
                    channels = 4
                else:
                    channels = len(mode)
                    
                # Determine bit depth
                if '16' in mode or img.info.get('bits', 8) == 16:
                    bit_depth = 16
                elif '32' in mode or img.info.get('bits', 8) == 32:
                    bit_depth = 32
                else:
                    bit_depth = 8
        except Exception:
            # Fallback: load image shape
            img = ImageLoader.load(path)
            height, width = img.shape[:2]
            channels = img.shape[2] if len(img.shape) > 2 else 1
            bit_depth = 16 if img.dtype == np.uint16 else (32 if img.dtype == np.float32 or img.dtype == np.uint32 else 8)
            
        metadata = {
            "file_name": path_obj.name,
            "file_size_bytes": path_obj.stat().st_size,
            "width": width,
            "height": height,
            "channels": channels,
            "bit_depth": bit_depth,
            "dtype": str(np.uint16 if bit_depth == 16 else (np.float32 if bit_depth == 32 else np.uint8))
        }
        
        return metadata
