from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class ImageLoader:
    @staticmethod
    def load(path: str) -> np.ndarray:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        suffix = path_obj.suffix.lower()

        if suffix in (".tif", ".tiff"):
            image = cv2.imread(str(path_obj), cv2.IMREAD_UNCHANGED)
            if image is None:
                with Image.open(path_obj) as img:
                    image = np.array(img)
        elif suffix in (".raw", ".dng"):
            try:
                import rawpy
                with rawpy.imread(str(path_obj)) as raw:
                    image = raw.postprocess()
            except ImportError:
                raise ImportError(f"Install rawpy for {suffix} support: uv add rawpy")
        else:
            image = cv2.imread(str(path_obj), cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                with Image.open(path_obj) as img:
                    image = np.array(img.convert("RGB"))

        if image is None:
            raise ValueError(f"Could not load image: {path}")
        return image

    @staticmethod
    def get_metadata(path: str) -> dict:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            with Image.open(path_obj) as img:
                width, height = img.size
                mode = img.mode
                channels = 1 if mode in ("L", "P") else len(mode)
                if "16" in mode or img.info.get("bits", 8) == 16:
                    bit_depth = 16
                elif "32" in mode or img.info.get("bits", 8) == 32:
                    bit_depth = 32
                else:
                    bit_depth = 8
        except Exception:
            img = ImageLoader.load(path)
            height, width = img.shape[:2]
            channels = img.shape[2] if len(img.shape) > 2 else 1
            bit_depth = 16 if img.dtype == np.uint16 else (32 if img.dtype == np.float32 else 8)

        return {
            "file_name": path_obj.name,
            "file_size_bytes": path_obj.stat().st_size,
            "width": width,
            "height": height,
            "channels": channels,
            "bit_depth": bit_depth,
        }
