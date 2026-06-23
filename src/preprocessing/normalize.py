from __future__ import annotations

import cv2
import numpy as np


class ImagePreprocessor:
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    @staticmethod
    def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
