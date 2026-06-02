import cv2
import numpy as np
from typing import Optional

from src.inference.marker_suppression import MarkerSuppressor


class MarkerInpaint:

    @staticmethod
    def detect_markers(
        image_rgb: np.ndarray,
        saturation_threshold: int = 85,
        value_threshold: int = 55,
    ) -> np.ndarray:
        return MarkerSuppressor.build_colored_marker_mask(
            image_rgb,
            saturation_threshold=saturation_threshold,
            value_threshold=value_threshold,
        )

    @staticmethod
    def inpaint_image(
        image_rgb: np.ndarray,
        marker_mask: Optional[np.ndarray] = None,
        saturation_threshold: int = 85,
        value_threshold: int = 55,
        inpaint_radius: int = 5,
        method: int = cv2.INPAINT_TELEA,
    ) -> np.ndarray:
        if marker_mask is None:
            marker_mask = MarkerSuppressor.build_colored_marker_mask(
                image_rgb,
                saturation_threshold=saturation_threshold,
                value_threshold=value_threshold,
            )

        if not np.any(marker_mask > 0):
            return image_rgb

        return MarkerSuppressor.inpaint_markers(image_rgb, marker_mask, inpaint_radius, method)

    @staticmethod
    def inpaint_tiled(
        image_rgb: np.ndarray,
        tile_size: int = 512,
        overlap: int = 64,
        saturation_threshold: int = 85,
        value_threshold: int = 55,
        inpaint_radius: int = 5,
    ) -> np.ndarray:
        return MarkerSuppressor.preprocess_image(
            image_rgb,
            tile_size=tile_size,
            overlap=overlap,
            saturation_threshold=saturation_threshold,
            value_threshold=value_threshold,
            inpaint_radius=inpaint_radius,
        )

    @staticmethod
    def add_synthetic_markers(
        image: np.ndarray,
        mask: np.ndarray,
        num_markers: int = 3,
        marker_width: int = 8,
    ) -> tuple:
        result_img = image.copy()
        h, w = image.shape[:2]

        colors = [
            (255, 0, 0),
            (0, 0, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        for _ in range(num_markers):
            color = colors[np.random.randint(len(colors))]
            cx = np.random.randint(w // 4, 3 * w // 4)
            cy = np.random.randint(h // 4, 3 * h // 4)
            radius = np.random.randint(20, 60)

            cv2.circle(result_img, (cx, cy), radius, color, marker_width)
            cv2.arrowedLine(
                result_img,
                (cx, cy),
                (cx + np.random.randint(-30, 30), cy + np.random.randint(-30, 30)),
                color,
                marker_width,
                tipLength=0.3,
            )

        return result_img, mask
