import cv2
import numpy as np
from skimage.morphology import skeletonize

from src.preprocessing.marker_suppression import MarkerSuppressor

class PostProcessor:
    @staticmethod
    def binarize_probability_map(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        binary = (prob_map >= threshold).astype(np.uint8) * 255
        return binary

    @staticmethod
    def remove_noise(binary_mask: np.ndarray, min_area: int = 20) -> np.ndarray:
        return MarkerSuppressor.remove_small_components(binary_mask, min_area)

    @staticmethod
    def remove_colored_markers(
        binary_mask: np.ndarray,
        tile_rgb: np.ndarray,
        saturation_threshold: int = 85,
        value_threshold: int = 55,
    ) -> np.ndarray:
        marker_mask = MarkerSuppressor.build_colored_marker_mask(
            tile_rgb,
            saturation_threshold=saturation_threshold,
            value_threshold=value_threshold,
        )
        return cv2.bitwise_and(binary_mask, cv2.bitwise_not(marker_mask))

    @staticmethod
    def suppress_marker_shapes(
        binary_mask: np.ndarray,
        frame_side_coverage_threshold: float = 0.35,
        roundness_threshold: float = 0.58,
    ) -> np.ndarray:
        return MarkerSuppressor.suppress_marker_like_shapes(
            binary_mask,
            frame_side_coverage_threshold=frame_side_coverage_threshold,
            roundness_threshold=roundness_threshold,
        )

    @staticmethod
    def skeletonize(binary_mask: np.ndarray) -> np.ndarray:
        bool_mask = binary_mask > 0
        skeleton_bool = skeletonize(bool_mask)
        return (skeleton_bool.astype(np.uint8)) * 255

    @staticmethod
    def estimate_dimensions(binary_mask: np.ndarray, skeleton: np.ndarray) -> dict:
        crack_pixels = np.sum(binary_mask > 0)
        skeleton_pixels = np.sum(skeleton > 0)

        if skeleton_pixels == 0:
            return {
                "length_pixels": 0,
                "average_width_pixels": 0.0,
                "crack_area_pixels": 0,
            }

        length = int(skeleton_pixels)
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        skeleton_distances = dist_transform[skeleton > 0]
        average_width = float(2.0 * np.mean(skeleton_distances))

        return {
            "length_pixels": length,
            "average_width_pixels": round(average_width, 2),
            "crack_area_pixels": int(crack_pixels),
        }
