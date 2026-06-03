import cv2
import numpy as np
from typing import Optional


class MarkerSuppressor:

    @staticmethod
    def build_colored_marker_mask(
        tile_rgb: np.ndarray,
        saturation_threshold: int = 85,
        value_threshold: int = 55,
    ) -> np.ndarray:
        hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # 1. Blue/Purple markers: Hue [90, 150], Sat >= sat_t_blue, Val >= val_t_blue
        sat_t_blue = max(40, saturation_threshold - 35)
        val_t_blue = max(40, value_threshold - 15)
        blue_mask = (h >= 90) & (h <= 150) & (sat >= sat_t_blue) & (val >= val_t_blue)

        # 2. Red/Pink/Orange markers: Hue [0, 10] or [170, 180], Sat >= sat_t_red, Val >= val_t_blue
        sat_t_red = max(80, saturation_threshold + 5)
        red_mask = ((h <= 10) | (h >= 170)) & (sat >= sat_t_red) & (val >= val_t_blue)

        # 3. Green/Teal markers: Hue [35, 85], Sat >= sat_t_green, Val >= val_t_blue
        sat_t_green = max(50, saturation_threshold - 25)
        green_mask = (h >= 35) & (h <= 85) & (sat >= sat_t_green) & (val >= val_t_blue)

        # 4. Yellow/Orange markers: Hue [10, 35], but extremely high saturation to avoid warm concrete
        sat_t_yellow = max(155, saturation_threshold + 70)
        yellow_mask = (h >= 10) & (h <= 35) & (sat >= sat_t_yellow) & (val >= val_t_blue)

        marker_like = (blue_mask | red_mask | green_mask | yellow_mask).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        marker_like = cv2.morphologyEx(marker_like, cv2.MORPH_OPEN, kernel)
        marker_like = cv2.morphologyEx(marker_like, cv2.MORPH_CLOSE, kernel)
        return marker_like

    @staticmethod
    def remove_small_components(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
        if min_area <= 1:
            return binary_mask

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        out = np.zeros_like(binary_mask)
        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] >= min_area:
                out[labels == idx] = 255
        return out

    @staticmethod
    def suppress_marker_like_shapes(
        binary_mask: np.ndarray,
        frame_side_coverage_threshold: float = 0.35,
        roundness_threshold: float = 0.58,
    ) -> np.ndarray:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned = np.zeros_like(binary_mask)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter + 1e-6)

            x, y, w, h = cv2.boundingRect(contour)
            aspect = max(w / (h + 1e-6), h / (w + 1e-6))

            component_crop = np.zeros((h, w), dtype=np.uint8)
            shifted = contour.copy()
            shifted[:, 0, 0] -= x
            shifted[:, 0, 1] -= y
            cv2.drawContours(component_crop, [shifted], -1, color=255, thickness=-1)

            top_cov = float(np.mean(component_crop[0, :] > 0)) if h > 0 else 0.0
            bottom_cov = float(np.mean(component_crop[-1, :] > 0)) if h > 0 else 0.0
            left_cov = float(np.mean(component_crop[:, 0] > 0)) if w > 0 else 0.0
            right_cov = float(np.mean(component_crop[:, -1] > 0)) if w > 0 else 0.0

            side_hits = sum(
                coverage >= frame_side_coverage_threshold
                for coverage in (top_cov, bottom_cov, left_cov, right_cov)
            )
            is_frame_like = side_hits >= 3 and min(w, h) >= 16
            is_round_marker = circularity >= roundness_threshold and aspect <= 2.2 and area >= 60

            if is_frame_like or is_round_marker:
                continue

            cv2.drawContours(cleaned, [contour], -1, color=255, thickness=-1)

        return cleaned

    @staticmethod
    def inpaint_markers(
        image_rgb: np.ndarray,
        marker_mask: np.ndarray,
        inpaint_radius: int = 5,
        method: int = cv2.INPAINT_TELEA,
    ) -> np.ndarray:
        mask_uint8 = (marker_mask > 0).astype(np.uint8) * 255
        return cv2.inpaint(image_rgb, mask_uint8, inpaint_radius, method)

    @classmethod
    def filter_mask(
        cls,
        mask_prob: np.ndarray,
        tile_rgb: np.ndarray,
        min_component_area: int = 10,
        disable_color_suppression: bool = False,
        saturation_threshold: int = 85,
        value_threshold: int = 55,
        frame_side_coverage_threshold: float = 0.35,
        roundness_threshold: float = 0.58,
    ) -> np.ndarray:
        mask_bin = (mask_prob >= 0.5).astype(np.uint8) * 255

        if not disable_color_suppression:
            marker_color = cls.build_colored_marker_mask(
                tile_rgb,
                saturation_threshold=saturation_threshold,
                value_threshold=value_threshold,
            )
            mask_bin = cv2.bitwise_and(mask_bin, cv2.bitwise_not(marker_color))

        mask_bin = cls.remove_small_components(mask_bin, min_component_area)
        mask_bin = cls.suppress_marker_like_shapes(
            mask_bin,
            frame_side_coverage_threshold=frame_side_coverage_threshold,
            roundness_threshold=roundness_threshold,
        )

        return mask_bin.astype(np.float32) / 255.0

    @classmethod
    def preprocess_tile(
        cls,
        tile_rgb: np.ndarray,
        inpaint: bool = True,
        saturation_threshold: int = 85,
        value_threshold: int = 55,
        inpaint_radius: int = 5,
    ) -> np.ndarray:
        if not inpaint:
            return tile_rgb
        marker_mask = cls.build_colored_marker_mask(
            tile_rgb,
            saturation_threshold=saturation_threshold,
            value_threshold=value_threshold,
        )
        if np.any(marker_mask > 0):
            return cls.inpaint_markers(tile_rgb, marker_mask, inpaint_radius)
        return tile_rgb

    @classmethod
    def preprocess_image(
        cls,
        image_rgb: np.ndarray,
        tile_size: int = 512,
        overlap: int = 64,
        saturation_threshold: int = 85,
        value_threshold: int = 55,
        inpaint_radius: int = 5,
    ) -> np.ndarray:
        from src.tiling.tile_generator import TileGenerator

        h, w = image_rgb.shape[:2]
        generator = TileGenerator(tile_size=tile_size, overlap=overlap)
        coords = generator.get_tile_coordinates(h, w)

        result = image_rgb.copy()

        for coord in coords:
            ymin, xmin = coord["ymin"], coord["xmin"]
            ymax, xmax = coord["ymax"], coord["xmax"]
            tile = result[ymin:ymax, xmin:xmax].copy()

            marker_mask = cls.build_colored_marker_mask(
                tile,
                saturation_threshold=saturation_threshold,
                value_threshold=value_threshold,
            )
            if np.any(marker_mask > 0):
                inpainted = cls.inpaint_markers(tile, marker_mask, inpaint_radius)
                result[ymin:ymax, xmin:xmax] = inpainted

        return result
