import cv2
import numpy as np
from skimage.morphology import skeletonize

class PostProcessor:
    @staticmethod
    def binarize_probability_map(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binarizes the probability map to a binary mask (0 or 255)."""
        binary = (prob_map >= threshold).astype(np.uint8) * 255
        return binary

    @staticmethod
    def remove_noise(binary_mask: np.ndarray, min_area: int = 20) -> np.ndarray:
        """Removes small connected components (noise) from the binary mask.
        
        Args:
            binary_mask: Binary mask of shape (H, W) containing values 0 or 255.
            min_area: Minimum area in pixels to keep a connected component.
        """
        # Find components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
        
        # Create empty output mask
        cleaned_mask = np.zeros_like(binary_mask)
        
        # Stat indices: cv2.CC_STAT_AREA is 4
        for label_idx in range(1, num_labels):  # Skip background (0)
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned_mask[labels == label_idx] = 255
                
        return cleaned_mask

    @staticmethod
    def skeletonize(binary_mask: np.ndarray) -> np.ndarray:
        """Thin the binary mask to a 1-pixel wide centerline skeleton.
        
        Args:
            binary_mask: Binary mask of shape (H, W) with values 0 or 255.
            
        Returns:
            Binary skeleton mask of shape (H, W) with values 0 or 255.
        """
        # Convert to boolean mask for skimage
        bool_mask = binary_mask > 0
        skeleton_bool = skeletonize(bool_mask)
        return (skeleton_bool.astype(np.uint8)) * 255

    @staticmethod
    def estimate_dimensions(binary_mask: np.ndarray, skeleton: np.ndarray) -> dict:
        """Estimates crack physical dimensions (length and average width).
        
        Args:
            binary_mask: Cleaned binary mask (0 or 255).
            skeleton: 1-pixel wide centerline skeleton (0 or 255).
            
        Returns:
            A dictionary containing estimated 'length_pixels', 'average_width_pixels',
            and 'crack_area_pixels'.
        """
        crack_pixels = np.sum(binary_mask > 0)
        skeleton_pixels = np.sum(skeleton > 0)
        
        if skeleton_pixels == 0:
            return {
                "length_pixels": 0,
                "average_width_pixels": 0.0,
                "crack_area_pixels": 0
            }
            
        # Crack length is approximately the number of skeleton pixels
        length = int(skeleton_pixels)
        
        # Use Distance Transform to find local radius (half-width) at every skeleton point
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        
        # Get distances at the skeleton coordinates
        skeleton_distances = dist_transform[skeleton > 0]
        
        # Average width is 2 * average distance (radius)
        # We multiply by 2 because distance transform measures distance to the nearest background pixel, which is half-width
        average_width = float(2.0 * np.mean(skeleton_distances))
        
        return {
            "length_pixels": length,
            "average_width_pixels": round(average_width, 2),
            "crack_area_pixels": int(crack_pixels)
        }
