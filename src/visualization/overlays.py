import cv2
import numpy as np

class Visualizer:
    @staticmethod
    def draw_mask_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        color: tuple = (255, 0, 0),
        alpha: float = 0.4
    ) -> np.ndarray:
        """Draws a semi-transparent colored overlay of the mask on the image.
        
        Args:
            image: Original RGB image of shape (H, W, 3).
            mask: Binary mask of shape (H, W) with values 0 or 255.
            color: RGB color tuple for the overlay.
            alpha: Transparency weight (0.0 to 1.0).
        """
        # Ensure mask is binary
        bin_mask = mask > 127
        
        # Copy image for output
        overlay_img = image.copy()
        
        # Set color on the overlay image where mask is True
        overlay_img[bin_mask] = color
        
        # Blend
        blended = cv2.addWeighted(overlay_img, alpha, image, 1.0 - alpha, 0)
        return blended

    @staticmethod
    def draw_error_analysis_overlay(
        image: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Draws an overlay color-coding True Positives (Green), False Positives (Red), and False Negatives (Blue).
        
        Args:
            image: Original RGB image of shape (H, W, 3).
            gt_mask: Binary ground truth mask (0 or 255).
            pred_mask: Binary prediction mask (0 or 255).
            alpha: Blending weight.
        """
        gt_bin = gt_mask > 127
        pred_bin = pred_mask > 127
        
        # Calculate TP, FP, FN
        tp = np.logical_and(gt_bin, pred_bin)
        fp = np.logical_and(np.logical_not(gt_bin), pred_bin)
        fn = np.logical_and(gt_bin, np.logical_not(pred_bin))
        
        # Create color mask (same size as image)
        color_mask = image.copy()
        
        # Color coding (RGB)
        color_mask[tp] = [0, 255, 0]   # True Positives: Green
        color_mask[fp] = [255, 0, 0]   # False Positives: Red
        color_mask[fn] = [0, 0, 255]   # False Negatives: Blue (Cyan/Yellow are options, Blue is standard)
        
        # Apply blending only to changed pixels
        mask_pixels = np.logical_or(np.logical_or(tp, fp), fn)
        
        output = image.copy()
        output[mask_pixels] = cv2.addWeighted(color_mask, alpha, image, 1.0 - alpha, 0)[mask_pixels]
        
        return output
