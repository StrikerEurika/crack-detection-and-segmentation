import cv2
import numpy as np

class ImagePreprocessor:
    @staticmethod
    def normalize_pixels(image: np.ndarray) -> np.ndarray:
        """Normalizes image pixel values to [0.0, 1.0] range as float32, based on original dtype."""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        elif np.issubdtype(image.dtype, np.floating):
            # Already float, clip to [0, 1] if needed, or return as is
            return np.clip(image.astype(np.float32), 0.0, 1.0)
        else:
            # Fallback
            max_val = np.max(image) if np.max(image) > 0 else 1.0
            return image.astype(np.float32) / max_val

    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance contrast.
        
        If the image is grayscale (2D or 3D with 1 channel), applies directly.
        If the image is color (RGB), converts to LAB color space, applies CLAHE on the L channel,
        and converts back to RGB.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Check if grayscale
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            # If 3D with 1 channel, squeeze, apply, and expand dim back
            is_3d = len(image.shape) == 3
            img_2d = image[:, :, 0] if is_3d else image
            
            # CLAHE requires uint8 or uint16
            orig_dtype = img_2d.dtype
            if np.issubdtype(orig_dtype, np.floating):
                # Temporary scale to uint8 for CLAHE
                img_2d_uint = (img_2d * 255.0).astype(np.uint8)
                enhanced = clahe.apply(img_2d_uint).astype(np.float32) / 255.0
            else:
                enhanced = clahe.apply(img_2d)
                
            return np.expand_dims(enhanced, axis=-1) if is_3d else enhanced
            
        else:
            # Color image (RGB)
            orig_dtype = image.dtype
            if np.issubdtype(orig_dtype, np.floating):
                # Temporary scale to uint8 for LAB conversion and CLAHE
                img_uint8 = (image * 255.0).astype(np.uint8)
                lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                return enhanced.astype(np.float32) / 255.0
            else:
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                return enhanced

    @staticmethod
    def denoise_bilateral(image: np.ndarray, d: int = 5, sigma_color: float = 25.0, sigma_space: float = 25.0) -> np.ndarray:
        """Applies bilateral filtering to smooth textures while preserving sharp edges (cracks)."""
        # Bilateral filter requires uint8 or float32 in OpenCV
        orig_dtype = image.dtype
        is_float = np.issubdtype(orig_dtype, np.floating)
        
        # If float, ensure it is float32 (OpenCV requirement)
        img_processed = image.astype(np.float32) if is_float else image
        
        # Apply filter
        denoised = cv2.bilateralFilter(img_processed, d, sigma_color, sigma_space)
        return denoised
