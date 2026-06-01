import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CrackTileDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        transform: Optional[A.Compose] = None,
        normalize: bool = True
    ):
        """Dataset for loading tiled images and corresponding crack masks.
        
        Args:
            image_paths: List of paths to the image tiles.
            mask_paths: List of paths to the mask tiles. Must correspond 1:1 with image_paths.
            transform: Albumentations transforms to apply.
            normalize: If True, rescales image pixel values to [0.0, 1.0].
        """
        assert len(image_paths) == len(mask_paths), "Number of images and masks must match"
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.transform = transform
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Verify alignment by checking filenames
        assert img_path.name == mask_path.name, f"Mismatched image/mask names: {img_path.name} vs {mask_path.name}"
        
        # Load image (RGB)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (Grayscale)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Convert mask to binary (0 and 1)
        mask = (mask > 127).astype(np.float32)
        
        # Apply normalization if not handled by transforms
        if self.normalize and self.transform is None:
            image = image.astype(np.float32) / 255.0
            
        # Apply albumentations transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image_aug = augmented['image']
            mask_aug = augmented['mask']
            
            # If transform did not convert to PyTorch tensors, do it manually
            if not isinstance(image_aug, torch.Tensor):
                image_aug = ToTensorV2()(image=image_aug)['image']
            if not isinstance(mask_aug, torch.Tensor):
                mask_aug = torch.from_numpy(mask_aug).float()
        else:
            # Convert to PyTorch tensors manually: HWC -> CHW
            image_aug = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask_aug = torch.from_numpy(mask).float()
            
        # Ensure mask has shape (1, H, W)
        if len(mask_aug.shape) == 2:
            mask_aug = mask_aug.unsqueeze(0)
            
        return image_aug, mask_aug

def get_default_transforms(img_size: int = 512, is_training: bool = True) -> A.Compose:
    """Returns standard data augmentation/preprocessing pipeline for crack detection.
    
    Data augmentations are carefully chosen not to blur or destroy the 1-3 pixel thin cracks.
    """
    if is_training:
        return A.Compose([
            # Flips and rotations are non-destructive to geometry
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            # Subtle random changes to brightness/contrast (simulating lighting changes)
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            
            # Normalization and Tensor conversion
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
