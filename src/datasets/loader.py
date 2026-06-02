import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MarkerAugmentation:
    def __init__(self, p: float = 0.3, num_markers: int = 2):
        self.p = p
        self.num_markers = num_markers
        self.colors = [
            (255, 0, 0),
            (0, 0, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 255),
        ]

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() >= self.p:
            return image, mask

        result_img = image.copy()
        h, w = image.shape[:2]

        for _ in range(np.random.randint(1, self.num_markers + 1)):
            color = self.colors[np.random.randint(len(self.colors))]
            cx = np.random.randint(w // 4, 3 * w // 4)
            cy = np.random.randint(h // 4, 3 * h // 4)
            radius = np.random.randint(15, 40)

            cv2.circle(result_img, (cx, cy), radius, color, np.random.randint(4, 10))
            dx = np.random.randint(-20, 20)
            dy = np.random.randint(-20, 20)
            cv2.arrowedLine(
                result_img,
                (cx, cy),
                (cx + dx, cy + dy),
                color,
                np.random.randint(4, 8),
                tipLength=0.3,
            )

        return result_img, mask


class CrackTileDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        transform: Optional[A.Compose] = None,
        normalize: bool = True,
        marker_augmentation: bool = False,
        marker_aug_prob: float = 0.3,
    ):
        assert len(image_paths) == len(mask_paths), "Number of images and masks must match"
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.transform = transform
        self.normalize = normalize
        self.marker_aug = MarkerAugmentation(p=marker_aug_prob) if marker_augmentation else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        assert img_path.name == mask_path.name, f"Mismatched image/mask names: {img_path.name} vs {mask_path.name}"

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.marker_aug:
            image, mask = self.marker_aug(image, mask)

        if self.normalize and self.transform is None:
            image = image.astype(np.float32) / 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image_aug = augmented['image']
            mask_aug = augmented['mask']

            if not isinstance(image_aug, torch.Tensor):
                image_aug = ToTensorV2()(image=image_aug)['image']
            if not isinstance(mask_aug, torch.Tensor):
                mask_aug = torch.from_numpy(mask_aug).float()
        else:
            image_aug = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask_aug = torch.from_numpy(mask).float()

        if len(mask_aug.shape) == 2:
            mask_aug = mask_aug.unsqueeze(0)

        return image_aug, mask_aug


def get_default_transforms(img_size: int = 512, is_training: bool = True) -> A.Compose:
    if is_training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
