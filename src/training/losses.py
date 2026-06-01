import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class ComboLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, focal_weight: float = 0.0):
        """Combines BCE, Dice, and Focal loss for thin crack detection.
        
        Args:
            bce_weight: Weight for the Binary Cross Entropy loss.
            dice_weight: Weight for the Dice loss.
            focal_weight: Weight for the Focal loss.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # Initialize sub-losses from SMP
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.focal = smp.losses.FocalLoss(mode="binary", alpha=0.25, gamma=2.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes combined loss.
        
        Args:
            logits: Logits predicted by the model, shape (B, 1, H, W).
            targets: Binary ground truth targets, shape (B, 1, H, W).
        """
        loss = 0.0
        
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce(logits, targets)
            
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice(logits, targets)
            
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal(logits, targets)
            
        return loss
