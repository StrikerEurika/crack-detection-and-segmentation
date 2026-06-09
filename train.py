import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.datasets.loader import CrackTileDataset, get_default_transforms
from src.models import CrackUnetModel
from src.training.losses import ComboLoss
from src.evaluation.metrics import CrackMetrics
from src.utils.logger import setup_logger

logger = setup_logger("train")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Crack Segmentation Model")
    parser.add_argument("--data-dir", type=str, default="data/tiles", help="Path to tiled dataset directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Backbone encoder network")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v1", help="Directory to save checkpoints")
    parser.add_argument("--marker-aug", action="store_true", help="Enable synthetic marker augmentation during training")
    parser.add_argument("--marker-aug-prob", type=float, default=0.3, help="Probability of marker augmentation per sample")
    return parser.parse_args()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item() * images.size(0)
            
            # Predict probabilities and threshold to binary
            probs = torch.sigmoid(logits).cpu().numpy()
            bin_preds = (probs > 0.5).astype(np.uint8)
            
            all_preds.extend(bin_preds)
            all_targets.extend(masks.cpu().numpy().astype(np.uint8))
            
    # Compute overall pixel-level metrics
    all_preds_np = np.stack(all_preds)
    all_targets_np = np.stack(all_targets)
    
    metrics = CrackMetrics.compute_pixel_metrics(all_preds_np, all_targets_np)
    metrics["val_loss"] = total_loss / len(loader.dataset)
    
    return metrics

import numpy as np  # Ensure numpy is imported for stacked metrics in validate()

def main():
    args = parse_args()
    
    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Device config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Datasets and loaders
    data_path = Path(args.data_dir)
    
    train_img_dir = data_path / "train" / "images"
    train_mask_dir = data_path / "train" / "masks"
    val_img_dir = data_path / "val" / "images"
    val_mask_dir = data_path / "val" / "masks"
    
    train_images = list(train_img_dir.glob("*.png"))
    train_masks = [train_mask_dir / p.name for p in train_images]
    
    val_images = list(val_img_dir.glob("*.png"))
    val_masks = [val_mask_dir / p.name for p in val_images]
    
    logger.info(f"Loaded {len(train_images)} training tiles and {len(val_images)} validation tiles.")
    
    if len(train_images) == 0:
        logger.error("No training tiles found. Run tile_dataset.py first.")
        return
        
    train_dataset = CrackTileDataset(
        train_images,
        train_masks,
        transform=get_default_transforms(img_size=512, is_training=True),
        marker_augmentation=args.marker_aug,
        marker_aug_prob=args.marker_aug_prob,
    )

    val_dataset = CrackTileDataset(
        val_images,
        val_masks,
        transform=get_default_transforms(img_size=512, is_training=False),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model, Optimizer, Loss
    model = CrackUnetModel(encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Combined Loss: 0.4 BCE + 0.6 Dice (standard for thin structure segmentation)
    criterion = ComboLoss(bce_weight=0.4, dice_weight=0.6, focal_weight=0.0)
    
    best_f1 = -1.0
    
    logger.info("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics["val_loss"]
        val_f1 = val_metrics["pixel_f1_dice"]
        val_iou = val_metrics["pixel_iou"]
        
        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1 (Dice): {val_f1:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )
        
        # Save best model based on Dice coefficient
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_path = checkpoint_dir / "best_model.pth"
            model.save(str(best_path))
            logger.info(f"New best model saved to {best_path} (F1: {best_f1:.4f})")
            
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
