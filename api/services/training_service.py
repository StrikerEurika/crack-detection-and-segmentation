from __future__ import annotations
import gc
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from api.config import settings
from api.services.task_manager import task_manager
from src.datasets.loader import CrackTileDataset, get_default_transforms
from src.models.unet import CrackUnetModel
from src.training.losses import ComboLoss
from src.evaluation.metrics import CrackMetrics
from src.utils.logger import setup_logger

logger = setup_logger("api.training_service")


class TrainingState:
    def __init__(self, task_id: str, params: dict):
        self.task_id = task_id
        self.params = params
        self.status = "pending"
        self.epoch = 0
        self.total_epochs = params.get("epochs", 15)
        self.train_loss: Optional[float] = None
        self.val_loss: Optional[float] = None
        self.val_f1: Optional[float] = None
        self.val_iou: Optional[float] = None
        self.best_checkpoint: Optional[str] = None
        self.error: Optional[str] = None
        self.progress: float = 0.0
        self.message: str = "Initializing..."
        self.created_at = datetime.now(timezone.utc)
        self.completed_at: Optional[datetime] = None


_training_states: dict[str, TrainingState] = {}
_training_lock = threading.Lock()


def get_training_status(task_id: str) -> Optional[dict]:
    state = _training_states.get(task_id)
    if not state:
        return None
    return {
        "task_id": state.task_id,
        "status": state.status,
        "epoch": state.epoch,
        "total_epochs": state.total_epochs,
        "train_loss": state.train_loss,
        "val_loss": state.val_loss,
        "val_f1": state.val_f1,
        "val_iou": state.val_iou,
        "progress": state.progress,
        "message": state.message,
        "best_checkpoint": state.best_checkpoint,
        "created_at": state.created_at.isoformat(),
        "completed_at": state.completed_at.isoformat() if state.completed_at else None,
    }


def _validate_paths(params: dict) -> str:
    data_path = Path(params.get("data_dir", "data/tiles"))
    if not data_path.is_absolute():
        data_path = settings.project_root / data_path
    if not data_path.exists():
        return f"Data directory not found: {data_path}"
    train_img = data_path / "train" / "images"
    if not train_img.exists() or not list(train_img.glob("*.png")):
        return f"No training images found in {train_img}"
    return ""


def run_training(params: dict, task=None) -> dict:
    task_id = task.task_id if task else "unknown"
    state = TrainingState(task_id, params)
    with _training_lock:
        _training_states[task_id] = state

    data_path = Path(params.get("data_dir", "data/tiles"))
    if not data_path.is_absolute():
        data_path = settings.project_root / data_path

    checkpoint_dir = Path(params.get("checkpoint_dir", "checkpoints"))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = settings.project_root / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = params.get("epochs", 15)
    batch_size = params.get("batch_size", 4)
    lr = params.get("lr", 3e-4)
    weight_decay = params.get("weight_decay", 1e-4)
    encoder = params.get("encoder", "resnet34")
    marker_aug = params.get("marker_aug", False)
    marker_aug_prob = params.get("marker_aug_prob", 0.3)

    try:
        state.status = "running"
        state.message = "Loading datasets..."

        train_img_dir = data_path / "train" / "images"
        train_mask_dir = data_path / "train" / "masks"
        val_img_dir = data_path / "val" / "images"
        val_mask_dir = data_path / "val" / "masks"

        train_images = list(train_img_dir.glob("*.png"))
        train_masks = [train_mask_dir / p.name for p in train_images]
        val_images = list(val_img_dir.glob("*.png"))
        val_masks = [val_mask_dir / p.name for p in val_images]

        logger.info(f"Loaded {len(train_images)} training tiles, {len(val_images)} validation tiles.")

        if len(train_images) == 0:
            raise RuntimeError("No training tiles found.")

        train_dataset = CrackTileDataset(
            train_images, train_masks,
            transform=get_default_transforms(img_size=512, is_training=True),
            marker_augmentation=marker_aug,
            marker_aug_prob=marker_aug_prob,
        )
        val_dataset = CrackTileDataset(
            val_images, val_masks,
            transform=get_default_transforms(img_size=512, is_training=False),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        model = CrackUnetModel(
            encoder_name=encoder, encoder_weights="imagenet", in_channels=3
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = ComboLoss(bce_weight=0.4, dice_weight=0.6, focal_weight=0.0)

        best_f1 = -1.0
        state.message = "Training started."

        for epoch in range(1, epochs + 1):
            state.epoch = epoch
            state.message = f"Epoch {epoch}/{epochs}"
            state.progress = (epoch - 1) / epochs * 90

            model.train()
            total_loss = 0.0
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)

            scheduler.step()
            train_loss = total_loss / len(train_loader.dataset)
            state.train_loss = round(train_loss, 4)

            model.eval()
            val_total_loss = 0.0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    logits = model(images)
                    loss = criterion(logits, masks)
                    val_total_loss += loss.item() * images.size(0)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    bin_preds = (probs > 0.5).astype(np.uint8)
                    all_preds.extend(bin_preds)
                    all_targets.extend(masks.cpu().numpy().astype(np.uint8))

            all_preds_np = np.stack(all_preds)
            all_targets_np = np.stack(all_targets)
            metrics = CrackMetrics.compute_pixel_metrics(all_preds_np, all_targets_np)
            val_loss = val_total_loss / len(val_loader.dataset)

            state.val_loss = round(val_loss, 4)
            state.val_f1 = metrics["pixel_f1_dice"]
            state.val_iou = metrics["pixel_iou"]

            logger.info(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {state.val_f1:.4f} | "
                f"Val IoU: {state.val_iou:.4f}"
            )

            if state.val_f1 > best_f1:
                best_f1 = state.val_f1
                best_path = checkpoint_dir / "best_model.pth"
                model.save(str(best_path))
                state.best_checkpoint = str(best_path)
                logger.info(f"New best model saved (F1: {best_f1:.4f})")

        state.progress = 100.0
        state.status = "completed"
        state.message = "Training completed successfully."
        state.completed_at = datetime.now(timezone.utc)
        logger.info("Training completed successfully!")

        return {"status": "completed", "best_checkpoint": state.best_checkpoint}

    except Exception as e:
        state.status = "failed"
        state.error = str(e)
        state.message = f"Training failed: {e}"
        state.completed_at = datetime.now(timezone.utc)
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def start_training(params: dict) -> str:
    validation_error = _validate_paths(params)
    if validation_error:
        raise ValueError(validation_error)

    task_id = await task_manager.submit(run_training, params)
    return task_id
