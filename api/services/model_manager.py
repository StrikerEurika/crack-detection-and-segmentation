from __future__ import annotations
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from api.config import settings
from src.inference import get_predictor, BasePredictor
from src.utils.logger import setup_logger

logger = setup_logger("api.model_manager")


class ModelManager:
    def __init__(self):
        self._predictor: Optional[BasePredictor] = None
        self._model_id: Optional[str] = None
        self._model_path: Optional[str] = None
        self._model_type: Optional[str] = None
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded_at: Optional[datetime] = None
        self._encoder: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    @property
    def model_info(self) -> Optional[dict]:
        if not self.is_loaded:
            return None
        return {
            "model_id": self._model_id,
            "model_path": self._model_path,
            "model_type": self._model_type,
            "device": self._device,
            "loaded_at": self._loaded_at,
            "encoder": self._encoder,
        }

    def load(
        self,
        model_path: str,
        model_type: str = "auto",
        encoder: str = "resnet34",
    ) -> dict:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        resolved_type = model_type
        if resolved_type == "auto":
            ext = path.suffix.lower()
            p_str = str(path).lower()
            if ext == ".pth" or "unet" in p_str:
                resolved_type = "unet"
            elif ext == ".pt" or "yolo" in p_str:
                resolved_type = "yolo"
            else:
                raise ValueError(
                    f"Cannot auto-detect model type for {model_path}. "
                    "Specify --model-type explicitly."
                )

        logger.info(
            f"Loading {resolved_type} model from {model_path} on {self._device}..."
        )

        kwargs = {}
        if resolved_type == "yolo":
            kwargs = {
                "conf": 0.18,
                "iou": 0.45,
                "min_component_area": 20,
            }

        with self._lock:
            predictor = get_predictor(
                model_path=str(path),
                model_type=resolved_type,
                device=self._device,
                encoder=encoder,
                **kwargs,
            )
            self._predictor = predictor
            self._model_id = path.stem
            self._model_path = str(path)
            self._model_type = resolved_type
            self._encoder = encoder
            self._loaded_at = datetime.now(timezone.utc)

        logger.info(f"Model '{self._model_id}' loaded successfully.")
        return self.model_info

    def unload(self):
        with self._lock:
            self._predictor = None
            self._model_id = None
            self._model_path = None
            self._model_type = None
            self._loaded_at = None
            self._encoder = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded.")

    def get_predictor(self) -> BasePredictor:
        if not self.is_loaded:
            self.load(
                model_path=settings.default_model_path,
                model_type=settings.default_model_type,
            )
        return self._predictor

    def list_available(self) -> list[dict]:
        checkpoints = []
        ckpt_dir = settings.checkpoints_dir
        if not ckpt_dir.exists():
            return checkpoints

        for f in sorted(ckpt_dir.iterdir()):
            if f.suffix.lower() in (".pth", ".pt", ".onnx", ".torchscript"):
                size_mb = round(f.stat().st_size / (1024 * 1024), 2)
                ext = f.suffix.lower()
                if ext == ".pth" or "unet" in str(f).lower():
                    mtype = "unet"
                elif ext == ".pt" or "yolo" in str(f).lower():
                    mtype = "yolo"
                else:
                    mtype = "unknown"
                checkpoints.append({
                    "file_name": f.name,
                    "path": str(f),
                    "type": mtype,
                    "size_mb": size_mb,
                    "loaded": str(f) == self._model_path,
                })
        return checkpoints


model_manager = ModelManager()
