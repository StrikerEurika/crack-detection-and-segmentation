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
        self._model_version: Optional[str] = None
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
            "model_version": self._model_version,
            "model_type": self._model_type,
            "device": self._device,
            "loaded_at": self._loaded_at,
            "encoder": self._encoder,
        }

    def extract_version(self, path: Path) -> str:
        try:
            rel = path.relative_to(settings.checkpoints_dir)
            parts = rel.parts
            return parts[0] if len(parts) > 1 else "root"
        except ValueError:
            return "external"

    def resolve_model_path(self, path_or_version: str) -> Path:
        p = Path(path_or_version)
        
        # 1. Absolute path
        if p.is_absolute() and p.exists() and p.is_file():
            return p

        # 2. Relative to checkpoints directory
        p_ckpt = settings.checkpoints_dir / path_or_version
        if p_ckpt.exists() and p_ckpt.is_file():
            return p_ckpt

        # 3. Directory under checkpoints (e.g. "v1")
        if p_ckpt.exists() and p_ckpt.is_dir():
            for ext in (".pth", ".pt", ".onnx", ".torchscript"):
                for f in sorted(p_ckpt.rglob(f"*{ext}")):
                    if f.is_file():
                        return f

        # 4. Check if the file name or relative path matches inside checkpoints recursively
        name_to_find = p.name
        for ext in (".pth", ".pt", ".onnx", ".torchscript"):
            for f in sorted(settings.checkpoints_dir.rglob(f"*{ext}")):
                if f.name == name_to_find or str(f.relative_to(settings.checkpoints_dir)) == path_or_version:
                    return f

        # 5. Relative to project root
        p_proj = settings.project_root / path_or_version
        if p_proj.exists() and p_proj.is_file():
            return p_proj

        return p

    def load(
        self,
        model_path: str,
        model_type: str = "auto",
        encoder: str = "resnet34",
    ) -> dict:
        resolved_path = self.resolve_model_path(model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path} (resolved to {resolved_path})")

        resolved_type = model_type
        if resolved_type == "auto":
            ext = resolved_path.suffix.lower()
            p_str = str(resolved_path).lower()
            if "unet-plusplus" in p_str or "unetplusplus" in p_str:
                resolved_type = "unet_plusplus_v1"
            elif ext == ".pth" or "unet" in p_str:
                resolved_type = "unet"
            elif ext == ".pt" or "yolo" in p_str:
                resolved_type = "yolo"
            else:
                raise ValueError(
                    f"Cannot auto-detect model type for {resolved_path}. "
                    "Specify --model-type explicitly."
                )


        logger.info(
            f"Loading {resolved_type} model from {resolved_path} on {self._device}..."
        )

        kwargs = {}
        if resolved_type == "yolo":
            kwargs = {
                "conf": settings.default_conf_yolo,
                "iou": settings.default_iou_yolo,
                "min_component_area": settings.default_min_area,
            }

        with self._lock:
            predictor = get_predictor(
                model_path=str(resolved_path),
                model_type=resolved_type,
                device=self._device,
                encoder=encoder,
                **kwargs,
            )
            self._predictor = predictor
            self._model_id = resolved_path.stem
            self._model_path = str(resolved_path)
            self._model_version = self.extract_version(resolved_path)
            self._model_type = resolved_type
            self._encoder = encoder
            self._loaded_at = datetime.now(timezone.utc)

        logger.info(f"Model '{self._model_id}' (version: '{self._model_version}') loaded successfully.")
        return self.model_info

    def unload(self):
        with self._lock:
            self._predictor = None
            self._model_id = None
            self._model_path = None
            self._model_version = None
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
            # Try loading default model
            try:
                resolved_default = self.resolve_model_path(settings.default_model_path)
                if resolved_default.exists() and resolved_default.is_file():
                    self.load(
                        model_path=str(resolved_default),
                        model_type=settings.default_model_type,
                    )
                else:
                    raise FileNotFoundError()
            except Exception:
                # Search for any model in checkpoints_dir
                available = self.list_available()
                if available:
                    logger.warning(
                        f"Default model {settings.default_model_path} not found. "
                        f"Falling back to first available model: {available[0]['path']}"
                    )
                    self.load(
                        model_path=available[0]["path"],
                        model_type=available[0]["type"],
                    )
                else:
                    raise FileNotFoundError(
                        f"No models found. Please train a model or place one in checkpoints directory."
                    )
        return self._predictor

    def list_available(self) -> list[dict]:
        checkpoints = []
        ckpt_dir = settings.checkpoints_dir
        if not ckpt_dir.exists():
            return checkpoints

        for f in sorted(ckpt_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in (".pth", ".pt", ".onnx", ".torchscript"):
                size_mb = round(f.stat().st_size / (1024 * 1024), 2)
                ext = f.suffix.lower()
                if "unet-plusplus" in str(f).lower() or "unetplusplus" in str(f).lower():
                    mtype = "unet_plusplus_v1"
                elif ext == ".pth" or "unet" in str(f).lower():
                    mtype = "unet"
                elif ext == ".pt" or "yolo" in str(f).lower():
                    mtype = "yolo"
                else:
                    mtype = "unknown"
                
                rel_path = str(f.relative_to(ckpt_dir)).replace("\\", "/")
                version = self.extract_version(f)

                checkpoints.append({
                    "file_name": f.name,
                    "version": version,
                    "relative_path": rel_path,
                    "path": str(f),
                    "type": mtype,
                    "size_mb": size_mb,
                    "loaded": str(f) == self._model_path,
                })
        return checkpoints


model_manager = ModelManager()
