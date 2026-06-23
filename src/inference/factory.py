from __future__ import annotations
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger("src.inference.factory")


def get_predictor(
    model_path: str,
    model_type: str = "auto",
    device: str = "cpu",
    encoder: str = "resnet34",
    **kwargs,
):
    path_str = str(model_path)

    if model_type == "auto":
        ext = Path(path_str).suffix.lower()
        lower = path_str.lower()
        if "unet-plusplus" in lower or "unetplusplus" in lower:
            model_type = "unet_plusplus_v1"
        elif ext == ".pth" or "unet" in lower:
            model_type = "unet"
        elif ext == ".pt" or "yolo" in lower:
            model_type = "yolo"
        else:
            raise ValueError(
                f"Cannot auto-detect model type for {path_str}. "
                "Specify model_type explicitly."
            )

    if model_type == "unet":
        from src.inference.unet_predictor import UNetPredictor
        return UNetPredictor(path_str, device=device, encoder=encoder, **kwargs)

    if model_type == "unet_plusplus_v1":
        from src.inference.unet_plusplus_predictor import UNetPlusPlusPredictor
        return UNetPlusPlusPredictor(path_str, device=device, encoder=encoder, **kwargs)

    if model_type == "yolo":
        from src.inference.yolo_predictor import YOLOPredictor
        return YOLOPredictor(path_str, device=device, **kwargs)

    raise ValueError(f"Unsupported model type: {model_type}")
