import os
from pathlib import Path
from src.inference.unet_predictor import UnetPredictor
from src.inference.yolo_predictor import YoloPredictor

def get_predictor(
    model_path: str,
    model_type: str = "auto",
    device: str = "cpu",
    encoder: str = "resnet34",
    **kwargs
):
    """Factory function to load model and get the appropriate predictor.
    
    Args:
        model_path: Path to model checkpoint/weights.
        model_type: "unet", "yolo", or "auto" to infer from file path/extension.
        device: Device to run inference on (e.g., "cuda" or "cpu").
        encoder: Encoder backbone network (used for UNet).
        **kwargs: Additional parameters passed to the predictor's constructor.
    """
    model_path_str = str(model_path)
    
    if model_type == "auto":
        ext = Path(model_path_str).suffix.lower()
        if ext == ".pth" or "unet" in model_path_str.lower():
            model_type = "unet"
        elif ext == ".pt" or "yolo" in model_path_str.lower():
            model_type = "yolo"
        else:
            raise ValueError(
                f"Could not auto-detect model type for {model_path_str}. "
                f"Please specify --model-type explicitly."
            )
            
    if model_type == "unet":
        from src.models import CrackUnetModel
        model = CrackUnetModel(encoder_name=encoder, encoder_weights=None, in_channels=3)
        model.load(model_path_str, device=device)
        return UnetPredictor(model=model, device=device, **kwargs)
        
    elif model_type == "yolo":
        # Lazy import of YOLO so that ultralytics is not required if only running U-Net
        from ultralytics import YOLO
        model = YOLO(model_path_str)
        # Note: YOLO manages its device internally
        return YoloPredictor(model=model, **kwargs)
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
