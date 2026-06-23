from __future__ import annotations

import numpy as np
import torch

from src.inference.base_predictor import BasePredictor


class UNetPredictor(BasePredictor):
    """UNet predictor using segmentation_models_pytorch directly."""

    def __init__(self, model_path: str, device: str = "cpu", encoder: str = "resnet34"):
        super().__init__(model_path, device)
        self.encoder = encoder
        self.model = self._load_model()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load_model(self):
        import segmentation_models_pytorch as smp

        model = smp.Unet(
            encoder_name=self.encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model.to(self.device).eval()
        return model

    def predict_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        tensor = self._preprocess(tile_rgb)
        with torch.no_grad():
            logit = self.model(tensor)
            prob = torch.sigmoid(logit)
        return prob.squeeze().cpu().numpy().astype(np.float32)

    def _preprocess(self, tile_rgb: np.ndarray) -> torch.Tensor:
        img = tile_rgb.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
