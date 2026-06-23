from __future__ import annotations

import numpy as np
import torch

from src.inference.base_predictor import BasePredictor


class UNetPlusPlusPredictor(BasePredictor):
    """UNet++ predictor using segmentation_models_pytorch directly.

    Expects raw [0-255] float input (no ImageNet normalization).
    """

    def __init__(self, model_path: str, device: str = "cpu", encoder: str = "efficientnet-b4"):
        super().__init__(model_path, device)
        self.encoder = encoder
        self.model = self._load_model()

    def _load_model(self):
        import segmentation_models_pytorch as smp

        model = smp.UnetPlusPlus(
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

        has_model_prefix = any(k.startswith("model.") for k in state.keys())
        if not has_model_prefix:
            model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=False)

        model.to(self.device).eval()
        return model

    def predict_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(
            tile_rgb.astype(np.float32).transpose(2, 0, 1)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(tensor)
            prob = torch.sigmoid(logit)
        return prob.squeeze().cpu().numpy().astype(np.float32)
