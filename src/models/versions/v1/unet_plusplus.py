import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class CrackUnetPlusPlusModel(nn.Module):
    def __init__(self, encoder_name: str = "efficientnet-b0", encoder_weights: str = None, in_channels: int = 3):
        """Initializes the Unet++ crack segmentation model using segmentation-models-pytorch.
        
        Args:
            encoder_name: Backbone encoder network (e.g., 'efficientnet-b0').
            encoder_weights: Pretrained weights name (e.g., None).
            in_channels: Number of input channels (e.g., 3 for RGB).
        """
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        
        self.model = smp.UnetPlusPlus(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=1,  # Binary segmentation: crack vs background
            activation=None  # Output raw logits; activation is handled in loss or inference
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, 1, height, width)
        """
        return self.model(x)

    def save(self, path: str):
        """Saves model weights to path."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu"):
        """Loads model weights from path."""
        state_dict = torch.load(path, map_location=device)
        # Check if state_dict keys have the 'model.' prefix.
        # If not, load directly into self.model.
        has_model_prefix = any(k.startswith("model.") for k in state_dict.keys())
        if not has_model_prefix:
            self.model.load_state_dict(state_dict)
        else:
            self.load_state_dict(state_dict)

