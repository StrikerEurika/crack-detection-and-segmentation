from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, Field


class PredictParams(BaseModel):
    model_type: str = Field("auto", description='Model type: "auto", "unet", "unet_plusplus_v1", or "yolo"')
    model_version: Optional[str] = Field(None, description="Model version, relative path, or filename under checkpoints")
    tile_size: Optional[int] = Field(None, description="Tile size in pixels for processing large images")
    overlap: Optional[int] = Field(None, description="Tile overlap in pixels to reduce seam artifacts")
    threshold: float = Field(0.5, description="Binarization probability threshold [0-1]")
    min_area: int = Field(20, description="Minimum connected component area in pixels for noise removal")
    blend: str = Field("cosine", description='Tile blending mode: "cosine" or "average"')
    conf: float = Field(0.18, description="YOLO confidence threshold (only used with YOLO models)")
    iou: float = Field(0.45, description="YOLO NMS IoU threshold (only used with YOLO models)")


class LoadModelRequest(BaseModel):
    model_path: str
    model_type: str = Field("auto", description='Model type: "auto", "unet", "unet_plusplus_v1", or "yolo"')
    encoder: str = Field("resnet34", description="UNet encoder backbone")
