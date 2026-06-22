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
    disable_color_marker_suppression: bool = Field(False, description="Disable automatic suppression of colored annotation markers")
    inpaint_markers: bool = Field(False, description="Apply inpainting to remove colored markers from tile regions")
    pre_inpaint_full: bool = Field(False, description="Apply marker inpainting on the full image before tiling")
    inpaint_radius: int = Field(5, description="Inpainting radius in pixels")
    marker_saturation_threshold: int = Field(85, description="Saturation threshold for colored marker detection in HSV space")
    marker_value_threshold: int = Field(55, description="Value (brightness) threshold for marker detection in HSV space")
    suppress_marker_shapes: bool = Field(False, description="Post-process to suppress crack-like shapes near frame edges")
    frame_side_coverage_threshold: float = Field(0.35, description="Frame-side coverage percentage threshold for marker suppression")
    roundness_threshold: float = Field(0.58, description="Roundness threshold to distinguish circular markers from cracks")
    conf: float = Field(0.18, description="YOLO confidence threshold (only used with YOLO models)")
    iou: float = Field(0.45, description="YOLO NMS IoU threshold (only used with YOLO models)")
    save_debug_mask: bool = Field(False, description="Save intermediate debug masks for troubleshooting")


class LoadModelRequest(BaseModel):
    model_path: str
    model_type: str = Field("auto", description='Model type: "auto", "unet", "unet_plusplus_v1", or "yolo"')
    encoder: str = Field("resnet34", description="UNet encoder backbone")
