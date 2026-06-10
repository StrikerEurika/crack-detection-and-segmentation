from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# --- Request Schemas ---

class PredictParams(BaseModel):
    model_type: str = Field("auto", description='Model type: "auto", "unet", "unet_plusplus_v1", or "yolo"')
    model_version: Optional[str] = Field(None, description="Model version, relative path, or filename under checkpoints")
    tile_size: Optional[int] = Field(None, description="Tile size in pixels")
    overlap: Optional[int] = Field(None, description="Tile overlap in pixels")
    threshold: float = Field(0.5, description="Binarization probability threshold")
    min_area: int = Field(20, description="Minimum component area for noise removal")
    blend: str = Field("cosine", description='Blend mode: "cosine" or "average"')
    disable_color_marker_suppression: bool = Field(False)
    inpaint_markers: bool = Field(False)
    pre_inpaint_full: bool = Field(False)
    inpaint_radius: int = Field(5)
    marker_saturation_threshold: int = Field(85)
    marker_value_threshold: int = Field(55)
    suppress_marker_shapes: bool = Field(False)
    frame_side_coverage_threshold: float = Field(0.35)
    roundness_threshold: float = Field(0.58)
    conf: float = Field(0.18, description="YOLO confidence threshold")
    iou: float = Field(0.45, description="YOLO NMS IoU threshold")
    save_debug_mask: bool = Field(False)


class LoadModelRequest(BaseModel):
    model_path: str
    model_type: str = Field("auto", description='Model type: "auto", "unet", "unet_plusplus_v1", or "yolo"')
    encoder: str = Field("resnet34", description="UNet encoder backbone")





# --- Response Schemas ---

class ModelInfo(BaseModel):
    model_id: str
    model_path: str
    model_version: Optional[str] = None
    model_type: str
    device: str
    loaded_at: Optional[datetime] = None
    encoder: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: int
    torch_version: str
    cuda_available: bool


class InfoResponse(BaseModel):
    project: str
    version: str
    device: str
    available_checkpoints: list[dict]
    default_model: str


class VisualizationUrls(BaseModel):
    overlay: str
    heatmap: str
    mask: str


class PredictResponse(BaseModel):
    result_id: str
    image_file: str
    image_resolution: list[int]
    crack_detected: bool
    mean_confidence: float
    estimated_length_pixels: int
    estimated_average_width_pixels: float
    crack_area_pixels: int
    centerline_coordinates: list[list[int]]
    processing_time_ms: float
    evaluation_metrics: Optional[dict] = None
    visualizations: VisualizationUrls


class ResultSummary(BaseModel):
    result_id: str
    image_file: str
    crack_detected: bool
    mean_confidence: float
    estimated_length_pixels: int
    crack_area_pixels: int
    created_at: str
    processing_time_ms: float


class ResultListResponse(BaseModel):
    total: int
    results: list[ResultSummary]


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None
    result: Optional[dict] = None
    created_at: str
    completed_at: Optional[str] = None


class BatchPredictResponse(BaseModel):
    task_id: str
    status: str
    message: str
    total_images: int



