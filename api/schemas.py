from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# --- Request Schemas ---

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





# --- Response Schemas ---

class ModelInfo(BaseModel):
    model_id: str = Field(description="Unique identifier for the loaded model")
    model_path: str = Field(description="Filesystem path to the model checkpoint")
    model_version: Optional[str] = Field(None, description="Model version name or tag")
    model_type: str = Field(description="Model architecture type (unet, unet_plusplus_v1, yolo)")
    device: str = Field(description="Compute device the model is loaded on (cpu/cuda)")
    loaded_at: Optional[datetime] = Field(None, description="Timestamp when the model was loaded")
    encoder: Optional[str] = Field(None, description="Encoder backbone name (for UNet-based models)")


class HealthResponse(BaseModel):
    status: str = Field(description="Server health status ('ok' if running)")
    device: str = Field(description="Compute device in use (cpu, cuda:0, etc.)")
    models_loaded: int = Field(description="Number of models currently loaded in memory")
    torch_version: str = Field(description="Installed PyTorch version")
    cuda_available: bool = Field(description="Whether CUDA GPU acceleration is available")


class InfoResponse(BaseModel):
    project: str = Field(description="Project name")
    version: str = Field(description="API version")
    device: str = Field(description="Compute device in use")
    available_checkpoints: list[dict] = Field(description="List of available model checkpoint files")
    default_model: str = Field(description="Path to the default model checkpoint")


class VisualizationUrls(BaseModel):
    overlay: str = Field(description="URL to overlay visualization (prediction mask on original image)")
    heatmap: str = Field(description="URL to confidence heatmap visualization")
    mask: str = Field(description="URL to binary prediction mask visualization")


class PredictResponse(BaseModel):
    result_id: str = Field(description="Unique result identifier for retrieving this prediction")
    image_file: str = Field(description="Original uploaded image filename")
    image_resolution: list[int] = Field(description="Image dimensions as [width, height]")
    crack_detected: bool = Field(description="Whether any crack was detected in the image")
    mean_confidence: float = Field(description="Mean prediction confidence score across detected crack pixels [0-1]")
    estimated_length_pixels: int = Field(description="Estimated crack length in pixels (from skeletonization)")
    estimated_average_width_pixels: float = Field(description="Estimated average crack width in pixels")
    crack_area_pixels: int = Field(description="Total detected crack area in pixels")
    centerline_coordinates: list[list[int]] = Field(description="Skeletonized crack centerline as [[x, y], ...] coordinates")
    processing_time_ms: float = Field(description="Inference processing time in milliseconds")
    evaluation_metrics: Optional[dict] = Field(None, description="Evaluation metrics (precision, recall, F1, IoU) — present when ground-truth mask was provided")
    visualizations: VisualizationUrls = Field(description="URLs to generated visualization images")


class ResultSummary(BaseModel):
    result_id: str = Field(description="Unique result identifier")
    image_file: str = Field(description="Original image filename")
    crack_detected: bool = Field(description="Whether crack was detected")
    mean_confidence: float = Field(description="Mean prediction confidence score")
    estimated_length_pixels: int = Field(description="Estimated crack length in pixels")
    crack_area_pixels: int = Field(description="Detected crack area in pixels")
    created_at: str = Field(description="ISO-8601 timestamp of when the result was created")
    processing_time_ms: float = Field(description="Inference processing time in milliseconds")


class ResultListResponse(BaseModel):
    total: int = Field(description="Total number of stored results")
    results: list[ResultSummary] = Field(description="Paginated list of result summaries")


class TaskStatusResponse(BaseModel):
    task_id: str = Field(description="Unique task identifier")
    status: str = Field(description="Current task status: pending, running, completed, failed, or cancelled")
    progress: Optional[float] = Field(None, description="Task progress percentage [0-100]")
    message: Optional[str] = Field(None, description="Status message or progress description")
    result: Optional[dict] = Field(None, description="Task result data (present when completed)")
    created_at: str = Field(description="ISO-8601 timestamp of task creation")
    completed_at: Optional[str] = Field(None, description="ISO-8601 timestamp of task completion")


class BatchPredictResponse(BaseModel):
    task_id: str = Field(description="Task identifier for tracking batch processing status")
    status: str = Field(description="Initial task status (always 'pending' on creation)")
    message: str = Field(description="Human-readable status message")
    total_images: int = Field(description="Total number of images submitted for batch processing")



