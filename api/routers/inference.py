from __future__ import annotations
import time
import traceback
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import numpy as np

from api.config import settings
from api.schemas import PredictParams, PredictResponse, BatchPredictResponse
from api.services.inference_service import (
    run_single_inference,
    run_batch_inference,
)
from api.services.task_manager import task_manager
from src.ingestion.image_loader import ImageLoader
from src.utils.logger import setup_logger

router = APIRouter(prefix="/predict", tags=["Inference"])
logger = setup_logger("api.inference")


def _build_params(
    model_type: str,
    model_version: Optional[str],
    tile_size: Optional[int],
    overlap: Optional[int],
    threshold: float,
    min_area: int,
    blend: str,
    disable_color_marker_suppression: bool,
    inpaint_markers: bool,
    pre_inpaint_full: bool,
    inpaint_radius: int,
    marker_saturation_threshold: int,
    marker_value_threshold: int,
    suppress_marker_shapes: bool,
    frame_side_coverage_threshold: float,
    roundness_threshold: float,
    conf: float,
    iou: float,
    save_debug_mask: bool,
) -> PredictParams:
    return PredictParams(
        model_type=model_type,
        model_version=model_version,
        tile_size=tile_size,
        overlap=overlap,
        threshold=threshold,
        min_area=min_area,
        blend=blend,
        disable_color_marker_suppression=disable_color_marker_suppression,
        inpaint_markers=inpaint_markers,
        pre_inpaint_full=pre_inpaint_full,
        inpaint_radius=inpaint_radius,
        marker_saturation_threshold=marker_saturation_threshold,
        marker_value_threshold=marker_value_threshold,
        suppress_marker_shapes=suppress_marker_shapes,
        frame_side_coverage_threshold=frame_side_coverage_threshold,
        roundness_threshold=roundness_threshold,
        conf=conf,
        iou=iou,
        save_debug_mask=save_debug_mask,
    )


async def _save_upload(file: UploadFile) -> Path:
    contents = await file.read()
    temp_dir = settings.api_results_dir / "_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename
    with open(temp_path, "wb") as f:
        f.write(contents)
    return temp_path


@router.post(
    "",
    response_model=PredictResponse,
    summary="Run Single Inference",
    description="Upload a single microscopic crack image for AI-powered crack detection. Supports UNet, UNet++, and YOLO segmentation models. The image is processed using tiled inference to handle large resolutions, and returns crack detection results along with visualization URLs.",
    response_description="Crack detection results with visualization URLs",
    responses={400: {"description": "Failed to load or process the image"}},
)
async def predict_single(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, TIFF, etc.)"),
    model_type: str = Form("auto", description='Model type: "auto", "unet", "unet_plusplus_v1", or "yolo". When "auto", automatically detects from the loaded model or defaults to UNet.'),
    model_version: Optional[str] = Form(None, description="Specific model version, relative path, or filename to load. E.g. 'v1', 'v1/best_model.pth'. Uses default model if not specified."),
    tile_size: Optional[int] = Form(None, description="Tile size in pixels for processing large images. Defaults to 512 (UNet), 1024 (UNet++), or 640 (YOLO)."),
    overlap: Optional[int] = Form(None, description="Tile overlap in pixels to reduce seam artifacts. Defaults to 64 (UNet), 204 (UNet++), or 96 (YOLO)."),
    threshold: float = Form(0.5, description="Probability threshold [0-1] for binarizing the crack prediction map. Higher values reduce false positives."),
    min_area: int = Form(20, description="Minimum connected component area in pixels. Components smaller than this are removed as noise."),
    blend: str = Form("cosine", description='Tile blending mode: "cosine" (smooth cosine window) or "average" (simple averaging). Cosine is recommended.'),
    disable_color_marker_suppression: bool = Form(False, description="Disable automatic suppression of colored annotation markers during inference."),
    inpaint_markers: bool = Form(False, description="Apply inpainting to remove colored markers from tile regions before prediction."),
    pre_inpaint_full: bool = Form(False, description="Apply marker inpainting on the full image before tiling (slower but more thorough)."),
    inpaint_radius: int = Form(5, description="Radius in pixels for the marker inpainting operation."),
    marker_saturation_threshold: int = Form(85, description="Saturation threshold for detecting colored markers in HSV space."),
    marker_value_threshold: int = Form(55, description="Value (brightness) threshold for detecting colored markers in HSV space."),
    suppress_marker_shapes: bool = Form(False, description="Enable post-processing to suppress crack-like shapes near frame edges (reduces false positives from frame artifacts)."),
    frame_side_coverage_threshold: float = Form(0.35, description="Threshold for frame-side coverage percentage when suppressing marker shapes."),
    roundness_threshold: float = Form(0.58, description="Roundness threshold for distinguishing circular markers from elongated cracks."),
    conf: float = Form(0.18, description="YOLO confidence threshold. Only used when model_type is 'yolo'."),
    iou: float = Form(0.45, description="YOLO NMS IoU threshold. Only used when model_type is 'yolo'."),
    save_debug_mask: bool = Form(False, description="Save intermediate debug masks for troubleshooting."),
):
    temp_path = await _save_upload(file)

    try:
        image = ImageLoader.load(str(temp_path))
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

    temp_path.unlink(missing_ok=True)

    params = _build_params(
        model_type, model_version, tile_size, overlap, threshold, min_area, blend,
        disable_color_marker_suppression, inpaint_markers, pre_inpaint_full,
        inpaint_radius, marker_saturation_threshold, marker_value_threshold,
        suppress_marker_shapes, frame_side_coverage_threshold, roundness_threshold,
        conf, iou, save_debug_mask,
    )

    try:
        t0 = time.perf_counter()
        result = run_single_inference(image, file.filename, params)
        t_elapsed = (time.perf_counter() - t0) * 1000
        result["processing_time_ms"] = round(t_elapsed, 2)
        return PredictResponse(**result)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/and-evaluate",
    response_model=PredictResponse,
    summary="Run Inference with Evaluation",
    description="Upload a microscopic crack image AND its ground-truth mask to run inference and evaluate prediction accuracy. Returns crack detection results along with pixel-level and centerline-buffered evaluation metrics (precision, recall, F1, IoU, etc.).",
    response_description="Crack detection results with evaluation metrics",
    responses={400: {"description": "Failed to load image or mask files"}},
)
async def predict_and_evaluate(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, TIFF, etc.)"),
    mask_file: UploadFile = File(..., description="Ground-truth binary mask image for the crack regions (JPEG, PNG, etc.). Used to compute evaluation metrics."),
    model_type: str = Form("auto", description='Model type: "auto", "unet", "unet_plusplus_v1", or "yolo".'),
    model_version: Optional[str] = Form(None, description="Specific model version or relative path to load."),
    tile_size: Optional[int] = Form(None, description="Tile size in pixels. Auto-selected based on model type if not specified."),
    overlap: Optional[int] = Form(None, description="Tile overlap in pixels."),
    threshold: float = Form(0.5, description="Probability threshold [0-1] for crack binarization."),
    min_area: int = Form(20, description="Minimum component area for noise removal."),
    blend: str = Form("cosine", description='Blend mode: "cosine" or "average".'),
    disable_color_marker_suppression: bool = Form(False, description="Disable color annotation marker suppression."),
    inpaint_markers: bool = Form(False, description="Inpaint colored markers before prediction."),
    pre_inpaint_full: bool = Form(False, description="Inpaint markers on full image before tiling."),
    inpaint_radius: int = Form(5, description="Inpainting radius in pixels."),
    marker_saturation_threshold: int = Form(85, description="Saturation threshold for marker detection."),
    marker_value_threshold: int = Form(55, description="Value threshold for marker detection."),
    suppress_marker_shapes: bool = Form(False, description="Suppress frame-edge crack-like shapes."),
    frame_side_coverage_threshold: float = Form(0.35, description="Frame-side coverage threshold."),
    roundness_threshold: float = Form(0.58, description="Roundness threshold for marker shape suppression."),
    conf: float = Form(0.18, description="YOLO confidence threshold."),
    iou: float = Form(0.45, description="YOLO NMS IoU threshold."),
    save_debug_mask: bool = Form(False, description="Save debug masks."),
):
    temp_img = await _save_upload(file)
    temp_mask = await _save_upload(mask_file)

    try:
        image = ImageLoader.load(str(temp_img))
        mask = ImageLoader.load(str(temp_mask))
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        temp_img.unlink(missing_ok=True)
        temp_mask.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to load files: {e}")

    temp_img.unlink(missing_ok=True)
    temp_mask.unlink(missing_ok=True)

    params = _build_params(
        model_type, model_version, tile_size, overlap, threshold, min_area, blend,
        disable_color_marker_suppression, inpaint_markers, pre_inpaint_full,
        inpaint_radius, marker_saturation_threshold, marker_value_threshold,
        suppress_marker_shapes, frame_side_coverage_threshold, roundness_threshold,
        conf, iou, save_debug_mask,
    )

    try:
        result = run_single_inference(image, file.filename, params, mask_array=mask)
        return PredictResponse(**result)
    except Exception as e:
        logger.error(f"Inference + evaluation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Run Batch Inference (Async)",
    description="Upload multiple images for batch crack detection. Returns immediately with a task_id for tracking progress. Inference runs asynchronously in the background. Check task status via GET /api/v1/tasks/{task_id}.",
    response_description="Batch task creation confirmation with task_id for status polling",
)
async def predict_batch(
    files: list[UploadFile] = File(..., description="Multiple image files to process in batch (JPEG, PNG, TIFF, etc.)"),
    model_type: str = Form("auto", description='Model type: "auto", "unet", "unet_plusplus_v1", or "yolo".'),
    model_version: Optional[str] = Form(None, description="Specific model version or relative path to load."),
    tile_size: Optional[int] = Form(None, description="Tile size in pixels."),
    overlap: Optional[int] = Form(None, description="Tile overlap in pixels."),
    threshold: float = Form(0.5, description="Probability threshold for crack binarization."),
    min_area: int = Form(20, description="Minimum component area for noise removal."),
    blend: str = Form("cosine", description='Blend mode: "cosine" or "average".'),
    disable_color_marker_suppression: bool = Form(False, description="Disable color annotation marker suppression."),
    inpaint_markers: bool = Form(False, description="Inpaint colored markers before prediction."),
    pre_inpaint_full: bool = Form(False, description="Inpaint markers on full image before tiling."),
    inpaint_radius: int = Form(5, description="Inpainting radius in pixels."),
    marker_saturation_threshold: int = Form(85, description="Saturation threshold for marker detection."),
    marker_value_threshold: int = Form(55, description="Value threshold for marker detection."),
    suppress_marker_shapes: bool = Form(False, description="Suppress frame-edge crack-like shapes."),
    frame_side_coverage_threshold: float = Form(0.35, description="Frame-side coverage threshold."),
    roundness_threshold: float = Form(0.58, description="Roundness threshold for marker shape suppression."),
    conf: float = Form(0.18, description="YOLO confidence threshold."),
    iou: float = Form(0.45, description="YOLO NMS IoU threshold."),
    save_debug_mask: bool = Form(False, description="Save debug masks."),
):
    temp_dir = settings.api_results_dir / "_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for f in files:
        dst = temp_dir / f.filename
        content = await f.read()
        with open(dst, "wb") as fh:
            fh.write(content)
        saved_paths.append(dst)

    params = _build_params(
        model_type, model_version, tile_size, overlap, threshold, min_area, blend,
        disable_color_marker_suppression, inpaint_markers, pre_inpaint_full,
        inpaint_radius, marker_saturation_threshold, marker_value_threshold,
        suppress_marker_shapes, frame_side_coverage_threshold, roundness_threshold,
        conf, iou, save_debug_mask,
    )

    task_id = await task_manager.submit(
        _batch_worker,
        saved_paths,
        params,
    )

    return BatchPredictResponse(
        task_id=task_id,
        status="pending",
        message=f"Batch processing {len(saved_paths)} images",
        total_images=len(saved_paths),
    )


def _batch_worker(image_paths: list[Path], params: PredictParams, task=None):
    return run_batch_inference(image_paths, params, task=task)
