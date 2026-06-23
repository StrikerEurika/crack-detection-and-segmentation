from __future__ import annotations
import time
import traceback
from pathlib import Path
from typing import Optional

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
    conf: float,
    iou: float,
) -> PredictParams:
    return PredictParams(
        model_type=model_type,
        model_version=model_version,
        tile_size=tile_size,
        overlap=overlap,
        threshold=threshold,
        min_area=min_area,
        blend=blend,
        conf=conf,
        iou=iou,
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
    description="Upload a single microscopic crack image for AI-powered crack detection. Supports UNet, UNet++, and YOLO segmentation models.",
    response_description="Crack detection results with visualization URLs",
    responses={400: {"description": "Failed to load or process the image"}},
)
async def predict_single(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, TIFF, etc.)"),
    model_type: str = Form("auto", description='Model type: "auto", "unet", "unet_plusplus_v1", or "yolo".'),
    model_version: Optional[str] = Form(None, description="Specific model version, relative path, or filename to load."),
    tile_size: Optional[int] = Form(None, description="Tile size in pixels for processing large images."),
    overlap: Optional[int] = Form(None, description="Tile overlap in pixels to reduce seam artifacts."),
    threshold: float = Form(0.5, description="Probability threshold [0-1] for binarizing the crack prediction map."),
    min_area: int = Form(20, description="Minimum connected component area in pixels. Components smaller than this are removed."),
    blend: str = Form("cosine", description='Tile blending mode: "cosine" or "average".'),
    conf: float = Form(0.18, description="YOLO confidence threshold. Only used when model_type is 'yolo'."),
    iou: float = Form(0.45, description="YOLO NMS IoU threshold. Only used when model_type is 'yolo'."),
):
    temp_path = await _save_upload(file)

    try:
        image = ImageLoader.load(str(temp_path))
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

    temp_path.unlink(missing_ok=True)

    params = _build_params(model_type, model_version, tile_size, overlap, threshold, min_area, blend, conf, iou)

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
    description="Upload a microscopic crack image AND its ground-truth mask to run inference and evaluate prediction accuracy.",
    response_description="Crack detection results with evaluation metrics",
    responses={400: {"description": "Failed to load image or mask files"}},
)
async def predict_and_evaluate(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, TIFF, etc.)"),
    mask_file: UploadFile = File(..., description="Ground-truth binary mask image for the crack regions."),
    model_type: str = Form("auto", description='Model type: "auto", "unet", "unet_plusplus_v1", or "yolo".'),
    model_version: Optional[str] = Form(None, description="Specific model version or relative path to load."),
    tile_size: Optional[int] = Form(None, description="Tile size in pixels."),
    overlap: Optional[int] = Form(None, description="Tile overlap in pixels."),
    threshold: float = Form(0.5, description="Probability threshold [0-1] for crack binarization."),
    min_area: int = Form(20, description="Minimum component area for noise removal."),
    blend: str = Form("cosine", description='Blend mode: "cosine" or "average".'),
    conf: float = Form(0.18, description="YOLO confidence threshold."),
    iou: float = Form(0.45, description="YOLO NMS IoU threshold."),
):
    temp_img = await _save_upload(file)
    temp_mask = await _save_upload(mask_file)

    try:
        image = ImageLoader.load(str(temp_img))
        mask = ImageLoader.load(str(temp_mask))
        if len(mask.shape) == 3:
            import cv2
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        temp_img.unlink(missing_ok=True)
        temp_mask.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to load files: {e}")

    temp_img.unlink(missing_ok=True)
    temp_mask.unlink(missing_ok=True)

    params = _build_params(model_type, model_version, tile_size, overlap, threshold, min_area, blend, conf, iou)

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
    description="Upload multiple images for batch crack detection. Returns immediately with a task_id for tracking progress.",
    response_description="Batch task creation confirmation with task_id for status polling",
)
async def predict_batch(
    files: list[UploadFile] = File(..., description="Multiple image files to process in batch."),
    model_type: str = Form("auto", description='Model type: "auto", "unet", "unet_plusplus_v1", or "yolo".'),
    model_version: Optional[str] = Form(None, description="Specific model version or relative path to load."),
    tile_size: Optional[int] = Form(None, description="Tile size in pixels."),
    overlap: Optional[int] = Form(None, description="Tile overlap in pixels."),
    threshold: float = Form(0.5, description="Probability threshold for crack binarization."),
    min_area: int = Form(20, description="Minimum component area for noise removal."),
    blend: str = Form("cosine", description='Blend mode: "cosine" or "average".'),
    conf: float = Form(0.18, description="YOLO confidence threshold."),
    iou: float = Form(0.45, description="YOLO NMS IoU threshold."),
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

    params = _build_params(model_type, model_version, tile_size, overlap, threshold, min_area, blend, conf, iou)

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
