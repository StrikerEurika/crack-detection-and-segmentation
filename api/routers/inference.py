from __future__ import annotations
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


@router.post("", response_model=PredictResponse)
async def predict_single(
    file: UploadFile = File(...),
    model_type: str = Form("auto"),
    tile_size: Optional[int] = Form(None),
    overlap: Optional[int] = Form(None),
    threshold: float = Form(0.5),
    min_area: int = Form(20),
    blend: str = Form("cosine"),
    disable_color_marker_suppression: bool = Form(False),
    inpaint_markers: bool = Form(False),
    pre_inpaint_full: bool = Form(False),
    inpaint_radius: int = Form(5),
    marker_saturation_threshold: int = Form(85),
    marker_value_threshold: int = Form(55),
    suppress_marker_shapes: bool = Form(False),
    frame_side_coverage_threshold: float = Form(0.35),
    roundness_threshold: float = Form(0.58),
    conf: float = Form(0.18),
    iou: float = Form(0.45),
    save_debug_mask: bool = Form(False),
):
    contents = await file.read()
    temp_dir = settings.api_results_dir / "_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename
    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        image = ImageLoader.load(str(temp_path))
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

    temp_path.unlink(missing_ok=True)

    params = PredictParams(
        model_type=model_type,
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

    try:
        import time
        t0 = time.perf_counter()
        result = run_single_inference(image, file.filename, params)
        t_elapsed = (time.perf_counter() - t0) * 1000
        result["processing_time_ms"] = round(t_elapsed, 2)
        return PredictResponse(**result)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/and-evaluate", response_model=PredictResponse)
async def predict_and_evaluate(
    file: UploadFile = File(...),
    mask_file: UploadFile = File(...),
    model_type: str = Form("auto"),
    tile_size: Optional[int] = Form(None),
    overlap: Optional[int] = Form(None),
    threshold: float = Form(0.5),
    min_area: int = Form(20),
    blend: str = Form("cosine"),
    disable_color_marker_suppression: bool = Form(False),
    inpaint_markers: bool = Form(False),
    pre_inpaint_full: bool = Form(False),
    inpaint_radius: int = Form(5),
    marker_saturation_threshold: int = Form(85),
    marker_value_threshold: int = Form(55),
    suppress_marker_shapes: bool = Form(False),
    frame_side_coverage_threshold: float = Form(0.35),
    roundness_threshold: float = Form(0.58),
    conf: float = Form(0.18),
    iou: float = Form(0.45),
    save_debug_mask: bool = Form(False),
):
    contents = await file.read()
    mask_contents = await mask_file.read()
    temp_dir = settings.api_results_dir / "_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_img = temp_dir / file.filename
    with open(temp_img, "wb") as f:
        f.write(contents)

    temp_mask = temp_dir / mask_file.filename
    with open(temp_mask, "wb") as f:
        f.write(mask_contents)

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

    params = PredictParams(
        model_type=model_type,
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

    try:
        result = run_single_inference(image, file.filename, params, mask_array=mask)
        return PredictResponse(**result)
    except Exception as e:
        logger.error(f"Inference + evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchPredictResponse)
async def predict_batch(
    files: list[UploadFile] = File(...),
    model_type: str = Form("auto"),
    tile_size: Optional[int] = Form(None),
    overlap: Optional[int] = Form(None),
    threshold: float = Form(0.5),
    min_area: int = Form(20),
    blend: str = Form("cosine"),
    disable_color_marker_suppression: bool = Form(False),
    inpaint_markers: bool = Form(False),
    pre_inpaint_full: bool = Form(False),
    inpaint_radius: int = Form(5),
    marker_saturation_threshold: int = Form(85),
    marker_value_threshold: int = Form(55),
    suppress_marker_shapes: bool = Form(False),
    frame_side_coverage_threshold: float = Form(0.35),
    roundness_threshold: float = Form(0.58),
    conf: float = Form(0.18),
    iou: float = Form(0.45),
    save_debug_mask: bool = Form(False),
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

    params = PredictParams(
        model_type=model_type,
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
