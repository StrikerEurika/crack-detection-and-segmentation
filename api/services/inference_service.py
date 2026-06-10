from __future__ import annotations
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from api.config import settings
from api.schemas import PredictParams, PredictResponse, VisualizationUrls
from api.services.model_manager import model_manager
from api.services.result_store import result_store, ResultRecord
from api.services.task_manager import task_manager
from src.ingestion.image_loader import ImageLoader
from src.preprocessing.marker_inpaint import MarkerInpaint
from src.inference import PostProcessor
from src.evaluation.metrics import CrackMetrics
from src.visualization.overlays import Visualizer
from src.utils.logger import setup_logger

logger = setup_logger("api.inference_service")


def run_single_inference(
    image_array: np.ndarray,
    image_filename: str,
    params: PredictParams,
    mask_array: Optional[np.ndarray] = None,
) -> dict:
    result_id = str(uuid.uuid4())
    t_start = time.perf_counter()

    metadata = {
        "width": image_array.shape[1],
        "height": image_array.shape[0],
    }

    # Resolve model path and type dynamically
    requested_model = params.model_version if params.model_version else settings.default_model_path
    resolved_requested_path = model_manager.resolve_model_path(requested_model)
    
    # Check if the requested model is already loaded
    if model_manager.is_loaded and Path(model_manager.model_info["model_path"]).resolve() == resolved_requested_path.resolve():
        model_type = model_manager.model_info["model_type"]
        model_path = model_manager.model_info["model_path"]
    else:
        model_manager.load(
            model_path=str(resolved_requested_path),
            model_type=params.model_type,
        )
        model_info = model_manager.model_info
        model_type = model_info["model_type"]
        model_path = model_info["model_path"]

    tile_size = params.tile_size
    if tile_size is None:
        if model_type == "unet_plusplus_v1":
            tile_size = 1024
        elif model_type == "yolo":
            tile_size = settings.default_tile_size_yolo
        else:
            tile_size = settings.default_tile_size_unet

    overlap = params.overlap
    if overlap is None:
        if model_type == "unet_plusplus_v1":
            overlap = 204
        elif model_type == "yolo":
            overlap = settings.default_overlap_yolo
        else:
            overlap = settings.default_overlap_unet

    # Optional pre-inpainting
    image_processed = image_array
    if params.pre_inpaint_full:
        logger.info("Inpainting markers on full image before tiling...")
        image_processed = MarkerInpaint.inpaint_tiled(
            image_array,
            tile_size=tile_size,
            overlap=overlap,
            saturation_threshold=params.marker_saturation_threshold,
            value_threshold=params.marker_value_threshold,
            inpaint_radius=params.inpaint_radius,
        )

    # Get predictor and run inference
    predictor = model_manager.get_predictor()

    predictor_kwargs = {}
    if model_type == "unet":
        predictor_kwargs = {
            "inpaint_markers": params.inpaint_markers,
            "marker_saturation_threshold": params.marker_saturation_threshold,
            "marker_value_threshold": params.marker_value_threshold,
            "inpaint_radius": params.inpaint_radius,
        }

    prob_map = predictor.predict_full_image(
        image_processed,
        tile_size=tile_size,
        overlap=overlap,
        batch_size=4,
        blend_mode=params.blend,
    )

    # Post-processing
    binary_mask = PostProcessor.binarize_probability_map(prob_map, threshold=params.threshold)
    cleaned_mask = PostProcessor.remove_noise(binary_mask, min_area=params.min_area)

    if params.suppress_marker_shapes:
        cleaned_mask = PostProcessor.suppress_marker_shapes(
            cleaned_mask,
            frame_side_coverage_threshold=params.frame_side_coverage_threshold,
            roundness_threshold=params.roundness_threshold,
        )
        cleaned_mask = PostProcessor.remove_noise(cleaned_mask, min_area=params.min_area)

    skeleton = PostProcessor.skeletonize(cleaned_mask)
    dims = PostProcessor.estimate_dimensions(cleaned_mask, skeleton)

    confidence_score = float(np.mean(prob_map[cleaned_mask > 127])) if np.any(cleaned_mask > 127) else 0.0
    coords = np.argwhere(skeleton > 0).tolist()

    t_elapsed = (time.perf_counter() - t_start) * 1000

    # Save visualizations
    overlay = Visualizer.draw_mask_overlay(image_array, cleaned_mask, color=(255, 0, 0), alpha=0.4)
    result_store.save_visualization(result_id, "overlay", overlay)

    heatmap = (prob_map * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    result_store.save_visualization(result_id, "heatmap", heatmap_rgb)

    result_store.save_mask_visualization(result_id, "mask", cleaned_mask)

    # Evaluation if mask provided
    eval_results = {}
    if mask_array is not None:
        if model_type == "unet_plusplus_v1":
            from src.inference import MarkerSuppressor
            logger.info("Applying blue marker filtering to ground truth mask for consistency...")
            mask_array = MarkerSuppressor.filter_blue_markers_from_mask(image_array, mask_array)
            
        gt_skeleton = PostProcessor.skeletonize(mask_array)
        pixel_metrics = CrackMetrics.compute_pixel_metrics(cleaned_mask, mask_array)
        buffered_metrics = CrackMetrics.compute_buffered_metrics(skeleton, gt_skeleton, tolerance=3.0)
        eval_results.update(pixel_metrics)
        eval_results.update(buffered_metrics)

        error_overlay = Visualizer.draw_error_analysis_overlay(image_array, mask_array, cleaned_mask, alpha=0.5)
        result_store.save_visualization(result_id, "error_analysis", error_overlay)

    # Build response
    record = ResultRecord(
        result_id=result_id,
        image_file=image_filename,
        image_resolution=[metadata["width"], metadata["height"]],
        crack_detected=bool(dims["crack_area_pixels"] > 0),
        mean_confidence=round(confidence_score, 4),
        estimated_length_pixels=dims["length_pixels"],
        estimated_average_width_pixels=dims["average_width_pixels"],
        crack_area_pixels=dims["crack_area_pixels"],
        centerline_coordinates=coords,
        processing_time_ms=round(t_elapsed, 2),
        evaluation_metrics=eval_results,
    )
    result_store.store(record)

    base_url = f"/api/v1/results/{result_id}/visualization"
    response = PredictResponse(
        result_id=result_id,
        image_file=image_filename,
        image_resolution=record.image_resolution,
        crack_detected=record.crack_detected,
        mean_confidence=record.mean_confidence,
        estimated_length_pixels=record.estimated_length_pixels,
        estimated_average_width_pixels=record.estimated_average_width_pixels,
        crack_area_pixels=record.crack_area_pixels,
        centerline_coordinates=coords,
        processing_time_ms=record.processing_time_ms,
        evaluation_metrics=eval_results or None,
        visualizations=VisualizationUrls(
            overlay=f"{base_url}?type=overlay",
            heatmap=f"{base_url}?type=heatmap",
            mask=f"{base_url}?type=mask",
        ),
    )
    return response.model_dump()


def run_batch_inference(
    image_paths: list[Path],
    params: PredictParams,
    task=None,
) -> list[dict]:
    results = []
    total = len(image_paths)
    for i, img_path in enumerate(image_paths):
        logger.info(f"Batch processing {img_path.name} ({i+1}/{total})...")
        image = ImageLoader.load(str(img_path))
        result = run_single_inference(image, img_path.name, params)
        results.append(result)
        if task is not None:
            task_manager.update_progress(
                task.task_id,
                progress=(i + 1) / total * 100,
                message=f"Processed {i+1}/{total}: {img_path.name}",
            )
    return results
