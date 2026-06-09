import os
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

from src.ingestion.image_loader import ImageLoader
from src.preprocessing.normalize import ImagePreprocessor
from src.preprocessing.marker_inpaint import MarkerInpaint
from src.inference import get_predictor, PostProcessor, MarkerSuppressor
from src.evaluation.metrics import CrackMetrics
from src.visualization.overlays import Visualizer
from src.utils.logger import setup_logger

logger = setup_logger("inference")

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Inference and Evaluation Pipeline (supports U-Net, YOLO, and future models)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--mask", type=str, default=None, help="Path to optional ground truth mask file")
    parser.add_argument("--model-path", type=str, default="checkpoints/v1/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default="auto", choices=["auto", "unet", "yolo"], help="Model architecture type")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save output files")
    
    # Tiling options (None defaults will be resolved dynamically based on model type)
    parser.add_argument("--tile-size", type=int, default=None, help="Size of tiles for inference (defaults to 512 for U-Net, 640 for YOLO)")
    parser.add_argument("--overlap", type=int, default=None, help="Tile overlap in pixels (defaults to 64 for U-Net, 96 for YOLO)")
    parser.add_argument("--blend", type=str, default="cosine", choices=["average", "cosine"], help="Tile blending mode")
    
    # Model-specific parameters
    parser.add_argument("--encoder", type=str, default="resnet34", help="Backbone encoder network for UNet")
    parser.add_argument("--threshold", "--mask-prob-threshold", type=float, default=0.5, dest="threshold", help="Binarization/probability threshold")
    parser.add_argument("--conf", type=float, default=0.18, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    
    # Cleanup & suppression
    parser.add_argument("--min-area", "--min-component-area", type=int, default=20, dest="min_area", help="Minimum pixel area for noise removal")
    parser.add_argument("--disable-color-marker-suppression", action="store_true", help="Disable color-based marker suppression")
    parser.add_argument("--inpaint-markers", action="store_true", help="Inpaint colored markers before/during tile inference")
    parser.add_argument("--pre-inpaint-full", action="store_true", help="Inpaint markers on the full image before tiling")
    parser.add_argument("--inpaint-radius", type=int, default=5, help="Inpainting radius for marker removal")
    
    # Saturation and value thresholds
    parser.add_argument("--marker-saturation-threshold", "--marker-sat-threshold", type=int, default=85, dest="marker_sat_threshold", help="HSV saturation threshold for marker detection")
    parser.add_argument("--marker-value-threshold", "--marker-val-threshold", type=int, default=55, dest="marker_val_threshold", help="HSV value threshold for marker detection")
    
    # Shape-based suppression
    parser.add_argument("--suppress-marker-shapes", action="store_true", help="Remove compact/round marker-like shapes from predictions")
    parser.add_argument("--frame-side-coverage-threshold", "--frame-coverage", type=float, default=0.35, dest="frame_coverage", help="Border coverage threshold for frame-like marker rejection")
    parser.add_argument("--roundness-threshold", type=float, default=0.58, help="Circularity threshold for round marker rejection")
    
    parser.add_argument("--save-debug-mask", action="store_true", help="Writes a debug image showing per-tile cleaned union mask or marker mask")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Resolve model type and dynamic defaults
    model_path = Path(args.model_path)
    model_type = args.model_type
    if model_type == "auto":
        ext = model_path.suffix.lower()
        if ext == ".pth" or "unet" in str(model_path).lower():
            model_type = "unet"
        elif ext == ".pt" or "yolo" in str(model_path).lower():
            model_type = "yolo"
        else:
            logger.error(f"Could not auto-detect model type for checkpoint: {model_path}. Please specify --model-type explicitly.")
            return

    # Determine default tile size and overlap if not specified
    tile_size = args.tile_size
    if tile_size is None:
        tile_size = 640 if model_type == "yolo" else 512
        
    overlap = args.overlap
    if overlap is None:
        overlap = 96 if model_type == "yolo" else 64
        
    # 2. Load Image and Metadata
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found at {args.image}")
        return
        
    logger.info(f"Loading image {image_path.name}...")
    image = ImageLoader.load(str(image_path))
    metadata = ImageLoader.get_metadata(str(image_path))
    logger.info(f"Image loaded. Resolution: {metadata['width']}x{metadata['height']}, depth: {metadata['bit_depth']} bits.")
    
    # Apply pre-tiling marker inpainting if requested
    image_preprocessed = image
    if args.pre_inpaint_full:
        logger.info("Inpainting markers on full image before tiling...")
        image_preprocessed = MarkerInpaint.inpaint_tiled(
            image,
            tile_size=tile_size,
            overlap=overlap,
            saturation_threshold=args.marker_sat_threshold,
            value_threshold=args.marker_val_threshold,
            inpaint_radius=args.inpaint_radius,
        )
    
    # 3. Initialize Predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model checkpoint from {args.model_path} ({model_type}) on {device}...")
    try:
        predictor_kwargs = {}
        if model_type == "unet":
            predictor_kwargs = {
                "inpaint_markers": args.inpaint_markers,
                "marker_saturation_threshold": args.marker_sat_threshold,
                "marker_value_threshold": args.marker_val_threshold,
                "inpaint_radius": args.inpaint_radius,
            }
        elif model_type == "yolo":
            predictor_kwargs = {
                "conf": args.conf,
                "iou": args.iou,
                "min_component_area": args.min_area,
                "disable_color_marker_suppression": args.disable_color_marker_suppression,
                "marker_saturation_threshold": args.marker_sat_threshold,
                "marker_value_threshold": args.marker_val_threshold,
                "frame_side_coverage_threshold": args.frame_coverage,
                "roundness_threshold": args.roundness_threshold,
            }
            
        predictor = get_predictor(
            model_path=str(args.model_path),
            model_type=model_type,
            device=device,
            encoder=args.encoder,
            **predictor_kwargs
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 4. Run Inference
    logger.info(f"Running tiled inference on full image (tile_size={tile_size}, overlap={overlap})...")
    prob_map = predictor.predict_full_image(
        image_preprocessed,
        tile_size=tile_size,
        overlap=overlap,
        blend_mode=args.blend,
    )

    # 5. Post-processing
    logger.info("Binarizing and cleaning prediction map...")
    binary_mask = PostProcessor.binarize_probability_map(prob_map, threshold=args.threshold)
    cleaned_mask = PostProcessor.remove_noise(binary_mask, min_area=args.min_area)

    # Apply shape-based marker suppression if requested
    if args.suppress_marker_shapes:
        logger.info("Suppressing marker-like shapes from prediction...")
        cleaned_mask = PostProcessor.suppress_marker_shapes(
            cleaned_mask,
            frame_side_coverage_threshold=args.frame_coverage,
            roundness_threshold=args.roundness_threshold,
        )
        cleaned_mask = PostProcessor.remove_noise(cleaned_mask, min_area=args.min_area)
    
    logger.info("Skeletonizing centerline...")
    skeleton = PostProcessor.skeletonize(cleaned_mask)
    
    # Estimate dimensions
    dims = PostProcessor.estimate_dimensions(cleaned_mask, skeleton)
    logger.info(
        f"Inference complete. "
        f"Crack area: {dims['crack_area_pixels']} px, "
        f"Estimated length: {dims['length_pixels']} px, "
        f"Average width: {dims['average_width_pixels']} px"
    )
    
    # Mean confidence score for detected pixels
    confidence_score = float(np.mean(prob_map[cleaned_mask > 127])) if np.any(cleaned_mask > 127) else 0.0
    
    # 6. Extract pixel coordinates of the centerline (skeleton)
    coords = np.argwhere(skeleton > 0).tolist()  # list of [y, x]
    
    # 7. Save visualizations
    # Overlay predictions on image
    overlay = Visualizer.draw_mask_overlay(image, cleaned_mask, color=(255, 0, 0), alpha=0.4)
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved crack overlay image to {overlay_path}")
    
    # Heatmap of probabilities
    heatmap = (prob_map * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_path = output_dir / f"{image_path.stem}_heatmap.png"
    cv2.imwrite(str(heatmap_path), heatmap_colored)
    logger.info(f"Saved confidence heatmap to {heatmap_path}")
    
    # Binary mask
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), cleaned_mask)
    logger.info(f"Saved binary crack mask to {mask_path}")
    
    # Save debug mask if requested
    if args.save_debug_mask:
        debug_path = output_dir / f"{image_path.stem}_debug_clean_union.png"
        cv2.imwrite(str(debug_path), cleaned_mask)
        logger.info(f"Saved debug clean mask to {debug_path}")
        
    # 8. Evaluate if ground truth mask is provided
    eval_results = {}
    if args.mask:
        mask_path = Path(args.mask)
        if mask_path.exists():
            logger.info("Loading ground truth mask for evaluation...")
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Compute skeleton of ground truth for centerline distance
            gt_skeleton = PostProcessor.skeletonize(gt_mask)
            
            # Standard pixel metrics
            pixel_metrics = CrackMetrics.compute_pixel_metrics(cleaned_mask, gt_mask)
            # Centerline buffered metrics (tolerance = 3 pixels)
            buffered_metrics = CrackMetrics.compute_buffered_metrics(skeleton, gt_skeleton, tolerance=3.0)
            
            eval_results.update(pixel_metrics)
            eval_results.update(buffered_metrics)
            
            logger.info("--- Evaluation Metrics ---")
            logger.info(f"Pixel IoU: {eval_results['pixel_iou']:.4f}")
            logger.info(f"Pixel F1-Dice: {eval_results['pixel_f1_dice']:.4f}")
            logger.info(f"Pixel Precision: {eval_results['pixel_precision']:.4f}")
            logger.info(f"Pixel Recall: {eval_results['pixel_recall']:.4f}")
            logger.info(f"Centerline F1 (tolerance=3px): {eval_results['buffered_f1']:.4f}")
            logger.info(f"Centerline Precision: {eval_results['buffered_precision']:.4f}")
            logger.info(f"Centerline Recall: {eval_results['buffered_recall']:.4f}")
            
            # Error analysis visualization
            error_overlay = Visualizer.draw_error_analysis_overlay(image, gt_mask, cleaned_mask, alpha=0.5)
            error_path = output_dir / f"{image_path.stem}_error_analysis.png"
            cv2.imwrite(str(error_path), cv2.cvtColor(error_overlay, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved error analysis visualization to {error_path}")
        else:
            logger.warning(f"Ground truth mask file not found at {args.mask}. Skipping evaluation.")
            
    # 9. Export structured metadata
    result_metadata = {
        "image_file": image_path.name,
        "image_resolution": [metadata["width"], metadata["height"]],
        "crack_detected": int(dims["crack_area_pixels"] > 0),
        "mean_confidence": round(confidence_score, 4),
        "estimated_length_pixels": dims["length_pixels"],
        "estimated_average_width_pixels": dims["average_width_pixels"],
        "crack_area_pixels": dims["crack_area_pixels"],
        "evaluation_metrics": eval_results,
        "centerline_coordinates": coords  # list of [y, x]
    }
    
    # Save to JSON
    json_path = output_dir / f"{image_path.stem}_results.json"
    with open(json_path, "w") as f:
        json.dump(result_metadata, f, indent=4)
    logger.info(f"Saved structured results to {json_path}")
    
    # Save summary row to CSV
    csv_row = {
        "image_file": image_path.name,
        "crack_detected": dims["crack_area_pixels"] > 0,
        "mean_confidence": confidence_score,
        "estimated_length_pixels": dims["length_pixels"],
        "estimated_average_width_pixels": dims["average_width_pixels"],
        "crack_area_pixels": dims["crack_area_pixels"],
        **eval_results
    }
    # Exclude coordinate list in CSV for compactness
    csv_path = output_dir / "summary_results.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Drop matching rows if already exists, then append
        df = df[df["image_file"] != image_path.name]
        df = pd.concat([df, pd.DataFrame([csv_row])], ignore_index=True)
    else:
        df = pd.DataFrame([csv_row])
        
    df.to_csv(csv_path, index=False)
    logger.info(f"Appended results to CSV summary at {csv_path}")

if __name__ == "__main__":
    main()
