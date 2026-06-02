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
from src.models.unet import CrackUnetModel
from src.inference.predictor import CrackPredictor
from src.inference.postprocess import PostProcessor
from src.evaluation.metrics import CrackMetrics
from src.visualization.overlays import Visualizer
from src.utils.logger import setup_logger

logger = setup_logger("inference")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference and Evaluation Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--mask", type=str, default=None, help="Path to optional ground truth mask file")
    parser.add_argument("--model-path", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--tile-size", type=int, default=512, help="Size of tiles for inference")
    parser.add_argument("--overlap", type=int, default=64, help="Tile overlap in pixels")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    parser.add_argument("--min-area", type=int, default=20, help="Minimum pixel area for crack component noise removal")
    parser.add_argument("--blend", type=str, default="cosine", choices=["average", "cosine"], help="Tile blending mode")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Backbone encoder network")

    # Marker suppression options
    parser.add_argument("--inpaint-markers", action="store_true", help="Inpaint colored markers before inference")
    parser.add_argument("--marker-sat-threshold", type=int, default=85, help="HSV saturation threshold for marker detection")
    parser.add_argument("--marker-val-threshold", type=int, default=55, help="HSV value threshold for marker detection")
    parser.add_argument("--inpaint-radius", type=int, default=5, help="Inpainting radius for marker removal")
    parser.add_argument("--suppress-marker-shapes", action="store_true", help="Remove compact/round marker-like shapes from predictions")
    parser.add_argument("--frame-coverage", type=float, default=0.35, help="Border coverage threshold for frame-like marker rejection")
    parser.add_argument("--roundness-threshold", type=float, default=0.58, help="Circularity threshold for round marker rejection")
    parser.add_argument("--pre-inpaint-full", action="store_true", help="Inpaint markers on full image before tiling (vs per-tile)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Image and Metadata
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
            tile_size=args.tile_size,
            overlap=args.overlap,
            saturation_threshold=args.marker_sat_threshold,
            value_threshold=args.marker_val_threshold,
            inpaint_radius=args.inpaint_radius,
        )
    
    # 2. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model checkpoint from {args.model_path} on {device}...")
    model = CrackUnetModel(encoder_name=args.encoder, encoder_weights=None, in_channels=3)
    try:
        model.load(args.model_path, device=device)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 3. Initialize Predictor and Run Inference
    predictor = CrackPredictor(
        model,
        device=device,
        inpaint_markers=args.inpaint_markers,
        marker_saturation_threshold=args.marker_sat_threshold,
        marker_value_threshold=args.marker_val_threshold,
        inpaint_radius=args.inpaint_radius,
    )
    logger.info("Running tiled inference on full image...")
    prob_map = predictor.predict_full_image(
        image_preprocessed,
        tile_size=args.tile_size,
        overlap=args.overlap,
        blend_mode=args.blend,
    )

    # 4. Post-processing
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
    
    # 5. Extract pixel coordinates of the centerline (skeleton)
    coords = np.argwhere(skeleton > 0).tolist()  # list of [y, x]
    
    # 6. Save visualizations
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
    
    # 7. Evaluate if ground truth mask is provided
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
            
    # 8. Export structured metadata
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
