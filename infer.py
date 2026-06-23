import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.ingestion.image_loader import ImageLoader
from src.inference import get_predictor, PostProcessor
from src.evaluation.metrics import CrackMetrics
from src.visualization.overlays import Visualizer
from src.utils.logger import setup_logger

logger = setup_logger("inference")


def parse_args():
    parser = argparse.ArgumentParser(description="Crack Segmentation Inference Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--mask", type=str, default=None, help="Path to optional ground truth mask file")
    parser.add_argument("--model-path", type=str, default="checkpoints/v1/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default="auto", choices=["auto", "unet", "unet_plusplus_v1", "yolo"], help="Model architecture type")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save output files")

    parser.add_argument("--tile-size", type=int, default=None, help="Size of tiles for inference")
    parser.add_argument("--overlap", type=int, default=None, help="Tile overlap in pixels")
    parser.add_argument("--blend", type=str, default="cosine", choices=["average", "cosine"], help="Tile blending mode")

    parser.add_argument("--encoder", type=str, default="resnet34", help="Backbone encoder network for UNet")
    parser.add_argument("--threshold", "--mask-prob-threshold", type=float, default=0.5, dest="threshold", help="Binarization/probability threshold")
    parser.add_argument("--conf", type=float, default=0.18, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")

    parser.add_argument("--min-area", "--min-component-area", type=int, default=20, dest="min_area", help="Minimum pixel area for noise removal")
    parser.add_argument("--save-debug-mask", action="store_true", help="Write a debug mask image")

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)
    model_type = args.model_type
    if model_type == "auto":
        ext = model_path.suffix.lower()
        if "unet-plusplus" in str(model_path).lower() or "unetplusplus" in str(model_path).lower():
            model_type = "unet_plusplus_v1"
        elif ext == ".pth" or "unet" in str(model_path).lower():
            model_type = "unet"
        elif ext == ".pt" or "yolo" in str(model_path).lower():
            model_type = "yolo"
        else:
            logger.error(f"Could not auto-detect model type for: {model_path}. Specify --model-type explicitly.")
            return

    tile_size = args.tile_size
    if tile_size is None:
        if model_type == "unet_plusplus_v1":
            tile_size = 1024
        elif model_type == "yolo":
            tile_size = 640
        else:
            tile_size = 512

    overlap = args.overlap
    if overlap is None:
        if model_type == "unet_plusplus_v1":
            overlap = 204
        elif model_type == "yolo":
            overlap = 96
        else:
            overlap = 64

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found at {args.image}")
        return

    logger.info(f"Loading image {image_path.name}...")
    image = ImageLoader.load(str(image_path))
    metadata = ImageLoader.get_metadata(str(image_path))
    logger.info(f"Image loaded. Resolution: {metadata['width']}x{metadata['height']}, depth: {metadata['bit_depth']} bits.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model from {args.model_path} ({model_type}) on {device}...")
    try:
        predictor_kwargs = {}
        if model_type == "yolo":
            predictor_kwargs = {"conf": args.conf, "iou": args.iou, "min_component_area": args.min_area}

        predictor = get_predictor(
            model_path=str(args.model_path),
            model_type=model_type,
            device=device,
            encoder=args.encoder,
            **predictor_kwargs,
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info(f"Running tiled inference (tile_size={tile_size}, overlap={overlap})...")
    prob_map = predictor.predict_full_image(
        image,
        tile_size=tile_size,
        overlap=overlap,
        blend_mode=args.blend,
    )

    logger.info("Binarizing and cleaning prediction map...")
    binary_mask = PostProcessor.binarize_probability_map(prob_map, threshold=args.threshold)
    cleaned_mask = PostProcessor.remove_noise(binary_mask, min_area=args.min_area)

    logger.info("Skeletonizing centerline...")
    skeleton = PostProcessor.skeletonize(cleaned_mask)

    dims = PostProcessor.estimate_dimensions(cleaned_mask, skeleton)
    logger.info(
        f"Inference complete. "
        f"Crack area: {dims['crack_area_pixels']} px, "
        f"Estimated length: {dims['length_pixels']} px, "
        f"Average width: {dims['average_width_pixels']} px"
    )

    confidence_score = float(np.mean(prob_map[cleaned_mask > 127])) if np.any(cleaned_mask > 127) else 0.0
    coords = np.argwhere(skeleton > 0).tolist()

    overlay = Visualizer.draw_mask_overlay(image, cleaned_mask, color=(255, 0, 0), alpha=0.4)
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved crack overlay to {overlay_path}")

    heatmap = (prob_map * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_path = output_dir / f"{image_path.stem}_heatmap.png"
    cv2.imwrite(str(heatmap_path), heatmap_colored)
    logger.info(f"Saved confidence heatmap to {heatmap_path}")

    mask_path = output_dir / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), cleaned_mask)
    logger.info(f"Saved binary crack mask to {mask_path}")

    if args.save_debug_mask:
        debug_path = output_dir / f"{image_path.stem}_debug_clean_union.png"
        cv2.imwrite(str(debug_path), cleaned_mask)
        logger.info(f"Saved debug mask to {debug_path}")

    eval_results = {}
    if args.mask:
        mask_path = Path(args.mask)
        if mask_path.exists():
            logger.info("Loading ground truth mask for evaluation...")
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            gt_skeleton = PostProcessor.skeletonize(gt_mask)
            pixel_metrics = CrackMetrics.compute_pixel_metrics(cleaned_mask, gt_mask)
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

            error_overlay = Visualizer.draw_error_analysis_overlay(image, gt_mask, cleaned_mask, alpha=0.5)
            error_path = output_dir / f"{image_path.stem}_error_analysis.png"
            cv2.imwrite(str(error_path), cv2.cvtColor(error_overlay, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved error analysis to {error_path}")
        else:
            logger.warning(f"Ground truth mask not found at {args.mask}. Skipping evaluation.")

    result_metadata = {
        "image_file": image_path.name,
        "image_resolution": [metadata["width"], metadata["height"]],
        "crack_detected": dims["crack_area_pixels"] > 0,
        "mean_confidence": round(confidence_score, 4),
        "estimated_length_pixels": dims["length_pixels"],
        "estimated_average_width_pixels": dims["average_width_pixels"],
        "crack_area_pixels": dims["crack_area_pixels"],
        "evaluation_metrics": eval_results,
        "centerline_coordinates": coords,
    }

    json_path = output_dir / f"{image_path.stem}_results.json"
    with open(json_path, "w") as f:
        json.dump(result_metadata, f, indent=4)
    logger.info(f"Saved results to {json_path}")

    csv_row = {
        "image_file": image_path.name,
        "crack_detected": dims["crack_area_pixels"] > 0,
        "mean_confidence": confidence_score,
        "estimated_length_pixels": dims["length_pixels"],
        "estimated_average_width_pixels": dims["average_width_pixels"],
        "crack_area_pixels": dims["crack_area_pixels"],
        **eval_results,
    }
    csv_path = output_dir / "summary_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["image_file"] != image_path.name]
        df = pd.concat([df, pd.DataFrame([csv_row])], ignore_index=True)
    else:
        df = pd.DataFrame([csv_row])
    df.to_csv(csv_path, index=False)
    logger.info(f"Appended results to {csv_path}")


if __name__ == "__main__":
    main()
