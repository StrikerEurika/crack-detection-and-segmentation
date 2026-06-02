import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.tiling.tile_generator import TileGenerator
from src.inference.marker_suppression import MarkerSuppressor


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO-seg tiled inference on large images with marker suppression")
    parser.add_argument("--image", type=str, required=True, help="Path to input large image")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/yolo26n-seg-train_2_weights/best.pt",
        help="YOLO model path",
    )
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size (match training size)")
    parser.add_argument("--overlap", type=int, default=96, help="Tile overlap in pixels")
    parser.add_argument("--conf", type=float, default=0.18, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--mask-prob-threshold",
        type=float,
        default=0.45,
        help="Final probability threshold for merged mask",
    )

    # Marker suppression / cleanup knobs
    parser.add_argument("--min-component-area", type=int, default=10, help="Drop tiny connected components")
    parser.add_argument(
        "--disable-color-marker-suppression",
        action="store_true",
        help="Disable suppression of highly saturated marker-like pixels",
    )
    parser.add_argument(
        "--marker-saturation-threshold",
        type=int,
        default=85,
        help="HSV saturation threshold used to detect colored markers",
    )
    parser.add_argument(
        "--marker-value-threshold",
        type=int,
        default=55,
        help="HSV value threshold used to ignore very dark pixels in marker detection",
    )
    parser.add_argument(
        "--frame-side-coverage-threshold",
        type=float,
        default=0.35,
        help="Border occupancy threshold for marker-frame rejection",
    )
    parser.add_argument(
        "--roundness-threshold",
        type=float,
        default=0.58,
        help="Circularity threshold for removing marker-like compact regions",
    )
    parser.add_argument(
        "--save-debug-mask",
        action="store_true",
        help="Also save a marker-suppressed binary debug mask",
    )
    return parser.parse_args()


def cosine_window(height: int, width: int, eps: float = 0.01) -> np.ndarray:
    y = np.sin(np.linspace(0, np.pi, height, dtype=np.float32))
    x = np.sin(np.linspace(0, np.pi, width, dtype=np.float32))
    y = np.clip(y, eps, 1.0)
    x = np.clip(x, eps, 1.0)
    return np.outer(y, x).astype(np.float32)


def filter_instance_mask(mask_prob: np.ndarray, tile_rgb: np.ndarray, args) -> np.ndarray:
    return MarkerSuppressor.filter_mask(
        mask_prob,
        tile_rgb,
        min_component_area=args.min_component_area,
        disable_color_suppression=args.disable_color_marker_suppression,
        saturation_threshold=args.marker_saturation_threshold,
        value_threshold=args.marker_value_threshold,
        frame_side_coverage_threshold=args.frame_side_coverage_threshold,
        roundness_threshold=args.roundness_threshold,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(args.image)
    if image is None:
        print(f"Failed to load image: {args.image}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    model = YOLO(args.model_path)

    generator = TileGenerator(tile_size=args.tile_size, overlap=args.overlap)
    coords = generator.get_tile_coordinates(h, w)

    full_prob = np.zeros((h, w), dtype=np.float32)
    full_weight = np.zeros((h, w), dtype=np.float32)

    debug_clean_union = np.zeros((h, w), dtype=np.uint8)

    for coord in coords:
        ymin, xmin, ymax, xmax = coord["ymin"], coord["xmin"], coord["ymax"], coord["xmax"]
        tile_rgb = image_rgb[ymin:ymax, xmin:xmax]
        tile_h, tile_w = tile_rgb.shape[:2]

        tile_prob = np.zeros((tile_h, tile_w), dtype=np.float32)

        results = model(tile_rgb, conf=args.conf, iou=args.iou, verbose=False)

        for result in results:
            if result.masks is None:
                continue

            for mask_tensor in result.masks.data:
                mask_prob = mask_tensor.cpu().numpy().astype(np.float32)
                mh, mw = mask_prob.shape
                if mh != tile_h or mw != tile_w:
                    mask_prob = cv2.resize(mask_prob, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

                cleaned = filter_instance_mask(mask_prob, tile_rgb, args)
                tile_prob = np.maximum(tile_prob, cleaned)

        w_tile = cosine_window(tile_h, tile_w)
        full_prob[ymin:ymax, xmin:xmax] += tile_prob * w_tile
        full_weight[ymin:ymax, xmin:xmax] += w_tile

        if args.save_debug_mask:
            debug_clean_union[ymin:ymax, xmin:xmax] = cv2.bitwise_or(
                debug_clean_union[ymin:ymax, xmin:xmax],
                (tile_prob >= args.mask_prob_threshold).astype(np.uint8) * 255,
            )

    merged_prob = full_prob / np.maximum(full_weight, 1e-6)
    full_mask = (merged_prob >= args.mask_prob_threshold).astype(np.uint8) * 255
    full_mask = MarkerSuppressor.remove_small_components(full_mask, args.min_component_area)

    stem = Path(args.image).stem
    cv2.imwrite(str(output_dir / f"{stem}_mask.png"), full_mask)

    if args.save_debug_mask:
        cv2.imwrite(str(output_dir / f"{stem}_debug_clean_union.png"), debug_clean_union)

    overlay = image_rgb.copy()
    overlay[full_mask > 127] = (255, 0, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay_bgr)

    print(f"Done. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
