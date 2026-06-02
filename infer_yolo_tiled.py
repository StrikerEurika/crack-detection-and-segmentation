import argparse
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

from src.tiling.tile_generator import TileGenerator

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26-seg tiled inference on large images")
    parser.add_argument("--image", type=str, required=True, help="Path to input large image")
    parser.add_argument("--model-path", type=str, default="checkpoints/train_2_weights/best.pt", help="YOLO model path")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size (match your training size)")
    parser.add_argument("--overlap", type=int, default=64, help="Tile overlap in pixels")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load large image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Failed to load image: {args.image}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    # 2. Load YOLO model
    model = YOLO(args.model_path)

    # 3. Generate tile coordinates
    generator = TileGenerator(tile_size=args.tile_size, overlap=args.overlap)
    coords = generator.get_tile_coordinates(h, w)

    # 4. Run YOLO on each tile, collect masks in full-image coordinates
    full_mask = np.zeros((h, w), dtype=np.uint8)

    for coord in coords:
        tile = image_rgb[coord["ymin"]:coord["ymax"], coord["xmin"]:coord["xmax"]]

        results = model(tile, conf=args.conf, iou=args.iou, verbose=False)

        for result in results:
            if result.masks is None:
                continue
            # Each mask is relative to the tile — resize to tile size and place in full image
            for mask_tensor in result.masks.data:
                # mask_tensor: (H, W) float, values ~ [0, 1]
                mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                # Resize if YOLO internal size differs from tile size
                mh, mw = mask_np.shape
                if mh != args.tile_size or mw != args.tile_size:
                    mask_np = cv2.resize(mask_np, (args.tile_size, args.tile_size),
                                         interpolation=cv2.INTER_NEAREST)
                # Place into full mask
                tile_mask = mask_np[:coord["ymax"] - coord["ymin"],
                                    :coord["xmax"] - coord["xmin"]]
                full_mask[coord["ymin"]:coord["ymax"],
                          coord["xmin"]:coord["xmax"]] = cv2.bitwise_or(
                    full_mask[coord["ymin"]:coord["ymax"],
                              coord["xmin"]:coord["xmax"]],
                    tile_mask
                )

    # 5. Save results
    stem = Path(args.image).stem
    cv2.imwrite(str(output_dir / f"{stem}_mask.png"), full_mask)

    overlay = image_rgb.copy()
    overlay[full_mask > 127] = (255, 0, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay_bgr)

    print(f"Done. Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
