# Microscopic Crack Detection — Inference README

This repository contains code for tiled inference of crack segmentation models (YOLO/seg and custom PyTorch pipelines) and utilities to reduce false positives caused by marker/annotation ink drawn around cracks.

Quick overview

- `infer_yolo_tiled.py`: tile-based YOLO (seg) inference with marker suppression and cosine blending.
- `Makefile`: convenience commands to install dependencies and run inference with presets.
- `requirements.txt`: Python dependencies used by the project.

Prereqs

- Python 3.8+ (3.10+ recommended)
- A CUDA-capable GPU if using GPU builds of `torch` and `ultralytics` (optional)

Install (recommended virtualenv)

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\\Scripts\\activate     # Windows PowerShell
pip install -r requirements.txt
# For best performance with GPU install torch using instructions at https://pytorch.org
```

Running inference

Generic command:

```bash
python infer_yolo_tiled.py --image path/to/image.png --output-dir output \\
  --model-path checkpoints/yolo26n-seg-train_2_weights/best.pt \\
  --tile-size 640 --overlap 96 --conf 0.18 --iou 0.45 \\
  --mask-prob-threshold 0.42 --min-component-area 8 --save-debug-mask
```

CLI presets (Makefile targets)

- Balanced (default tuning):

```bash
make balanced IMAGE=path/to/image.png
```

- Aggressive marker removal (remove more colored ink):

```bash
make aggressive IMAGE=path/to/image.png
```

- Tiny-crack-sensitive (preserve faint cracks):

```bash
make sensitive IMAGE=path/to/image.png
```

Quick flags you can tune

- `--mask-prob-threshold`: merged probability threshold (lower to keep faint cracks)
- `--min-component-area`: discard components smaller than this (in pixels)
- `--marker-saturation-threshold` / `--marker-value-threshold`: tune how aggressive color-marker suppression is
- `--disable-color-marker-suppression`: skip color-based suppression (use shape filters only)
- `--save-debug-mask`: writes a debug image showing per-tile cleaned union mask

Debugging tips

- Run with `--save-debug-mask` and inspect `output/<image>_debug_clean_union.png` to see what the marker filter removed.
- If markers are black/gray rather than colored, use `--disable-color-marker-suppression` and tune `--min-component-area` and `--roundness-threshold`.

If you'd like, I can tune these defaults directly in `infer_yolo_tiled.py` for your three example images.

License & notes

- No license file is included by default. This README is purely documentation to help running inference locally.

# Microscopic Crack Detection on Large-Scale Aerial Images

An end-to-end deep learning semantic segmentation pipeline designed to identify very thin surface cracks (1–3 pixels wide, ~2 mm real-world width) in ultra-high-resolution aerial and drone photography (100MP+).

---

## Key Features

- **High-Resolution Tiling & Reconstruction**: Efficiently processes massive images by slicing them into overlapping tiles during training and inference, blending tile boundaries using a **2D cosine window** to prevent boundary seam artifacts.
- **Micro-Crack Loss Formulation**: Combines Binary Cross Entropy and Dice Loss (**ComboLoss**) to punish boundary mismatches and ensure high centerline connectivity.
- **Advanced Post-Processing**: Includes skimage centerline skeletonization, connected-components noise filtering, and physical dimension estimation (average crack width and length via distance transforms).
- **Centerline-Buffered Metrics**: Evaluates performance using a distance-buffered metric (tolerance = 3 pixels) which is much more representative for thin structures than standard pixel-level IoU.
- **Autonomous Agent Fleet Integration**: Orchestrated by a team of specialized subagents cooperating across data pipeline, training, inference, evaluation, and reporting layers.

---

## System Architecture

The pipeline consists of modular subsystems integrated through clean contracts:

```text
                    +----------------+
                    | Raw Images     |
                    | RAW/TIFF/JPG   |
                    +-------+--------+
                            │
                            ▼
                 +----------+-----------+
                 | Data Ingestion Layer | (ImageLoader & Metadata extraction)
                 +----------+-----------+
                            │
                            ▼
                 +----------+-----------+
                 | Image Preprocessing  | (CLAHE, Normalization, Denoising)
                 +----------+-----------+
                            │
                            ▼
                 +----------+-----------+
                 | Tile Generator       | (Overlapping 512x512 tile extraction)
                 +----------+-----------+
                            │
             +--------------+--------------+
             │                             │
             ▼                             ▼
+-----------------------+     +-----------------------+
| Annotation Processing |     | Dataset Management    | (Loader with custom augmentations)
+-----------+-----------+     +-----------+-----------+
            │                             │
            +-------------+---------------+
                          │
                          ▼
                +---------+---------+
                | Model Training    | (SMP Unet / ResNet backbones)
                +---------+---------+
                          │
                          ▼
                +---------+---------+
                | Inference Engine  | (Tiled batched inference)
                +---------+---------+
                          │
                          ▼
                +---------+---------+
                | Tile Merger       | (Cosine window blending)
                +---------+---------+
                          │
                          ▼
                +---------+---------+
                | Evaluation Layer  | (Pixel metrics & Centerline buffered F1)
                +---------+---------+
                          │
                          ▼
                +---------+---------+
                | Visualization     | (Heatmaps & TP/FP/FN error overlays)
                +-------------------+
```

---

## Installation & Setup

This project uses `uv` for python versioning and package management.

1. **Clone the repository**:

   ```bash
   git clone <repo-url>
   cd miroscopic-crack-detection
   ```

2. **Sync the environment**:
   ```bash
   uv sync
   ```
   This will automatically install dependencies like PyTorch, torchvision, segmentation-models-pytorch, albumentations, opencv-python-headless, and scikit-image inside the `.venv`.

---

## Quickstart Guide

### 1. Generate Synthetic Data

If you don't have a dataset yet, generate a concrete-textured dataset containing synthetic crack patterns:

```bash
uv run generate_synthetic_data.py
```

This generates raw images and corresponding binary pixel-level masks under `data/raw/`.

### 2. Generate Tiled Dataset

Tile the high-resolution raw images into 512x512 training patches (keeping only 10% of negative patches without cracks to balance the dataset):

```bash
uv run tile_dataset.py
```

Outputs are saved in `data/tiles/`.

### 3. Train the Segmentation Model

Train the ResNet34-UNet baseline model for 15 epochs using the ComboLoss:

```bash
uv run train.py --epochs 15 --batch-size 4 --lr 3e-4
```

The best checkpoint based on validation F1 score is saved to `checkpoints/v1/best_model.pth`.

### 4. Run Inference & Evaluation

Run tiled inference on a full-resolution image, post-process the mask, estimate crack dimensions, and evaluate against ground truth:

```bash
uv run infer.py --image data/raw/test/images/concrete_000.png --mask data/raw/test/masks/concrete_000.png
```

Outputs are written to the `output/` directory:

- `{image}_overlay.png`: Semi-transparent crack prediction mask overlaid in red.
- `{image}_heatmap.png`: Raw confidence map colored from blue (0.0) to red (1.0).
- `{image}_error_analysis.png`: Color-coded performance comparison: **Green (True Positive)**, **Red (False Positive)**, and **Blue (False Negative)**.
- `{image}_results.json`: Detailed dimensions (area, length, average width), mean confidence, centerline coordinates list, and evaluation metrics.
- `summary_results.csv`: Table containing row-by-row image metrics.

---

## Development Team (Agent Fleet)

We coordinate development through a team of 5 specialized subagents. For detailed agent profiles, system prompts, file ownership lists, and workflows, please refer to:
**[FLEET_DESIGN.md](file:///D:/codes/projects/interns/intern-year-four/miroscopic-crack-detection/AGENTS/FLEET_DESIGN.md)**
