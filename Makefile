MODEL=checkpoints/yolo26n-seg-train_2_weights/best.pt
OUTDIR=output
TILE_SIZE=640
OVERLAP=96
CONF=0.18
IOU=0.45
MASK_TH=0.42
MIN_AREA=8
MARKER_SAT=70
MARKER_VAL=35

.PHONY: install run balanced aggressive sensitive help

install:
	python -m venv venv
	@echo "Activate the virtualenv then run:"
	@echo "  source venv/bin/activate  # Linux/macOS"
	@echo "  venv\\Scripts\\activate  # Windows PowerShell"
	@echo "Then: pip install -r requirements.txt"

run:
	@if [ -z "$(IMAGE)" ]; then \
		echo "Provide IMAGE=path/to/image.png"; exit 1; \
	fi
	python infer_yolo_tiled.py --image $(IMAGE) --model-path $(MODEL) --output-dir $(OUTDIR) \
		--tile-size $(TILE_SIZE) --overlap $(OVERLAP) --conf $(CONF) --iou $(IOU) \
		--mask-prob-threshold $(MASK_TH) --min-component-area $(MIN_AREA) \
		--marker-saturation-threshold $(MARKER_SAT) --marker-value-threshold $(MARKER_VAL) $(EXTRA)

balanced:
	$(MAKE) run IMAGE=$(IMAGE)

aggressive:
	$(MAKE) run IMAGE=$(IMAGE) MASK_TH=0.44 MIN_AREA=6 MARKER_SAT=60 MARKER_VAL=25 EXTRA="--save-debug-mask"

sensitive:
	$(MAKE) run IMAGE=$(IMAGE) MASK_TH=0.38 MIN_AREA=5 MARKER_SAT=80 MARKER_VAL=40 EXTRA="--save-debug-mask"

help:
	@echo "Usage: make <target> IMAGE=path/to/image.png"
	@echo "Targets: run, balanced, aggressive, sensitive, install"
