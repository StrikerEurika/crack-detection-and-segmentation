import sys
from pathlib import Path

# Add project root to sys.path if not there
sys.path.append(str(Path(__file__).resolve().parent))

# Import main from infer.py and run it
from infer import main

if __name__ == "__main__":
    # Force model_type to 'yolo' if it's not set
    if "--model-type" not in sys.argv:
        sys.argv.extend(["--model-type", "yolo"])
    main()
