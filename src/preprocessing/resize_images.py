#!/usr/bin/env python3
"""
Image Folder Resizer (Upscale/Downscale)
A high-performance utility to batch resize images in a folder recursively,
maintaining the subfolder structure.
"""

import os
import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from PIL import Image, ImageOps
except ImportError:
    print("\033[91mError: The 'Pillow' library is required to run this script.\033[0m")
    print("Please install it using: pip install Pillow")
    sys.exit(1)

# Try importing tqdm for a nice progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Mapping from string to Pillow resampling filters
if hasattr(Image, "Resampling"):
    RESAMPLE_MAP = {
        "nearest": Image.Resampling.NEAREST,
        "box": Image.Resampling.BOX,
        "bilinear": Image.Resampling.BILINEAR,
        "hamming": Image.Resampling.HAMMING,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
else:
    # Older Pillow compatibility
    RESAMPLE_MAP = {
        "nearest": Image.NEAREST,
        "box": Image.BOX,
        "bilinear": Image.BILINEAR,
        "hamming": Image.HAMMING,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }

# Supported image extensions (case-insensitive)
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}

def calculate_new_dimensions(orig_w, orig_h, scale=None, target_w=None, target_h=None):
    """Calculates new dimensions maintaining aspect ratio if only one dimension is given."""
    if scale is not None:
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        return new_w, new_h

    if target_w is not None and target_h is not None:
        return target_w, target_h

    if target_w is not None:
        ratio = target_w / orig_w
        new_h = max(1, int(orig_h * ratio))
        return target_w, new_h

    if target_h is not None:
        ratio = target_h / orig_h
        new_w = max(1, int(orig_w * ratio))
        return new_w, target_h

    return orig_w, orig_h

def resize_single_image(input_path: Path, output_path: Path, scale=None, target_w=None, target_h=None, 
                        resample_filter=None, quality=95, keep_exif=True, in_place=False):
    """Resizes a single image and saves it to the output path."""
    if resample_filter is None:
        resample_filter = RESAMPLE_MAP["lanczos"]

    try:
        with Image.open(input_path) as img:
            orig_w, orig_h = img.size
            new_w, new_h = calculate_new_dimensions(orig_w, orig_h, scale, target_w, target_h)
            
            # Perform resize
            resized_img = img.resize((new_w, new_h), resample=resample_filter)
            
            # Apply EXIF transpose to maintain correct physical orientation (e.g. phone camera tilt)
            if keep_exif:
                resized_img = ImageOps.exif_transpose(resized_img)
            
            # Setup save parameters
            save_args = {}
            
            # Format detection
            save_format = img.format if img.format else None
            if not save_format:
                ext = input_path.suffix.lower()
                if ext in ['.jpg', '.jpeg']:
                    save_format = 'JPEG'
                elif ext == '.png':
                    save_format = 'PNG'
                elif ext == '.webp':
                    save_format = 'WEBP'
                elif ext == '.bmp':
                    save_format = 'BMP'
                elif ext in ['.tiff', '.tif']:
                    save_format = 'TIFF'

            # Format-specific adjustments
            if save_format in ('JPEG', 'MPO'):
                save_args['quality'] = quality
                # Convert RGBA to RGB for JPEG format
                if resized_img.mode in ('RGBA', 'LA'):
                    resized_img = resized_img.convert('RGB')
            elif save_format == 'WEBP':
                save_args['quality'] = quality
                
            # Exif preservation
            if keep_exif:
                exif_data = img.info.get('exif')
                if exif_data:
                    save_args['exif'] = exif_data

            # Make sure parent directory of output_path exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if in_place:
                # Atomic save to prevent corruption on crash
                tmp_path = output_path.with_suffix(output_path.suffix + '.tmp')
                resized_img.save(tmp_path, format=save_format, **save_args)
                if os.path.exists(output_path):
                    os.remove(output_path)
                tmp_path.rename(output_path)
            else:
                resized_img.save(output_path, format=save_format, **save_args)
                
        return True, (orig_w, orig_h, new_w, new_h)
    except Exception as e:
        return False, str(e)

def find_images(input_dir: Path):
    """Recursively walks input directory to find supported images."""
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            path = Path(root) / file
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(path)
    return image_paths

def main():
    parser = argparse.ArgumentParser(
        description="Recursively upscale or downscale images in a directory and its subfolders.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    group_input = parser.add_argument_group("Input/Output Options")
    group_input.add_argument("-i", "--input", required=True, type=str, help="Path to input directory containing images.")
    group_input.add_argument("-o", "--output", type=str, help="Path to output directory.\nIf omitted, appends '_resized' to the input directory name.\nIgnored if --in-place is specified.")
    group_input.add_argument("--in-place", action="store_true", help="Overwrite the original images in-place. (Caution: Make backups first!)")
    
    group_scale = parser.add_argument_group("Scaling Options")
    scale_type = group_scale.add_mutually_exclusive_group(required=True)
    scale_type.add_argument("-s", "--scale", type=float, help="Scaling factor (e.g. 0.5 for 50%% downscaling, 2.0 for 200%% upscaling).")
    scale_type.add_argument("-w", "--width", type=int, help="Target width in pixels. Aspect ratio will be preserved unless --height is also set.")
    group_scale.add_argument("-he", "--height", type=int, help="Target height in pixels. Aspect ratio will be preserved unless --width is also set.")
    
    group_advanced = parser.add_argument_group("Advanced Options")
    group_advanced.add_argument(
        "--interpolation", 
        choices=list(RESAMPLE_MAP.keys()), 
        default="lanczos", 
        help="Interpolation resampling filter (default: lanczos)."
    )
    group_advanced.add_argument("-j", "--threads", type=int, default=os.cpu_count(), help="Number of concurrent threads (default: number of CPU cores).")
    group_advanced.add_argument("-q", "--quality", type=int, default=95, help="Output image quality for JPEG and WebP formats (1-100, default: 95).")
    group_advanced.add_argument("--no-exif", action="store_true", help="Discard metadata/EXIF tags from output images.")
    group_advanced.add_argument("-v", "--verbose", action="store_true", help="Print details for every processed file.")

    args = parser.parse_args()

    # Verify input path
    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        print(f"\033[91mError: Input directory '{input_dir}' does not exist.\033[0m")
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"\033[91mError: Input path '{input_dir}' is not a directory.\033[0m")
        sys.exit(1)

    # Determine output path
    if args.in_place:
        output_dir = input_dir
        print("\033[93m WARNING: Operating in-place. Original images will be modified directly. \033[0m")
        confirm = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Operation cancelled by user.")
            sys.exit(0)
    else:
        if args.output:
            output_dir = Path(args.output).resolve()
        else:
            output_dir = Path(str(input_dir) + "_resized").resolve()

    resample_filter = RESAMPLE_MAP[args.interpolation]
    keep_exif = not args.no_exif

    # Find images
    print(f"\nScanning for images in '{input_dir}'...")
    image_paths = find_images(input_dir)
    total_images = len(image_paths)
    
    if total_images == 0:
        print("No supported images found in the specified directory.")
        sys.exit(0)
        
    print(f"Found {total_images} images.")
    if not args.in_place:
        print(f"Resized images will be saved to: '{output_dir}'")
    print(f"Using {args.threads} worker threads.")

    # Start timer
    start_time = time.time()
    
    success_count = 0
    fail_count = 0
    failures = []

    # Prepare job list
    jobs = []
    for img_path in image_paths:
        if args.in_place:
            out_path = img_path
        else:
            # Map input path to relative path under output directory
            rel_path = img_path.relative_to(input_dir)
            out_path = output_dir / rel_path
        
        jobs.append((img_path, out_path))

    # Process images using ThreadPoolExecutor
    print("\nProcessing images...")
    
    # Custom progress bar function if tqdm is not installed
    def print_progress(completed, total):
        percent = (completed / total) * 100
        bar_len = 40
        filled_len = int(bar_len * completed // total)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write(f"\r[{bar}] {percent:.1f}% ({completed}/{total})")
        sys.stdout.flush()

    completed_jobs = 0
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit all tasks
        future_to_paths = {}
        for in_p, out_p in jobs:
            future = executor.submit(
                resize_single_image, 
                input_path=in_p, 
                output_path=out_p, 
                scale=args.scale, 
                target_w=args.width, 
                target_h=args.height, 
                resample_filter=resample_filter, 
                quality=args.quality, 
                keep_exif=keep_exif, 
                in_place=args.in_place
            )
            future_to_paths[future] = (in_p, out_p)

        # Monitor progress
        if HAS_TQDM:
            for future in tqdm(as_completed(future_to_paths), total=total_images, desc="Resizing", unit="img"):
                in_p, out_p = future_to_paths[future]
                success, result = future.result()
                if success:
                    success_count += 1
                    if args.verbose:
                        orig_w, orig_h, new_w, new_h = result
                        print(f"✓ Resized '{in_p.name}': {orig_w}x{orig_h} -> {new_w}x{new_h}")
                else:
                    fail_count += 1
                    failures.append((in_p, result))
                    if args.verbose:
                        print(f"✗ Failed '{in_p.name}': {result}")
        else:
            for future in as_completed(future_to_paths):
                in_p, out_p = future_to_paths[future]
                success, result = future.result()
                completed_jobs += 1
                print_progress(completed_jobs, total_images)
                if success:
                    success_count += 1
                    if args.verbose:
                        orig_w, orig_h, new_w, new_h = result
                        print(f"\n✓ Resized '{in_p.name}': {orig_w}x{orig_h} -> {new_w}x{new_h}")
                else:
                    fail_count += 1
                    failures.append((in_p, result))
                    if args.verbose:
                        print(f"\n✗ Failed '{in_p.name}': {result}")
            print() # Print a final newline after the manual progress bar

    duration = time.time() - start_time
    print("\n" + "=" * 50)
    print("RESIZE PROCESS COMPLETE")
    print("=" * 50)
    print(f"Total time taken : {duration:.2f} seconds")
    print(f"Successfully resized: {success_count} / {total_images}")
    
    if fail_count > 0:
        print(f"Failed to resize  : {fail_count}")
        print("\nFailed Files:")
        for idx, (path, err) in enumerate(failures[:20], 1):
            print(f"{idx}. {path} - Error: {err}")
        if len(failures) > 20:
            print(f"... and {len(failures) - 20} more errors.")
    else:
        print("All images processed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
