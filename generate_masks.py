import os
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

def generate_masks_for_split(dataset_dir: Path, split: str):
    split_dir = dataset_dir / split
    annotation_file = split_dir / "_annotations.coco.json"
    
    if not annotation_file.exists():
        print(f"Annotation file not found for split '{split}' at {annotation_file}. Skipping.")
        return
        
    print(f"Processing split: {split}")
    
    # Create target directories
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = split_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Reorganize: Move any images in the root of split_dir to the images subfolder
    img_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
    moved_count = 0
    for path in split_dir.iterdir():
        if path.is_file() and path.suffix.lower() in img_extensions:
            if path.name == "_annotations.coco.json":
                continue
            target_path = images_dir / path.name
            shutil.move(str(path), str(target_path))
            moved_count += 1
    if moved_count > 0:
        print(f"Moved {moved_count} raw images to {images_dir}")
    
    # Load JSON annotations
    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
        
    # Map image ID to image info
    images_info = {img['id']: img for img in coco_data.get('images', [])}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
        
    generated_count = 0
    
    for img_id, img_info in images_info.items():
        file_name = img_info['file_name']
        height = img_info['height']
        width = img_info['width']
        
        # Create a blank black mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Retrieve annotations for this image
        anns = annotations_by_image.get(img_id, [])
        
        for ann in anns:
            segmentation = ann.get('segmentation', [])
            
            # Draw polygons
            if isinstance(segmentation, list):
                for poly_coords in segmentation:
                    if len(poly_coords) >= 6:  # Need at least 3 points
                        poly = np.array(poly_coords, dtype=np.int32).reshape((-1, 2))
                        cv2.fillPoly(mask, [poly], 255)
                    else:
                        print(f"Warning: Polygon in image {file_name} has too few coordinates: {poly_coords}")
            elif isinstance(segmentation, dict):
                print(f"Warning: RLE segmentation format found for image {file_name}. Skipping RLE annotation.")
        
        # Save mask
        mask_path = masks_dir / file_name
        cv2.imwrite(str(mask_path), mask)
        generated_count += 1
        
    print(f"Finished split '{split}': Generated {generated_count} masks in {masks_dir}\n")

def main():
    dataset_dir = Path("D:/codes/projects/interns/intern-year-four/crack/datasets/actual_images/datasets_v1_annotated_coco-segmentation")
    splits = ["train", "valid", "test"]
    
    for split in splits:
        generate_masks_for_split(dataset_dir, split)

if __name__ == "__main__":
    main()
