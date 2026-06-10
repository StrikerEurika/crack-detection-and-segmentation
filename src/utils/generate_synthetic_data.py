import os
import cv2
import numpy as np
import random
from pathlib import Path

def generate_crack_path(width, height, min_length=200):
    """Generates a random walk path representing a crack."""
    # Start at a random border or random inside point
    x = random.randint(int(width * 0.1), int(width * 0.9))
    y = random.randint(int(height * 0.1), int(height * 0.9))
    
    path = [(x, y)]
    
    # Random walk parameters
    angle = random.uniform(0, 2 * np.pi)
    length = 0
    target_length = random.randint(min_length, min_length * 3)
    
    while length < target_length:
        # Change angle slightly
        angle += random.uniform(-0.3, 0.3)
        # Move step
        step_len = random.randint(5, 15)
        dx = int(step_len * np.cos(angle))
        dy = int(step_len * np.sin(angle))
        
        new_x = max(0, min(width - 1, x + dx))
        new_y = max(0, min(height - 1, y + dy))
        
        if new_x == x and new_y == y:
            break  # Stuck at border
            
        path.append((new_x, new_y))
        x, y = new_x, new_y
        length += step_len
        
    return path

def draw_synthetic_crack_image(width, height, num_cracks=3):
    """Creates a synthetic image resembling concrete with thin cracks, and its mask."""
    # 1. Create concrete-like background
    # Start with a base gray color
    base_color = random.randint(120, 180)
    image = np.full((height, width, 3), base_color, dtype=np.uint8)
    
    # Add high-frequency noise (sand texture)
    noise = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add low-frequency variations (shading/stains)
    small_h, small_w = max(4, height // 128), max(4, width // 128)
    stains = np.random.normal(0, 15, (small_h, small_w, 3))
    stains = cv2.resize(stains, (width, height), interpolation=cv2.INTER_CUBIC)
    image = np.clip(image.astype(np.float32) + stains, 0, 255).astype(np.uint8)
    
    # 2. Prepare empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 3. Draw cracks
    for _ in range(num_cracks):
        path = generate_crack_path(width, height)
        # Decide crack width: 1, 2, or 3 pixels
        crack_width = random.choice([1, 2, 3])
        
        # Color of the crack (typically darker gray/black)
        crack_color = max(0, int(base_color - random.randint(40, 80)))
        
        # Draw on image and mask
        for i in range(len(path) - 1):
            pt1 = path[i]
            pt2 = path[i+1]
            # Draw on image
            cv2.line(image, pt1, pt2, (crack_color, crack_color, crack_color), crack_width, cv2.LINE_AA)
            # Draw on mask
            cv2.line(mask, pt1, pt2, 255, crack_width, cv2.LINE_4) # Use line_4/8 to avoid anti-aliasing on binary mask
            
    # Apply small blur to the image to simulate lens blur/softness
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    return image, mask

def generate_dataset(output_dir, num_train=5, num_val=1, num_test=1, size=(2048, 2048)):
    """Generates train/val/test splits of synthetic crack data."""
    output_dir = Path(output_dir)
    print(f"Generating synthetic dataset at: {output_dir}")
    
    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test
    }
    
    for split, count in splits.items():
        split_img_dir = output_dir / split / "images"
        split_mask_dir = output_dir / split / "masks"
        
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)
        
        for idx in range(count):
            print(f"Generating {split} image {idx+1}/{count}...")
            img, mask = draw_synthetic_crack_image(size[0], size[1], num_cracks=random.randint(2, 5))
            
            # Save
            img_path = split_img_dir / f"concrete_{idx:03d}.png"
            mask_path = split_mask_dir / f"concrete_{idx:03d}.png"
            
            cv2.imwrite(str(img_path), img)
            cv2.imwrite(str(mask_path), mask)

if __name__ == "__main__":
    generate_dataset("data/raw", num_train=4, num_val=1, num_test=1, size=(2048, 2048))
    print("Synthetic dataset generation completed successfully!")
