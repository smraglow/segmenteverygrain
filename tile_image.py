#!/usr/bin/env python3

from pathlib import Path
from PIL import Image
import numpy as np

# =========================
# USER INPUTS (EDIT THESE)
# =========================
image_path = r"/home/elanca/Documents/GitHub/segmenteverygrain/processed_image.png"
output_dir = r"/home/elanca/Documents/GitHub/segmenteverygrain/perovskite_testing/"

tile_width = 4096  
tile_height = 4096
overlap = 128  # set to 0 for no overlap

keep_partial_tiles = False  # keep edge tiles that are smaller


# =========================
# SCRIPT
# =========================
Image.MAX_IMAGE_PIXELS = None  # allow large images

image_path = Path(image_path)
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

img = Image.open(image_path)
width, height = img.size

print(f"Loaded image: {image_path}")
print(f"Size: {width} x {height}")

step_x = tile_width - overlap
step_y = tile_height - overlap

tile_count = 0

for row_idx, top in enumerate(range(0, height, step_y)):
    for col_idx, left in enumerate(range(0, width, step_x)):

        right = left + tile_width
        bottom = top + tile_height

        # Handle edges
        if right > width or bottom > height:
            if not keep_partial_tiles:
                continue
            right = min(right, width)
            bottom = min(bottom, height)

        tile = img.crop((left, top, right, bottom))

        # Save tile
        filename = (
            f"{image_path.stem}"
            f"_r{row_idx:03d}_c{col_idx:03d}"
            f"_x{left:05d}-{right:05d}"
            f"_y{top:05d}-{bottom:05d}"
            f"{image_path.suffix}"
        )

        tile.save(output_dir / filename)
        tile_count += 1

print(f"Saved {tile_count} tiles to {output_dir}")