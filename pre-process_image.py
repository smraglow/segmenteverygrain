#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


Image.MAX_IMAGE_PIXELS = None
# =========================
# USER INPUT
# =========================
image_path = "/home/elanca/Documents/GitHub/segmenteverygrain/Peptoid_65uL_02162026.jpg"

# Contrast settings
lower_percentile = 1      # clip dark outliers
upper_percentile = 99     # clip bright pinholes

use_clahe = True          # local contrast enhancement
clahe_clip = 2.0
clahe_grid = (8, 8)

# =========================
# LOAD IMAGE
# =========================
img = np.array(Image.open(image_path))

# Convert to grayscale if RGB
if img.ndim == 3:
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
else:
    img_gray = img.copy()

img_gray = img_gray.astype(np.float32)

# =========================
# STEP 1: Robust clipping
# =========================
p_low = np.percentile(img_gray, lower_percentile)
p_high = np.percentile(img_gray, upper_percentile)

img_clipped = np.clip(img_gray, p_low, p_high)

# =========================
# STEP 2: Normalize
# =========================
img_norm = (img_clipped - p_low) / (p_high - p_low)
img_norm = (img_norm * 255).astype(np.uint8)

# =========================
# STEP 3: CLAHE (optional)
# =========================
if use_clahe:
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    img_final = clahe.apply(img_norm)
else:
    img_final = img_norm

# =========================
# DISPLAY RESULTS
# =========================
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(img_gray, cmap='gray')
axs[0].set_title("Original")

axs[1].imshow(img_norm, cmap='gray')
axs[1].set_title("Clipped + Normalized")

axs[2].imshow(img_final, cmap='gray')
axs[2].set_title("Final (CLAHE)")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()

# =========================
# SAVE OUTPUT
# =========================
Image.fromarray(img_final).save("processed_image2.png")

print("Saved processed_image.png")