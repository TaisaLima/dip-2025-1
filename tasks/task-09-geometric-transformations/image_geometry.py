# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:

    h, w = img.shape

    offset_x, offset_y = w // 4, h // 4
    translated = np.zeros_like(img)
    translated[offset_y:, offset_x:] = img[:h - offset_y, :w - offset_x]


    rotated = np.rot90(img, k=3) 


    new_w = int(w * 1.5)
    grid_x = (np.arange(new_w) / 1.5).astype(int)
    grid_x = np.clip(grid_x, 0, w - 1)
    stretched = img[:, grid_x]

 
    mirrored = img[:, ::-1]

    yy, xx = np.indices((h, w))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    dx, dy = xx - cx, yy - cy
    radius = np.sqrt(dx**2 + dy**2)
    radius_norm = radius / radius.max()

    k = 0.4  
    factor = 1 + k * (radius_norm**2)

    src_x = np.clip((cx + dx / factor).round().astype(int), 0, w - 1)
    src_y = np.clip((cy + dy / factor).round().astype(int), 0, h - 1)

    distorted = img[src_y, src_x]

    return {
        "translated": translated,
        "rotated": rotated,
        "stretched": stretched,
        "mirrored": mirrored,
        "distorted": distorted
    }
