#utils/heatmap_overlay.py
'''Overlay heatmap'''

import numpy as np
import cv2
from PIL import Image
#############################################################
def apply_colormap_to_cam(cam: np.ndarray) -> np.ndarray:
    """Map a [0,1] CAM to RGB using a simple jet-like colormap."""
    cam = np.clip(cam, 0, 1)
   
    c = np.zeros((cam.shape[0], cam.shape[1], 3), dtype=np.uint8)

    #Blue → cyan → yellow → red
    c[..., 0] = np.clip(255 * (cam * 2 - 0.5), 0, 255)      # R
    c[..., 1] = np.clip(255 * (1 - np.abs(cam * 2 - 1)), 0, 255)  # G
    c[..., 2] = np.clip(255 * (1 - cam * 2), 0, 255)        # B
    return c

#############################################################
def colormap_legend(width=220, height=26) -> Image.Image:
    """Small horizontal legend image: blue → red."""
    xs = np.linspace(0, 1, width)
    strip = np.stack([xs for _ in range(height)], axis=0)   # (H, W)
    rgb = apply_colormap_to_cam(strip)
    
    return Image.fromarray(rgb, mode="RGB")

#############################################################
def gradcam_overlay_cv2(base_pil: Image.Image, cam_np, alpha: float = 0.5) -> Image.Image:
    """
    Make a classic Grad-CAM visualization:
    - normalize CAM
    - apply JET colormap
    - blend with original image
    """
    # 1) normalize CAM
    cam = np.maximum(cam_np, 0)
    cam = cam / (cam.max() + 1e-8)
    cam_uint8 = (cam * 255).astype(np.uint8)

    # 2) apply JET colormap
    heat = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)       # BGR
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)                # -> RGB

    # 3) resize both to same size
    w, h = base_pil.size
    heat = cv2.resize(heat, (w, h))
    base = base_pil.convert("RGB").resize((w, h))
    base_np = np.array(base).astype(np.float32)

    # 4) blend: blue-ish stays, red pops
    over = heat.astype(np.float32) * alpha + base_np * (1 - alpha)
    over = np.clip(over, 0, 255).astype(np.uint8)
    return Image.fromarray(over)