# processor/effects/mode_20_fire.py
import cv2
import numpy as np

def apply_fire(img, w, h, out_dir, base_name):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bright = cv2.threshold(gray, np.percentile(gray, 70), 255, cv2.THRESH_BINARY)[1]
    bright_blur = cv2.GaussianBlur(bright, (21,21), 0)
    bright_mask = (bright_blur / 255.0)[:, :, None]
    warm = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    glow = (warm.astype(np.float32) * (0.8 * bright_mask)).clip(0,255).astype(np.uint8)
    result = cv2.addWeighted(img, 0.6, glow, 0.9, 0)
    sparks = (np.random.rand(*gray.shape) > 0.995).astype(np.uint8) * 255
    sparks = cv2.GaussianBlur(sparks, (5,5), 0)
    sparks_rgb = cv2.cvtColor(sparks, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(result, 1.0, sparks_rgb, 0.15, 0)