# processor/effects/mode_6_dust.py
import cv2
import numpy as np

def apply_dust(img, w, h, out_dir, base_name):
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(noisy, cv2.COLOR_RGB2GRAY)
    bright = cv2.threshold(gray, np.percentile(gray, 85), 255, cv2.THRESH_BINARY)[1]
    bright = cv2.GaussianBlur(bright, (21,21), 0)
    bright = (bright / 255.0)[:, :, None]
    bloom = (noisy.astype(np.float32) * (0.6 + 0.4 * bright)).clip(0,255).astype(np.uint8)
    return cv2.GaussianBlur(bloom, (3,3), 0)