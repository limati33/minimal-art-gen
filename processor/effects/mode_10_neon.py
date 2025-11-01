# processor/effects/mode_10_neon.py
import cv2
import numpy as np

def apply_neon(img, w, h, out_dir, base_name):
    dark = (img * 0.25).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    glow = np.zeros_like(img)
    glow[..., 0] = cv2.GaussianBlur(edges, (9,9), 0)
    glow[..., 2] = cv2.GaussianBlur(edges, (15,15), 0)
    glow = cv2.blur(glow, (7,7))
    glow = cv2.convertScaleAbs(glow * 2)
    return cv2.addWeighted(dark, 1.0, glow, 0.9, 0)