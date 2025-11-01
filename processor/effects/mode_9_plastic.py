# processor/effects/mode_9_plastic.py
import cv2
import numpy as np

def apply_plastic(img, w, h, out_dir, base_name):
    plast = cv2.bilateralFilter(img, 7, 90, 90)
    plast = cv2.convertScaleAbs(plast, alpha=1.15, beta=8)
    gray = cv2.cvtColor(plast, cv2.COLOR_RGB2GRAY)
    spec = cv2.threshold(gray, np.percentile(gray, 92), 255, cv2.THRESH_BINARY)[1]
    spec = cv2.GaussianBlur(spec, (31,31), 0)
    spec = (spec / 255.0)[:, :, None]
    highlight = (255 * (0.6 * spec)).astype(np.uint8)
    return np.clip(plast.astype(np.int16) + highlight.astype(np.int16), 0, 255).astype(np.uint8)