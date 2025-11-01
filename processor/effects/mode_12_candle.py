# processor/effects/mode_12_candle.py
import cv2
import numpy as np

def apply_candle(img, w, h, out_dir, base_name):
    overlay = np.full_like(img, (20, 30, 60))
    blurred = cv2.GaussianBlur(img, (15,15), 30)
    warm = cv2.addWeighted(blurred, 0.85, overlay, 0.25, 0)
    return cv2.addWeighted(img, 0.75, warm, 0.6, 0)