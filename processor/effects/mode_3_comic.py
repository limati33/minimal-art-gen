# processor/effects/mode_3_comic.py
import cv2
import numpy as np

def apply_comic(img, w, h, out_dir, base_name):
    smooth = cv2.bilateralFilter(img, 9, 75, 75)
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    edges_rgb = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(smooth, edges_rgb)