# processor/effects/mode_4_paper.py
import cv2
import numpy as np

def apply_paper(img, w, h, out_dir, base_name):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(mask, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    mask_inv = cv2.bitwise_not(edges)
    out = cv2.bitwise_and(img, img, mask=mask_inv)
    shadow = cv2.GaussianBlur(out, (5,5), 0)
    return cv2.addWeighted(out, 1.0, shadow, 0.08, 0)