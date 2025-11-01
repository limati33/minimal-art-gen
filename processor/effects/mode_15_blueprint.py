# processor/effects/mode_15_blueprint.py
import cv2
import numpy as np

def apply_blueprint(img, w, h, out_dir, base_name):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    lines = cv2.GaussianBlur(edges, (3,3), 0)
    lines = cv2.threshold(lines, 40, 255, cv2.THRESH_BINARY)[1]
    lines_rgb = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)
    blue_bg = np.full_like(img, (10, 60, 140))
    white_lines = cv2.bitwise_and(255 - lines_rgb, np.full_like(lines_rgb, 255))
    white_lines = (white_lines > 0).astype(np.uint8) * 255
    return cv2.addWeighted(blue_bg, 1.0, white_lines, 1.0, 0)