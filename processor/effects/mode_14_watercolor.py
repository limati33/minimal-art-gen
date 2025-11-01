# processor/effects/mode_14_watercolor.py
import cv2

def apply_watercolor(img, w, h, out_dir, base_name):
    soft = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.6)
    color_flow = cv2.GaussianBlur(soft, (9,9), 3)
    return cv2.addWeighted(soft, 0.6, color_flow, 0.4, 0)