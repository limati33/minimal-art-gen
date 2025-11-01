# processor/effects/mode_19_pixel.py
import cv2

def apply_pixel(img, w, h, out_dir, base_name):
    scale_factor = max(0.04, min(0.2, 32.0 / max(w, h)))
    small = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)