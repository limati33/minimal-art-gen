# processor/effects/mode_2_smooth.py
import cv2

def apply_smooth(img, w, h, out_dir, base_name):
    quantized = cv2.bilateralFilter(img, 9, 75, 75)
    quantized = cv2.convertScaleAbs(quantized, alpha=1.05, beta=6)
    return quantized