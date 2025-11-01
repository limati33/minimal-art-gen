# processor/effects/mode_16_map.py
import cv2

def apply_map(img, w, h, out_dir, base_name):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray8 = cv2.convertScaleAbs(gray)
    cmap = cv2.__dict__.get('COLORMAP_TERRAIN', None)
    if cmap is None:
        cmap = cv2.COLORMAP_JET
    return cv2.applyColorMap(gray8, cmap)