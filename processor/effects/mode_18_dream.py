# processor/effects/mode_18_dream.py
import cv2
import numpy as np

def apply_dream(img, w, h, out_dir, base_name):
    offset = max(4, int(min(w,h) * 0.01))
    warped = np.zeros_like(img)
    for y in range(img.shape[0]):
        dx = int(np.sin(y / 20.0) * offset)
        for x in range(img.shape[1]):
            nx = np.clip(x + dx + np.random.randint(-offset, offset+1), 0, img.shape[1]-1)
            ny = np.clip(y + np.random.randint(-offset, offset+1), 0, img.shape[0]-1)
            warped[y, x] = img[ny, nx]
    return cv2.GaussianBlur(warped, (3,3), 0)