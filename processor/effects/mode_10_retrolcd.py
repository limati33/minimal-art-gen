# processor/effects/mode_10_retrolcd
import cv2
import numpy as np

def apply_retrolcd(img, w, h, out_dir, base_name, **kwargs):
    small = cv2.resize(img, (w//2, h//2))
    quant = (small // 64) * 64
    up = cv2.resize(quant, (w, h), interpolation=cv2.INTER_NEAREST)

    # горизонтальные линии
    line = np.tile(np.linspace(0.7, 1.0, h).reshape(h,1), (1,w))
    up_float = up.astype(np.float32) / 255.0
    lcd = (up_float * line[...,None] * 255).astype(np.uint8)
    return lcd
