# processor/effects/mode_31_drift.py
import cv2
import numpy as np
import os
import random

def apply_drift(img, w=None, h=None, out_dir=None, base_name=None):
    if img is None:
        return None

    img_h, img_w = img.shape[:2]
    if w and h and (img_w != w or img_h != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]
    result = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(gray, (0, 0), 5)
    mask = mask / 255.0

    num_waves = int(w * 0.15)

    for _ in range(num_waves):
        x = random.randint(0, w - 1)
        strip_w = random.randint(4, 14)

        y_start = random.randint(0, int(h * 0.6))
        drift_len = random.randint(int(h * 0.15), int(h * 0.5))

        x_end = min(w, x + strip_w)
        strip = img[:, x:x_end].copy()

        for i in range(drift_len):
            y = y_start + i
            if y >= h:
                break

            t = i / drift_len
            ease = 1 - (t * t)  # мягкое затухание

            # волновое смещение
            wave_offset = int(np.sin(t * np.pi * 2) * 3)

            src_y = max(0, y_start - wave_offset)
            alpha = ease * mask[y, x]

            result[y:y+1, x:x_end] = (
                result[y:y+1, x:x_end] * (1 - alpha) +
                strip[src_y:src_y+1, :] * alpha
            ).astype(np.uint8)

    # лёгкий motion blur вниз
    kernel = np.zeros((9, 1))
    kernel[:, 0] = np.linspace(1, 0, 9)
    kernel /= kernel.sum()
    result = cv2.filter2D(result, -1, kernel)

    # немного цветового «подтекания»
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.05, 0, 255)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    return result
