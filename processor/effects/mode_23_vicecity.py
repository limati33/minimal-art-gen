# processor/effects/mode_23_vicecity.py
import cv2
import numpy as np

def apply_vicecity(img, w=None, h=None, out_dir=None, base_name=None):
    # масштаб
    if w and h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img_f = img.astype(np.float32) / 255.0
    h_img, w_img = img.shape[:2]

    # яркость и маски
    gray = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)
    shadow_mask = np.clip(1.0 - gray, 0, 1)
    highlight_mask = np.clip(gray, 0, 1)

    # цветовые наклоны
    shadow_color = np.array([1.0, 0.3, 0.6])  # BGR: тёмные зоны → пурпур/розовый
    highlight_color = np.array([0.6, 0.8, 1.0])  # BGR: света → голубой/сиреневый

    tinted = img_f.copy()
    tinted = tinted * (1 - shadow_mask[...,None]*0.5 - highlight_mask[...,None]*0.5) + \
             shadow_mask[...,None]*shadow_color*0.5 + highlight_mask[...,None]*highlight_color*0.5
    tinted = np.clip(tinted, 0, 1)

    # лёгкий glow: размытие и смешивание
    glow = cv2.GaussianBlur(tinted, (15,15), 0)
    result = cv2.addWeighted(tinted, 0.7, glow, 0.3, 0)

    # лёгкий VHS/неон шум
    noise = (np.random.randn(h_img, w_img, 1) * 0.03 + 1.0).astype(np.float32)
    result = np.clip(result * noise, 0, 1)

    return (result * 255).astype(np.uint8)
