# processor/effects/mode_10_retrolcd
import cv2
import numpy as np

def apply_retrolcd(img, w, h, out_dir, base_name, **kwargs):
    # Пиксели крупные
    small = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    up_float = up.astype(np.float32) / 255.0

    # LCD строки
    line = np.linspace(0.9, 1.05, h).reshape(h,1)
    backlight = 0.08
    lcd = up_float * line[...,None] + backlight
    lcd = np.clip(lcd, 0, 1)

    # Game Boy палитра
    PALETTE = np.array([
        [15, 56, 15],
        [48, 98, 48],
        [139, 172, 15],
        [155, 188, 15]
    ], dtype=np.float32) / 255.0

    # Квантование яркости
    gray = cv2.cvtColor(lcd, cv2.COLOR_BGR2GRAY)
    idx = np.clip((gray * (len(PALETTE)-1)).astype(np.int32), 0, len(PALETTE)-1)
    lcd = PALETTE[idx]

    return (lcd * 255).astype(np.uint8)
