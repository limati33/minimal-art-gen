import cv2
import numpy as np
import os
import random

# Набор символов от "плотных" к "лёгким"
DENSITY_CHARS = "@#M&W8%B$OXA*+=-:. "

def apply_focus(img, w=None, h=None, out_dir=None, base_name=None):
    """
    Mode 35: Печатная машина (Typography Print)
    Полутоновая печать буквами вместо точек.
    """
    if img is None:
        return None

    # --- Resize ---
    ih, iw = img.shape[:2]
    if w and h and (iw != w or ih != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    h_img, w_img = img.shape[:2]

    # --- Бумага ---
    paper_color = (238, 232, 220)  # газетная
    result = np.full((h_img, w_img, 3), paper_color, dtype=np.uint8)

    # --- ЧБ с контрастом ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # --- Параметры "машины" ---
    cell = max(8, int(min(h_img, w_img) * 0.015))  # размер символа
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = cell / 22.0
    thickness = max(1, cell // 14)

    # лёгкий шум яркости (неровная печать)
    noise = np.random.normal(0, 6, gray.shape).astype(np.int16)
    gray_n = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # --- Основной проход ---
    for y in range(0, h_img, cell):
        for x in range(0, w_img, cell):
            block = gray_n[y:y+cell, x:x+cell]
            if block.size == 0:
                continue

            mean_val = np.mean(block)

            # выбор символа по яркости
            idx = int((mean_val / 255) * (len(DENSITY_CHARS) - 1))
            char = DENSITY_CHARS[::-1][idx]

            # случайный микросдвиг (живость)
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)

            # цвет "чернил" (не чисто чёрный)
            ink = int(20 + random.randint(-5, 10))
            ink_color = (ink, ink, ink)

            cv2.putText(
                result,
                char,
                (x + dx, y + cell + dy),
                font,
                font_scale,
                ink_color,
                thickness,
                cv2.LINE_AA
            )

    # --- Лёгкое растекание чернил ---
    result = cv2.GaussianBlur(result, (3,3), 0)

    # --- Бумажная текстура ---
    paper_noise = np.random.normal(0, 4, result.shape).astype(np.int16)
    result = np.clip(result.astype(np.int16) + paper_noise, 0, 255).astype(np.uint8)

    # --- Сохранение ---
    if out_dir and base_name:
        path = os.path.join(out_dir, f"{base_name}_mode35_typography.png")
        cv2.imwrite(path, result)

    return result
