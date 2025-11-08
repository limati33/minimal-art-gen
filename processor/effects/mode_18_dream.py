# processor/effects/mode_18_dream.py
import cv2
import numpy as np

def apply_dream(img, w, h, out_dir, base_name):
    # 1. Параметры искажения
    base_offset = int(min(w, h) * 0.015)  # ~1.5% от размера
    freq_y = 25.0  # Частота вертикальных волн
    freq_x = 40.0  # Частота горизонтальных волн
    time_phase = np.random.uniform(0, 2*np.pi)  # Случайная фаза

    # 2. Создаём координатные сетки
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    y_float = y_grid.astype(np.float32)
    x_float = x_grid.astype(np.float32)

    # 3. Плавные синусоидальные смещения
    dy = np.sin(y_float / freq_y + time_phase) * base_offset
    dx = np.cos(x_float / freq_x + time_phase * 0.7) * base_offset * 0.8

    # 4. Добавляем лёгкий случайный шум (как в снах — нестабильность)
    noise_y = (np.random.rand(h, w).astype(np.float32) - 0.5) * base_offset * 0.6
    noise_x = (np.random.rand(h, w).astype(np.float32) - 0.5) * base_offset * 0.6
    dy += noise_y
    dx += noise_x

    # 5. Итоговые карты смещения
    map_y = (y_float + dy).astype(np.float32)
    map_x = (x_float + dx).astype(np.float32)

    # 6. Применяем remap (быстро и без артефактов)
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 7. Мягкое размытие — как в тумане сна
    blurred = cv2.GaussianBlur(warped, (0, 0), sigmaX=2.5)

    # 8. Лёгкий цветовой дрейф в HSV (нестабильность восприятия)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hue_shift = np.sin(y_float / 30 + time_phase) * 8  # ±8 в hue
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.05, 0, 255)  # +5% насыщенности
    dream_hsv = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 9. Финальное мягкое свечение
    glow = cv2.GaussianBlur(dream_hsv, (0, 0), sigmaX=8)
    result = cv2.addWeighted(dream_hsv, 0.85, glow, 0.15, 0)

    return result