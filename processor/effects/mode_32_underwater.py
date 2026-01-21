# processor/effects/mode_32_abyss.py
import cv2
import numpy as np
import os

def apply_underwater(img, w=None, h=None, out_dir=None, base_name=None):
    """
    Mode 32: Абиссаль (Глубоководный эффект).
    Мягкие органические искажения, каустика и глубокий синий градиент.
    """
    if img is None: return None
    
    # 1. Ресайз
    img_h, img_w = img.shape[:2]
    if w and h and (img_w != w or img_h != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    # 2. ОРГАНИЧЕСКИЕ ИСКАЖЕНИЯ (вместо ряби)
    # Создаем карту низкочастотного шума
    noise_size = (int(w/20), int(h/20))
    noise = np.random.uniform(-10, 10, noise_size).astype(np.float32)
    distort_map = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x.astype(np.float32) + distort_map
    map_y = map_y.astype(np.float32) + distort_map
    
    # Применяем плавное искажение
    submerged = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 3. ЦВЕТ ГЛУБИНЫ (Color Grading)
    # Накладываем сине-зеленый фильтр
    sea_color = np.full_like(submerged, (120, 100, 20), dtype=np.uint8) # BGR: Темная морская волна
    submerged = cv2.addWeighted(submerged, 0.7, sea_color, 0.3, 0)

    # 4. КАУСТИКА (Световые блики воды)
    # Создаем эффект сетки света
    caustic_noise = np.random.normal(128, 50, (int(h/10), int(w/10))).astype(np.uint8)
    caustic_mask = cv2.resize(caustic_noise, (w, h), interpolation=cv2.INTER_CUBIC)
    _, caustic_mask = cv2.threshold(caustic_mask, 180, 255, cv2.THRESH_BINARY)
    caustic_mask = cv2.GaussianBlur(caustic_mask, (15, 15), 0)
    
    # Добавляем блики (Screen mode)
    caustic_layer = cv2.merge([caustic_mask, caustic_mask, caustic_mask])
    submerged = cv2.addWeighted(submerged, 1.0, caustic_layer, 0.2, 0)

    # 5. ВИНЬЕТКА (Давление глубины)
    kernel_x = cv2.getGaussianKernel(w, w/2)
    kernel_y = cv2.getGaussianKernel(h, h/2)
    vignette = kernel_y * kernel_x.T
    vignette = vignette / vignette.max()
    submerged = (submerged * vignette[:, :, np.newaxis]).astype(np.uint8)

    return submerged