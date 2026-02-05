# processor/effects/mode_34_colorize.py
import cv2
import numpy as np
import os

def _get_sorted_palette(img, k=5):
    """
    Извлекает k доминантных цветов и СОРТИРУЕТ их по яркости.
    Это критически важно, чтобы тени были темными, а свет - светлым.
    """
    # Уменьшаем картинку для скорости K-Means
    small = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)

    _, labels, centers = cv2.kmeans(
        pixels, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        3, cv2.KMEANS_PP_CENTERS
    )
    
    centers = np.uint8(centers)
    
    # Считаем яркость каждого цвета (Luma formula)
    # Y = 0.299R + 0.587G + 0.114B
    luma = np.dot(centers, [0.114, 0.587, 0.299]) # BGR order
    
    # Сортируем центры по яркости
    sorted_indices = np.argsort(luma)
    return centers[sorted_indices]

def _create_lut_from_palette(palette):
    """
    Создает 256-значную таблицу (LUT) для плавного перехода между цветами палитры.
    """
    lut = np.zeros((1, 256, 3), dtype=np.uint8)
    
    # Создаем ключевые точки (anchor points) на шкале 0..255
    indices = np.linspace(0, 255, len(palette)).astype(int)
    
    # Интерполируем (растягиваем палитру на 256 значений)
    # R канал
    lut[0, :, 0] = np.interp(np.arange(256), indices, palette[:, 0])
    # G канал
    lut[0, :, 1] = np.interp(np.arange(256), indices, palette[:, 1])
    # B канал
    lut[0, :, 2] = np.interp(np.arange(256), indices, palette[:, 2])
    
    return lut

def apply_colorize(img, w=None, h=None, out_dir=None, base_name=None):
    """
    Mode 34: Gradient Map (Кинематографическая тонировка).
    Преобразует яркость изображения в цвета заданной палитры.
    """
    if img is None: return None

    # Resize
    ih, iw = img.shape[:2]
    if w and h and (iw != w or ih != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    # 1. Готовим палитры
    
    # АВТО: "Anime Sunset" (Фиолетовый -> Красный -> Желтый -> Белый)
    # Это всегда выглядит эпично для персонажей
    auto_palette = np.array([
        [40, 20, 35],    # Deep Purple (Shadows) - BGR
        [80, 40, 180],   # Magenta (Mids)
        [50, 120, 255],  # Orange (High Mids)
        [180, 230, 255], # Pale Yellow (Lights)
        [255, 255, 255]  # White (Highlights)
    ], dtype=np.uint8)
    
    # ИЗ ФОТО: Извлекаем и сортируем
    extracted_palette = _get_sorted_palette(img, k=5)

    # 2. Создаем LUT (Look-Up Tables)
    lut_auto = _create_lut_from_palette(auto_palette)
    lut_extract = _create_lut_from_palette(extracted_palette)

    # 3. Подготовка ЧБ основы (используем Lab L-канал для лучшего контраста)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Немного повышаем контраст перед маппингом, чтобы цвета были сочнее
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_base = clahe.apply(l_channel)

    # 4. Применяем Gradient Map (магия OpenCV)
    # Это ОЧЕНЬ быстро заменяет каждый серый пиксель на цвет из LUT
    res_auto = cv2.LUT(cv2.cvtColor(gray_base, cv2.COLOR_GRAY2BGR), lut_auto)
    res_extract = cv2.LUT(cv2.cvtColor(gray_base, cv2.COLOR_GRAY2BGR), lut_extract)

    # 5. Смешивание с оригиналом (опционально, для сохранения деталей)
    # Режим "Soft Light" или просто Blend (0.3 оригинала + 0.7 эффекта)
    # Смешиваем по яркости (Luma), сохраняя цвет эффекта
    
    # Для эффекта "Дуотон" лучше оставить чистый LUT, но добавим немного шума для стиля
    noise = np.random.normal(0, 5, res_auto.shape).astype(np.int16)
    res_auto = np.clip(res_auto.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 6. Сохранение
    if out_dir and base_name:
        # Сохраняем вариант с палитрой из фото
        cv2.imwrite(os.path.join(out_dir, f"{base_name}_mode34_palette_extract.png"), res_extract)
        # Сохраняем авто-вариант
        out_path = os.path.join(out_dir, f"{base_name}_mode34_gradient_map.png")
        cv2.imwrite(out_path, res_auto)

    return res_auto