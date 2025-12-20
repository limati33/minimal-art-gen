# processor/effects/mode_23_vicecity.py
import cv2
import numpy as np


def apply_vicecity(img, w=None, h=None, out_dir=None, base_name=None, image_path=None, n_colors=None, blur_strength=None, mode=None):
    
    # ПРЕДУСЛОВИЕ: Для этого эффекта мы ожидаем, что 'img' уже является квантованным кадром (с палитрой, ограниченной n_colors).
    
    # 1. Подготовка
    if w and h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        
    img_f = img.astype(np.float32) / 255.0
    h_img, w_img = img.shape[:2]

    # --- ИЗМЕНЕНИЕ: Вводим жесткий неоновый цветовой сдвиг ---

    # 2. Вычисляем яркость
    gray = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)
    
    # 3. Определяем неоновую палитру (BGR, float 0-1)
    # Используем несколько ключевых цветов (например, 4), чтобы "заставить" изображение быть неоновым.
    # Это работает лучше, чем просто тонирование.
    
    # Пурпур, Циан, Яркий Розовый, Глубокий Фиолетовый
    NEON_COLORS = np.array([
        [0.0, 0.8, 1.0],  # Светло-Циан (Бирюза)
        [1.0, 0.0, 1.0],  # Ярко-Пурпурный/Розовый
        [0.3, 0.0, 0.6],  # Темно-Фиолетовый (Тени)
        [0.8, 0.9, 0.9],  # Почти Белый (Блики)
        [0.1, 0.1, 0.1],  # Черный (Силуэт)
    ], dtype=np.float32)

    # 4. Переназначаем цвета на основе исходной яркости (для имитации n_colors)
    
    # Сортируем оригинальные цвета по яркости (чтобы создать карту переназначения)
    # Здесь мы имитируем квантование, не меняя n_colors, но меняя сами цвета
    
    # Маска яркости
    bins = np.linspace(0, 1, len(NEON_COLORS) + 1)
    color_map_indices = np.digitize(gray, bins[1:])
    color_map_indices = np.clip(
        color_map_indices,
        0,
        len(NEON_COLORS) - 1
    )

    result = NEON_COLORS[color_map_indices]
    
    # 5. Усиление Glow (Свечение)
    
    # Определяем "яркие" области (где будет свечение)
    brightness = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # Только очень яркие цвета будут светиться
    bright_mask = (brightness > 0.7).astype(np.float32)
    
    # Увеличиваем яркость ярких областей (симулируем эмиссию света)
    bright_pixels = result * bright_mask[...,None]
    
    # Создаем сильный, мягкий glow из ярких пикселей
    glow_kernel_size = int(w_img * 0.05)
    glow_kernel_size = glow_kernel_size if glow_kernel_size % 2 != 0 else glow_kernel_size + 1 # Нечетный размер
    
    glow = cv2.GaussianBlur(bright_pixels, (glow_kernel_size, glow_kernel_size), 0)
    
    # Смешиваем glow с основой
    # Основа (result) получает 80% веса, glow получает 20%
    result = cv2.addWeighted(result, 0.8, glow, 0.2, 0)
    
    # 6. Добавляем VHS/аналоговый шум
    noise = (np.random.randn(h_img, w_img, 3) * 0.02 + 1.0).astype(np.float32) # Меньше шума, чтобы не убить неон
    result = np.clip(result * noise, 0, 1)

    return (result * 255).astype(np.uint8)