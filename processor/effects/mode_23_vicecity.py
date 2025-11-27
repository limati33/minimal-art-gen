# processor/effects/mode_23_vicecity.py
import cv2
import numpy as np
from utils.input_utils import print_progress
from utils.logging_utils import MAGENTA, RESET

def apply_vicecity(img, w=None, h=None, out_dir=None, base_name=None, image_path=None, n_colors=None, blur_strength=None, mode=None):
    print_progress(3, prefix=f"{MAGENTA}Неон / Ретро (Vice City)...{RESET} ")
    
    # 1. Подготовка и масштабирование
    if w and h:
        # Убедимся, что масштабирование использует ближайшего соседа для некоторой блочности 80-х
        # Хотя AREA тоже подойдет, оставим INTER_AREA, как в других частях кода, для совместимости.
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        
    img_f = img.astype(np.float32) / 255.0
    h_img, w_img = img.shape[:2]

    # 2. Вычисление яркости и масок
    gray = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)
    
    # Контрастные маски: тень и свет
    # Используем экспоненциальные маски для более резкого перехода
    shadow_mask = np.power(1.0 - gray, 2.0)  # Усилим эффект тени
    highlight_mask = np.power(gray, 2.0)     # Усилим эффект света

    # 3. Цветовые наклоны (Пурпур/Синий Неон)
    # BGR: Глубокие тени → Пурпур/Фиолетовый
    shadow_color = np.array([0.8, 0.2, 0.9])  
    # BGR: Света → Яркий Голубой/Розовый
    highlight_color = np.array([1.0, 0.6, 0.9]) 

    # 4. Наложение цветовых наклонов
    tinted = img_f.copy()
    
    # Убираем исходные цвета и накладываем градиент
    # Уменьшаем влияние исходного цвета (0.5), накладывая цветовой наклон (0.5)
    tinted = (
        tinted * (1.0 - shadow_mask[...,None] * 0.4 - highlight_mask[...,None] * 0.4) + 
        shadow_mask[...,None] * shadow_color * 0.6 + 
        highlight_mask[...,None] * highlight_color * 0.6
    )
    tinted = np.clip(tinted, 0, 1)

    # 5. Усиление контраста (S-кривая)
    # Немного усилим контраст, чтобы цвета "выпрыгивали"
    tinted = np.power(tinted, 0.9) # Слегка осветляем средние тона
    tinted = np.clip(tinted * 1.1, 0, 1) # Увеличиваем общий контраст

    # 6. Неоновый Glow (Свечение)
    # Увеличенный радиус (25) и сильный вес (0.4) для явного неонового свечения
    glow = cv2.GaussianBlur(tinted, (25, 25), 0)
    result = cv2.addWeighted(tinted, 0.6, glow, 0.4, 0)

    # 7. Шум/Зернистость VHS (Analog Noise)
    # Более заметный шум (0.05) для эффекта грязного VHS
    noise = (np.random.randn(h_img, w_img, 3) * 0.05 + 1.0).astype(np.float32) 
    result = np.clip(result * noise, 0, 1)

    return (result * 255).astype(np.uint8)