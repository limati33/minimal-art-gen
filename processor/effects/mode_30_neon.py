# processor/effects/mode_30_neon.py
import cv2
import numpy as np
import os

def apply_neon(img, w=None, h=None, out_dir=None, base_name=None):
    """
    Mode 30: Динамический неоновый градиент.
    Толщина и яркость линий зависят от силы контура.
    Внешние контуры — жирные и яркие, внутренние детали — тонкие.
    """
    if img is None: return None
    
    # 1. Ресайз
    img_h, img_w = img.shape[:2]
    if w and h and (img_w != w or img_h != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    # 2. Подготовка цвета
    color_source = cv2.bilateralFilter(img, 9, 75, 75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Вычисляем "Силу" границ (Градиент Собеля)
    grad_x = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Нормализуем силу границ (0.0 - 1.0)
    mag_norm = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    # 4. Получаем базовые линии (Canny)
    edges = cv2.Canny(gray_blur, 50, 150)

    # 5. СОЗДАНИЕ ДИНАМИЧЕСКОЙ МАСКИ
    # Создаем два слоя: тонкий и толстый
    kernel_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    edges_thin = cv2.dilate(edges, kernel_thin, iterations=1)
    edges_thick = cv2.dilate(edges, kernel_thick, iterations=1)
    
    # Смешиваем их на основе силы градиента (mag_norm)
    # Там где градиент сильный — берем толстую линию, где слабый — тонкую
    dynamic_mask = (edges_thick.astype(np.float32) * mag_norm + 
                    edges_thin.astype(np.float32) * (1.0 - mag_norm))
    
    # Сглаживание для Anti-aliasing
    dynamic_mask = cv2.GaussianBlur(dynamic_mask, (3, 3), 0)
    
    # 6. Наложение цвета
    alpha = cv2.merge([dynamic_mask, dynamic_mask, dynamic_mask]) / 255.0
    # Усиливаем яркость в зависимости от силы градиента
    brightness_map = cv2.merge([mag_norm, mag_norm, mag_norm]) * 1.5 + 0.5
    
    colored_core = (color_source.astype(np.float32) * alpha * brightness_map).astype(np.uint8)

    # 7. Динамическое свечение (Glow)
    # Сильные линии светятся шире
    glow_map = cv2.GaussianBlur(colored_core, (15, 15), 0)
    glow_wide = cv2.GaussianBlur(colored_core, (45, 45), 0)
    
    # Итоговый композит: Ядро + два слоя свечения
    final_result = cv2.addWeighted(colored_core, 1.5, glow_map, 0.7, 0)
    final_result = cv2.addWeighted(final_result, 1.0, glow_wide, 0.4, 0)

    # 8. Финальная очистка фона
    final_result = np.clip(final_result, 0, 255).astype(np.uint8)
    _, thresh = cv2.threshold(cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY), 5, 255, cv2.THRESH_BINARY)
    final_result = cv2.bitwise_and(final_result, final_result, mask=thresh)

    # 9. Сохранение
    # if out_dir and base_name:
    #     out_path = os.path.join(out_dir, f"{base_name}_mode30_dynamic.png")
    #     cv2.imwrite(final_result)

    return final_result