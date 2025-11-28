# processor/effects/mode_24_tvbox.py
import cv2
import numpy as np
from utils.input_utils import print_progress
from utils.logging_utils import MAGENTA, RESET

def apply_tvbox(img, w=None, h=None, out_dir=None, base_name=None, image_path=None, n_colors=None, blur_strength=None, mode=None):
    print_progress(3, prefix=f"{MAGENTA}TV Box (Рыбий глаз + Сканлайны)...{RESET} ")
    
    # 1. Подготовка
    # Принимаем, что img уже квантовано и, возможно, смасштабировано
    if img.dtype != np.float32:
        img_f = img.astype(np.float32) / 255.0
    else:
        img_f = img
        
    h_img, w_img = img_f.shape[:2]

    # --- 1. "Рыбий глаз" (Fish-eye) ---
    # Мы используем cv2.remap для применения нелинейного искажения
    
    # Создание карты искажений
    center_x, center_y = w_img / 2, h_img / 2
    max_radius = np.sqrt(center_x**2 + center_y**2)
    # Коэффициент искажения (положительное число для "рыбьего глаза", чем больше, тем сильнее)
    K = 0.5 
    
    map_x = np.zeros((h_img, w_img), dtype=np.float32)
    map_y = np.zeros((h_img, w_img), dtype=np.float32)

    for y in range(h_img):
        for x in range(w_img):
            # Смещение от центра
            dx = x - center_x
            dy = y - center_y
            r = np.sqrt(dx**2 + dy**2)
            
            # Функция искажения: чем дальше от центра, тем сильнее смещение
            # Здесь применяется радиальное искажение линзы (Fish-eye)
            r_distorted = r * (1 + K * (r / max_radius)**2)
            
            # Новые координаты
            if r > 0:
                map_x[y, x] = center_x + dx * (r_distorted / r)
                map_y[y, x] = center_y + dy * (r_distorted / r)
            else:
                map_x[y, x] = center_x
                map_y[y, x] = center_y

    # Применение искажения. Используем INTER_LINEAR для плавности.
    distorted = cv2.remap(img_f, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # --- 2. Горизонтальные полосы развертки (Scanlines) ---
    
    # Создание маски сканлайнов (темные линии через одну)
    scanline_mask = np.zeros((h_img, w_img, 3), dtype=np.float32)
    # Каждая 2-я/3-я строка будет затемнена. 
    scanline_height = 2 # Полосы в 2 пикселя высотой
    darken_factor = 0.8 # Насколько затемнять (0.8 = 20% затемнение)

    for i in range(h_img):
        # Если номер строки делится на scanline_height, делаем ее темной
        if (i % (scanline_height * 2)) < scanline_height:
            scanline_mask[i, :, :] = darken_factor
        else:
            scanline_mask[i, :, :] = 1.0 # Оставляем без изменений

    # Применение маски сканлайнов
    result = distorted * scanline_mask
    
    # --- 3. Виньетка (Vignette) ---
    # Создаем градиентную маску, которая затемняет края
    
    # Расстояние от центра
    Y, X = np.ogrid[:h_img, :w_img]
    dist_sq = (Y - center_y)**2 + (X - center_x)**2
    
    # Максимальное расстояние (от центра до угла)
    max_dist_sq = max_radius**2
    
    # Виньетка: 1.0 в центре, 0.0 на дальних углах
    # Используем экспоненту для более плавного затухания
    vignette_power = 3.0 # Степень затухания
    vignette_factor = np.exp(-vignette_power * dist_sq / max_dist_sq)
    
    # Применение виньетки
    result = result * vignette_factor[..., None]
    
    # Финализация
    result = np.clip(result * 1.05, 0, 1) # Немного повысим общую яркость
    
    return (result * 255).astype(np.uint8)