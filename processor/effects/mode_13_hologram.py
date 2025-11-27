# processor/effects/mode_13_hologram.py
import cv2
import numpy as np
from utils.input_utils import print_progress
from utils.logging_utils import CYAN, RESET, log_error

def apply_hologram(img, w, h, out_dir, base_name, image_path=None, n_colors=None, blur_strength=None, mode=None):
    print_progress(3, prefix=f"{CYAN}Анаглиф 3D (Красный-Циан)...{RESET} ")
    try:
        # NOTE: Оригинальные шаги 1, 9, 10 и 11, добавляющие шум, свечение и радужный градиент, 
        # удалены, так как они мешают чистому 3D-восприятию. Мы используем исходный img.

        # 1. Используем исходное изображение как базовый кадр
        base_frame = img.copy() # Копируем для чистоты
        
        # 2. Вычисляем карту глубины (depth map) из grayscale, сглаживаем
        depth = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        depth = cv2.GaussianBlur(depth, (0, 0), 12)  # Мягче для лучшего 3D-эффекта
        # Инвертируем: темные объекты (0) кажутся ближе и должны иметь больший сдвиг
        depth = 1.0 - depth 
        
        # 3. Параметры сдвига: max_offset для параллакса
        max_offset = 12  
        
        # 4. Создаём координаты
        rows, cols = np.indices((h, w))
        
        # 5. Вычисляем сдвиг на основе глубины (параллакс: ближе = больший сдвиг)
        shift = (depth * max_offset).astype(np.int32)  
        
        # 6. Создаём левый вид (left view): сдвиг вправо (положительный)
        left_nx = np.clip(cols + shift, 0, w - 1)
        left_frame = np.empty_like(base_frame)
        for c in range(3):
            left_frame[..., c] = np.take_along_axis(base_frame[..., c], left_nx, axis=1)
        
        # 7. Создаём правый вид (right view): сдвиг влево (отрицательный)
        right_nx = np.clip(cols - shift, 0, w - 1)
        right_frame = np.empty_like(base_frame)
        for c in range(3):
            right_frame[..., c] = np.take_along_axis(base_frame[..., c], right_nx, axis=1)
        
        # 8. Создаём чистый 3D-анаглиф (Red-Cyan): 
        # Красный канал от левого глаза (сдвиг вправо)
        # Циан (Зеленый + Синий) от правого глаза (сдвиг влево)
        anaglyph = np.zeros_like(base_frame)
        anaglyph[..., 0] = left_frame[..., 0]   # R (Red) от левого
        anaglyph[..., 1] = right_frame[..., 1]  # G (Green) от правого
        anaglyph[..., 2] = right_frame[..., 2]  # B (Blue) от правого
        
        # В этом месте анаглиф готов для просмотра в 3D-очках
        return anaglyph
        
    except Exception as e:
        log_error("Ошибка при генерации 3D-анаглифа", e, image_path, n_colors, blur_strength, mode)
        return img.copy()