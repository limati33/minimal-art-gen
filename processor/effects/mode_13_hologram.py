# processor/effects/mode_13_hologram.py
import cv2
import numpy as np
from utils.input_utils import print_progress
from utils.logging_utils import CYAN, RESET, log_error

def apply_hologram(img, w, h, out_dir, base_name, image_path=None, n_colors=None, blur_strength=None, mode=None):
    print_progress(3, prefix=f"{CYAN}Голограмма (3D-анаглиф)...{RESET} ")
    try:
        # 1. Добавляем лёгкий шум в HSV для "голографического" вида
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue_noise = np.random.randn(h, w) * 12  # Меньше шума для ясности 3D
        hsv[..., 0] = (hsv[..., 0] + hue_noise) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(1.05, 1.3, (h, w)), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * np.random.uniform(1.0, 1.2, (h, w)), 0, 255)
        base_frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 2. Вычисляем карту глубины (depth map) из grayscale, сглаживаем
        depth = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        depth = cv2.GaussianBlur(depth, (0, 0), 12)  # Мягче для лучшего 3D-эффекта
        depth = 1.0 - depth  # Инвертируем: тёмное = ближе (0=ближе, 1=дальше) для большего сдвига на переднем плане
        
        # 3. Параметры сдвига: max_offset для параллакса
        max_offset = 12  # Увеличьте для stronger 3D, но не слишком, чтобы избежать артефактов
        
        # 4. Создаём координаты
        rows, cols = np.indices((h, w))
        
        # 5. Вычисляем сдвиг на основе глубины (параллакс: ближе = больший сдвиг)
        shift = (depth * max_offset).astype(np.int32)  # Положительный сдвиг для левого/правого
        
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
        
        # 8. Создаём анаглиф: red от left, green/blue от right
        anaglyph = np.zeros_like(base_frame)
        anaglyph[..., 0] = left_frame[..., 0]  # Красный от левого
        anaglyph[..., 1] = right_frame[..., 1]  # Зелёный от правого
        anaglyph[..., 2] = right_frame[..., 2]  # Синий от правого
        
        # 9. Добавляем голографический glow и noise для атмосферы
        gray = cv2.cvtColor(anaglyph, cv2.COLOR_RGB2GRAY)
        bright = (gray > 170).astype(np.uint8) * 255  # Порог для бликов
        bright = cv2.GaussianBlur(bright, (11, 11), 0)
        glow = np.full_like(anaglyph, (180, 220, 255))  # Светло-голубой glow
        anaglyph = cv2.addWeighted(anaglyph, 1.0, glow, 0.08, 0)
        anaglyph = np.where(bright[..., None], np.clip(anaglyph + 20, 0, 255), anaglyph)
        
        # 10. Лёгкий шум для "голограммы"
        noise = np.random.normal(0, 2.0, anaglyph.shape).astype(np.int16)
        anaglyph = np.clip(anaglyph.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 11. Опционально: добавьте лёгкий радужный градиент для голографического вида
        rainbow_base = cv2.applyColorMap(np.linspace(0, 255, w, dtype=np.uint8), cv2.COLORMAP_HSV)
        rainbow_base = cv2.cvtColor(rainbow_base, cv2.COLOR_BGR2RGB)
        rainbow = np.roll(rainbow_base, int(np.random.uniform(-10, 10)), axis=1)
        rainbow = cv2.resize(rainbow, (w, h), interpolation=cv2.INTER_AREA)
        anaglyph = cv2.addWeighted(anaglyph, 0.85, rainbow, 0.15, 0)
        
        return anaglyph
    except Exception as e:
        log_error("Ошибка при генерации 3D-голограммы", e, image_path, n_colors, blur_strength, mode)
        return img.copy()