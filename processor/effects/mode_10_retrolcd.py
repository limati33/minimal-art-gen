# processor/effects/mode_10_retrolcd
import cv2
import numpy as np

def apply_retrolcd(img, w, h, out_dir, base_name, **kwargs):
    # Пиксели крупные
    small = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    up_float = up.astype(np.float32) / 255.0

    # Явно указываем dtype=np.float32, чтобы избежать апкаста в float64
    line = np.linspace(0.9, 1.05, h, dtype=np.float32).reshape(h, 1)
    backlight = np.float32(0.08)
    
    lcd = up_float * line[..., None] + backlight
    lcd = np.clip(lcd, 0, 1).astype(np.float32) # На всякий случай фиксируем тип

    # Game Boy палитра
    PALETTE = np.array([
        [15, 56, 15],
        [48, 98, 48],
        [139, 172, 15],
        [155, 188, 15]
    ], dtype=np.float32) / 255.0

    # Теперь cv2.cvtColor получит float32 и отработает корректно
    gray = cv2.cvtColor(lcd, cv2.COLOR_BGR2GRAY)
    
    # Квантование яркости
    idx = np.floor(gray * (len(PALETTE)-1)).astype(np.int32)
    # Ограничиваем индексы, чтобы не вылететь за пределы массива
    idx = np.clip(idx, 0, len(PALETTE)-1)
    
    lcd_final = PALETTE[idx]

    return (lcd_final * 255).astype(np.uint8)