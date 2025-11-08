# processor/effects/mode_9_plastic.py
import cv2
import numpy as np

def apply_plastic(img, w, h, out_dir, base_name, intensity=0.8):
    # 1. Базовое сглаживание (для гладкой пластиковой поверхности)
    base = cv2.bilateralFilter(img, 11, 120, 120)
    
    # 2. Переход в HSV для лучшей манипуляции бликами и цветом
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 3. Улучшенная маска бликов: CLAHE для локального контраста + blur
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    enhanced_gray = clahe.apply(gray)
    gray_blur = cv2.GaussianBlur(enhanced_gray, (15, 15), 0)
    spec_mask = cv2.normalize(gray_blur.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    spec_mask = cv2.pow(spec_mask, 3.5)  # Концентрация на светлых, но чуть шире
    spec_mask = spec_mask * intensity  # Масштаб по intensity
    
    # 4. Генерация бликов: усиление в V-канале HSV + адаптивное расширение highlights
    # (Адаптировано из shadow/highlight correction для добавления блеска)
    highlight_gain = 1 + intensity * 2.5  # Усиление для глянца
    LUT_highlight = np.power(np.arange(256) / 255.0, 1 / highlight_gain) * 255
    LUT_highlight = np.clip(LUT_highlight, 0, 255).astype(np.uint8)
    v_enhanced = cv2.LUT(v, LUT_highlight)  # Расширение динамики в ярких зонах
    spec = cv2.GaussianBlur(v_enhanced, (0, 0), 8)
    spec = cv2.addWeighted(spec, 1.2, v, -0.6, 0)  # Контрастные блики
    spec = (spec.astype(np.float32) * spec_mask * 1.2).astype(np.uint8)
    
    # 5. Добавление лёгкой текстуры пластика (noise для grain/царапин)
    noise = np.zeros_like(base, dtype=np.float32)
    cv2.randn(noise, 0, 10)  # Gaussian noise
    noise = cv2.GaussianBlur(noise, (3, 3), 0)  # Смягчение для Perlin-like
    base_with_texture = np.clip(base.astype(np.float32) + noise * (intensity * 0.3), 0, 255).astype(np.uint8)
    
    # 6. Лёгкий emboss для "выпуклости" пластика (симуляция рельефа)
    kernel = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=np.float32)  # Bottom-right emboss
    embossed = cv2.filter2D(base_with_texture, -1, kernel)
    embossed = cv2.add(embossed, np.full_like(embossed, 128))  # Neutral gray offset
    base_with_emboss = cv2.addWeighted(base_with_texture, 0.8, embossed, 0.2 * intensity, 0)
    
    # 7. Комбинация: добавляем блики и усиливаем насыщенность в глянцевых зонах
    plastic_v = cv2.add(v_enhanced, spec)
    s_enhanced = cv2.addWeighted(s, 1.0, s, 0.2 * intensity, 0)  # Лёгкое усиление цвета для пластика
    hsv_plastic = cv2.merge([h, s_enhanced, plastic_v])
    plastic = cv2.cvtColor(hsv_plastic, cv2.COLOR_HSV2BGR)
    
    # 8. Финальный mix с emboss-текстурой + detail enhance для глянца
    plastic = cv2.addWeighted(plastic, 0.7, base_with_emboss, 0.3, 0)
    gloss = cv2.detailEnhance(plastic, sigma_s=10, sigma_r=0.15 * intensity)
    
    return gloss