# processor/effects/mode_15_blueprint.py
import cv2
import numpy as np


def apply_blueprint(img, w, h, out_dir, base_name):
    """
    Эффект 'Технический чертёж' (Blueprint):
    - Синий фон (как чертежная бумага)
    - Белые линии (как светящиеся чернила)
    - Чистый, профессиональный вид
    """
    # 1. Градации серого
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2. Усиливаем контраст (для чётких линий)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 3. Выделяем края
    edges = cv2.Canny(gray, 70, 150)

    # 4. Утолщаем линии (как на чертеже)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lines = cv2.dilate(edges, kernel, iterations=1)

    # 5. Лёгкое размытие — убирает "пиксельность", но сохраняет чёткость
    lines = cv2.GaussianBlur(lines, (3, 3), 0)

    # 6. Бинаризация
    _, lines = cv2.threshold(lines, 50, 255, cv2.THRESH_BINARY)

    # 7. Синий фон (глубокий blueprint-синий)
    blue_bg = np.full((h, w, 3), (20, 55, 120), dtype=np.uint8)  # BGR: тёмно-синий

    # 8. Белые линии: маска → белый цвет
    lines_mask = lines  # Уже 0/255
    white_lines = np.zeros_like(blue_bg)
    white_lines[lines_mask > 0] = [255, 255, 255]  # Белый

    # 9. Накладываем белые линии на синий фон
    result = blue_bg.copy()
    result = cv2.addWeighted(result, 1.0, white_lines, 1.0, 0)

    # 10. Опционально: лёгкий шум для "бумажной" текстуры
    noise = np.random.normal(0, 4, result.shape).astype(np.int16)
    result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return result