import cv2
import numpy as np

def apply_stainedglass(img, w=None, h=None, out_dir=None, base_name=None, **kwargs):
    if img is None:
        return None

    # 1. Подготовка
    if w and h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    
    h_img, w_img = img.shape[:2]
    
    # Слегка размываем оригинал, чтобы усреднение цвета в ячейках было чище
    img_blur = cv2.medianBlur(img, 5)

    # 2. Генерация точек для ячеек (Seeds)
    # Чтобы витраж выглядел красиво, точки не должны быть совсем случайными (будет хаос),
    # но и не должны быть строгой сеткой (будет скучно).
    # Делаем сетку с сильным смещением (jitter).
    
    # Плотность ячеек (чем меньше число, тем больше кусочков стекла)
    # Для 1920x1080 шаг 40-50 дает хороший результат
    grid_step = int(max(20, min(w_img, h_img) / 40)) 
    
    points = []
    for y in range(0, h_img, grid_step):
        for x in range(0, w_img, grid_step):
            # Смещаем точку случайно внутри квадрата сетки
            noise_x = np.random.randint(0, grid_step)
            noise_y = np.random.randint(0, grid_step)
            px = min(w_img - 1, x + noise_x)
            py = min(h_img - 1, y + noise_y)
            points.append((px, py))

    # 3. Построение Диаграммы Вороного (Subdiv2D)
    # OpenCV имеет встроенный инструмент для этого
    rect = (0, 0, w_img, h_img)
    subdiv = cv2.Subdiv2D(rect)
    
    for p in points:
        subdiv.insert(p)
        
    # Получаем список ячеек (фасетов)
    # facets - это список полигонов, centers - их центры
    facets, _ = subdiv.getVoronoiFacetList([])

    # 4. Рисование витража
    # Создаем черный холст (это будут наши границы, если мы не закрасим пиксель)
    stained_glass = np.zeros_like(img)
    
    # Для каждой ячейки:
    for i, facet in enumerate(facets):
        # Преобразуем точки полигона в нужный формат
        facet_poly = np.array(facet, dtype=np.int32)
        
        # Создаем маску для текущего кусочка стекла
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillConvexPoly(mask, facet_poly, 255)
        
        # Вычисляем средний цвет оригинала внутри этого кусочка
        # cv2.mean возвращает (B, G, R, A)
        mean_col = cv2.mean(img_blur, mask=mask)[:3]
        
        # Заливаем кусочек этим цветом
        # (Опционально: можно немного рандомизировать цвет для живости)
        r_boost = np.random.randint(-15, 15)
        color_bgr = (
            np.clip(mean_col[0] + r_boost, 0, 255),
            np.clip(mean_col[1] + r_boost, 0, 255),
            np.clip(mean_col[2] + r_boost, 0, 255)
        )
        
        cv2.fillConvexPoly(stained_glass, facet_poly, color_bgr)
        
        # Рисуем "свинцовую" границу (Lead lines)
        # Толщина 2 пикселя, темно-серый цвет
        cv2.polylines(stained_glass, [facet_poly], True, (20, 20, 20), 2, cv2.LINE_AA)

    # 5. Добавляем текстуру стекла и свечение
    # Стекло не идеально ровное, добавим легкий шум
    noise = np.random.normal(0, 10, (h_img, w_img, 3)).astype(np.int16)
    glassy = cv2.add(stained_glass, noise, dtype=cv2.CV_8U)
    
    # Усиливаем насыщенность (свет через стекло всегда яркий)
    hsv = cv2.cvtColor(glassy, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 255) # Saturation boost
    glassy = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Легкий Bloom (свечение), чтобы стекло казалось светящимся
    glow = cv2.GaussianBlur(glassy, (0, 0), sigmaX=3)
    final_result = cv2.addWeighted(glassy, 0.8, glow, 0.4, 0) # Смешивание для свечения

    return final_result