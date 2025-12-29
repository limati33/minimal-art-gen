# processor/effects/mode_29_filter.py
import os
import cv2
import numpy as np

def apply_filter(img, w=None, h=None, out_dir=None, base_name=None,
                     n_colors=5,          # Сколько главных цветов искать (макс)
                     min_saturation=40,   # Порог серого (0..255), всё что ниже - фон
                     min_value=30,        # Порог темного (0..255)
                     color_threshold=25,  # Радиус похожесть цвета (в LAB). Больше = захватит больше оттенков
                     min_area_ratio=0.015, # Мин. 1.5% площади, чтобы сохранить файл
                     blur_mask=True):     # Мягкие края

    if img is None: return None
    
    # 1. Ресайз и подготовка
    img_h, img_w = img.shape[:2]
    if w and h and (img_w != w or img_h != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Работаем в BGR для сохранения, но анализируем в HSV (для отсева серого) и LAB (для поиска цветов)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # ЧБ подложка
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 2. Маска валидных пикселей (насыщенные и не слишком темные)
    # H: 0-180, S: 0-255, V: 0-255
    # Отсекаем всё серое и черное, чтобы K-Means не искал там цвета
    valid_mask = cv2.inRange(hsv, np.array([0, min_saturation, min_value]), 
                                  np.array([180, 255, 255]))
    
    valid_pixels_count = cv2.countNonZero(valid_mask)
    if valid_pixels_count < (img_w * img_h * 0.01):
        print(" [AutoSplash] Image is mostly grayscale. Skipping.")
        return gray_3ch

    # 3. Кластеризация (K-Means)
    # Берем только a и b каналы из LAB (цветовая информация без яркости)
    # Это позволяет объединить "темно-красный" и "светло-красный" в один цвет
    ab_channels = lab[:, :, 1:3] # shape (H, W, 2)
    
    # Выбираем только валидные пиксели для обучения
    masked_ab = ab_channels[valid_mask > 0] # shape (N, 2)
    
    # Если пикселей слишком много, берем случайную выборку для скорости (макс 100k точек)
    if masked_ab.shape[0] > 100000:
        indices = np.random.choice(masked_ab.shape[0], 100000, replace=False)
        training_data = masked_ab[indices]
    else:
        training_data = masked_ab

    training_data = np.float32(training_data)

    # Запускаем K-Means
    # n_colors - это K. Мы просим найти K центров кучек.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # flags=cv2.KMEANS_PP_CENTERS помогает выбрать хорошие начальные центры
    compactness, labels, centers = cv2.kmeans(training_data, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # centers - это координаты (a, b) наших найденных цветов
    
    results = []

    # 4. Генерация картинок для каждого найденного центра
    img_float = img.astype(np.float32)
    gray_float = gray_3ch.astype(np.float32)
    
    # Преобразуем все изображение в float для вычисления расстояний
    ab_float = ab_channels.astype(np.float32)

    for i, center in enumerate(centers):
        # center = [a, b]
        
        # Считаем Евклидово расстояние каждого пикселя картинки до этого центра цвета
        # dist = sqrt( (a - ca)^2 + (b - cb)^2 )
        diff = ab_float - center
        dist_sq = diff[:,:,0]**2 + diff[:,:,1]**2
        dist = np.sqrt(dist_sq)
        
        # Создаем маску: 
        # Если расстояние < color_threshold -> пиксель берем.
        # Добавляем плавность через sigmoid или linear ramp
        
        # Жесткий порог + мягкий переход
        # Пиксели ближе, чем threshold, будут белыми. 
        # Пиксели дальше — быстро уходят в черное.
        
        # Нормализуем маску: 1.0 в центре цвета, 0.0 на границе порога
        # (threshold - dist) / transition
        mask_float = (color_threshold - dist) / 10.0 # 10.0 - ширина перехода (мягкость)
        mask_float = np.clip(mask_float, 0.0, 1.0)
        
        # Принудительно обнуляем там, где исходно было серо (valid_mask)
        mask_float[valid_mask == 0] = 0.0
        
        # Проверяем площадь цвета
        mask_sum = np.sum(mask_float)
        total_pixels = img_w * img_h
        ratio = mask_sum / total_pixels
        
        if ratio < min_area_ratio:
            continue # Цвет слишком редкий (шум или артефакт)

        # Дополнительное размытие маски (антиалиасинг)
        if blur_mask:
            mask_float = cv2.GaussianBlur(mask_float, (3, 3), 0)

        # Сборка
        alpha = mask_float[:, :, None]
        result = img_float * alpha + gray_float * (1.0 - alpha)
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Определяем "имя" цвета для файла (конвертируем центр LAB обратно в RGB для превью)
        # Создаем 1x1 пиксель с этим цветом LAB и L=128 (средняя яркость)
        preview_lab = np.uint8([[[128, center[0], center[1]]]])
        preview_rgb = cv2.cvtColor(preview_lab, cv2.COLOR_LAB2RGB)[0,0]
        hex_color = "{:02x}{:02x}{:02x}".format(preview_rgb[0], preview_rgb[1], preview_rgb[2])
        
        out_filename = f"{base_name}_color_{i}_{hex_color}.png"
        
        if out_dir:
            out_path = os.path.join(out_dir, out_filename)
            cv2.imwrite(out_path, result)
            print(f"  [AutoSplash] Found color #{hex_color} ({ratio*100:.1f}%) -> {out_filename}")
            results.append(result)

    if not results:
        print("  [AutoSplash] No dominant colors found.")
        return gray_3ch # Возвращаем ЧБ

    # Возвращаем первое найденное (или можно склеить коллаж, но пока просто вернем image)
    return results[0]