# processor/effects/mode_28_shadowhatch.py
import os
import cv2
import numpy as np

def apply_shadowhatch(img, w=None, h=None, out_dir=None, base_name=None,
                      # --- Параметры тени ---
                      shadow_gain=3.5,
                      shadow_gamma=0.85,
                      # --- Параметры штриха ---
                      hatch_angle=45,         # Угол кратный 15 (45, 30, 60...)
                      hatch_spacing=8,        # Фиксированное расстояние между центрами линий
                      min_line_width=1.0,     # Толщина линии на свету (почти 0 - исчезает)
                      max_line_width=6.0,     # Толщина линии в тени (почти заливка)
                      cross_angle_offset=90,  # Сдвиг угла для креста (90 = перпендикулярно)
                      cross_threshold=0.4,    # Насколько темной должна быть тень для второго слоя
                      # --- Постобработка ---
                      strength=1.0,           # Прозрачность наложения (1.0 = жестко)
                      return_debug=False,
                      debug_save=False):

    if img is None: return None

    # Приведение формата
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if w and h:
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    h_img, w_img = img.shape[:2]

    # =========================================================
    # 1) КАРТА ТЕНЕЙ (ОБЪЕМ)
    # =========================================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Используем размытую версию для выделения "массы" тени, а не мелких деталей
    # Это предотвращает появление "мусора" в текстурах
    blur_k = max(31, int(min(w_img, h_img) * 0.05) | 1)
    local_mean = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # Вычисляем тень: разница между средним и текущим + гамма-коррекция
    shadow = np.clip((local_mean - gray) * shadow_gain, 0.0, 1.0)
    shadow = np.power(shadow, shadow_gamma)
    
    # Немного сгладим саму карту теней, чтобы толщина линий менялась плавно
    shadow = cv2.GaussianBlur(shadow, (5, 5), 0)

    # =========================================================
    # 2) ГЕОМЕТРИЯ ПРЯМЫХ ЛИНИЙ
    # =========================================================
    # Создаем координатную сетку один раз
    yy, xx = np.mgrid[0:h_img, 0:w_img].astype(np.float32)

    def generate_hatch(angle_deg, spacing, width_map):
        """Генерирует слой штриховки с переменной толщиной"""
        theta = np.deg2rad(angle_deg)
        # Формула проекции точки на вектор направления
        # Поворот координат для получения прямых линий под углом
        projection = xx * np.cos(theta) + yy * np.sin(theta)
        
        # Основной паттерн: пилообразная волна от 0 до spacing
        pattern = np.mod(projection, spacing)
        
        # Определяем, где пиксель является линией.
        # Линия рисуется, если значение паттерна меньше текущей ширины.
        # width_map определяет толщину линии в данной точке.
        # Центрируем линию: abs(pattern - spacing/2) < width/2 дает более аккуратный вид
        half_space = spacing / 2.0
        dist_from_center = np.abs(pattern - half_space)
        
        # width_map делится на 2, т.к. ширина расходится в обе стороны от центра
        return (dist_from_center < (width_map / 2.0)).astype(np.float32)

    # =========================================================
    # 3) РАСЧЕТ ТОЛЩИНЫ (ВМЕСТО ПЛОТНОСТИ)
    # =========================================================
    # Жесткий подход: расстояние между линиями (spacing) всегда константа.
    # Меняется только толщина линии. Это гарантирует ПРЯМЫЕ линии.
    
    # Карта толщины: от min_width до max_width в зависимости от силы тени
    # Если тень 0 -> толщина min_width. Если тень 1 -> толщина max_width.
    line_width_map = min_line_width + (max_line_width - min_line_width) * shadow

    # =========================================================
    # 4) ГЕНЕРАЦИЯ СЛОЕВ
    # =========================================================
    
    # Слой 1: Основной угол
    hatch1 = generate_hatch(hatch_angle, hatch_spacing, line_width_map)

    # Слой 2: Перекрестный штрих (только там, где тень глубокая)
    # Уменьшаем толщину для второго слоя, чтобы не забивать рисунок полностью
    hatch2 = generate_hatch(hatch_angle + cross_angle_offset, hatch_spacing, line_width_map * 0.8)
    
    # Маска для второго слоя (резкий переход)
    cross_mask = (shadow > cross_threshold).astype(np.float32)
    
    # Объединяем: либо слой 1, либо (слой 2 при условии маски)
    # np.maximum дает жесткое объединение
    hatch_final = np.maximum(hatch1, hatch2 * cross_mask)

    # =========================================================
    # 5) НАНЕСЕНИЕ НА ИЗОБРАЖЕНИЕ
    # =========================================================
    # Жесткость: минимальное размытие или вообще без него для "печатного" вида
    # Раскомментируй строку ниже, если нужны супер-четкие пиксельные края:
    # hatch_final = (hatch_final > 0.5).astype(np.float32) 
    
    # Если нужно чуть-чуть сгладить алиасинг (лесенку), но оставить жестким:
    hatch_final = cv2.GaussianBlur(hatch_final, (3, 3), 0) 
    # После размытия делаем порог, чтобы вернуть жесткость краев
    hatch_final = np.clip(hatch_final * 1.5, 0, 1)

    # Цвет чернил (почти черный)
    ink_color = np.array([20, 20, 20], dtype=np.float32)
    
    img_float = img.astype(np.float32)
    hatch_3ch = hatch_final[:, :, None]

    # Линейная интерполяция (LERP) для наложения
    # pixel = pixel * (1 - alpha) + ink * alpha
    # alpha здесь зависит от наличия штриха (hatch_3ch) и общей силы эффекта (strength)
    alpha = hatch_3ch * strength * shadow[:,:,None] # Учитываем shadow, чтобы штрих не лез на яркий свет
    
    # Ограничиваем альфу, чтобы на свету штрихов вообще не было (чистка светов)
    alpha[shadow[:,:,None] < 0.1] = 0.0

    result = img_float * (1.0 - alpha) + ink_color * alpha

    return np.clip(result, 0, 255).astype(np.uint8)