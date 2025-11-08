# processor/effects/mode_22_graph.py
import cv2
import numpy as np
from pathlib import Path
import warnings
from numpy.polynomial import Polynomial

def apply_graph(img, w=None, h=None, out_dir=None, base_name=None):
    # --- Подготовка и масштабирование ---
    if w and h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    h_img, w_img = img.shape[:2]
    
    # 1. Инверсия яркости для рентген-эффекта
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray  # Инвертируем: тёмное → светлое
    
    # 2. Усиление контраста и деталей (CLAHE для локального)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)
    
    # 3. Multi-scale edge detection (тонкие + толстые линии)
    edges_fine = cv2.Canny(enhanced, 50, 150)  # Тонкие линии
    edges_thick = cv2.Canny(enhanced, 30, 100)  # Более толстые структуры
    edges = cv2.addWeighted(edges_fine, 0.6, edges_thick, 0.4, 0)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)  # Утолщаем слегка
    
    # 4. Монохромная база с рентген-tint (светло-синий/зелёный)
    mono = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    tint = np.full_like(mono, (180, 220, 200))  # Светло-зелёный tint для рентгена
    mono_tinted = cv2.addWeighted(mono, 0.8, tint, 0.2, 0)
    
    # 5. Добавляем линии (чёрные для контуров)
    lines_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    result = cv2.subtract(mono_tinted, lines_bgr // 2)  # Тёмные линии на tinted фоне
    
    # 6. Цветной акцент: выделяем сильные края цветом (красный/оранжевый)
    strong_edges = cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY)[1]
    accent_color = np.full_like(result, (50, 100, 255))  # Оранжевый акцент
    accent_mask = cv2.cvtColor(strong_edges, cv2.COLOR_GRAY2BGR)
    result = np.where(accent_mask > 0, cv2.addWeighted(result, 0.7, accent_color, 0.3, 0), result)
    
    # 7. Полиномиальная аппроксимация для "графического" вида (улучшенная)
    # Параметры
    n_levels = 6  # Меньше уровней для скорости
    min_contour_points = 50
    max_degree = 5  # Меньше степени для стабильности
    equations = []
    curve_count = 0
    levels = np.linspace(50, 220, n_levels).astype(np.int32)
    
    for thresh in levels:
        _, bw = cv2.threshold(enhanced, thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # SIMPLE для скорости
        for cnt in contours:
            if len(cnt) < min_contour_points:
                continue
            pts = cnt.reshape(-1, 2).astype(np.float64)
            xs, ys = pts[:, 0], pts[:, 1]
            
            # Сортируем по x (теперь y = f(x), стандарт для графиков)
            order = np.argsort(xs)
            xs_sorted = xs[order]
            ys_sorted = ys[order]
            
            # Убираем дубликаты по x
            keep = np.diff(xs_sorted) > 0.5
            keep = np.insert(keep, 0, True)
            xs_clean = xs_sorted[keep]
            ys_clean = ys_sorted[keep]
            
            if len(xs_clean) < min_contour_points // 2:
                continue
            
            # Ограничиваем степень
            deg = min(max_degree, max(1, len(xs_clean) // 4))
            
            # Подсэмплинг если слишком много точек
            if len(xs_clean) > 800:
                idxs = np.linspace(0, len(xs_clean) - 1, 800).astype(int)
                xs_fit = xs_clean[idxs]
                ys_fit = ys_clean[idxs]
            else:
                xs_fit, ys_fit = xs_clean, ys_clean
            
            # Фиттинг y = p(x)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = Polynomial.fit(xs_fit, ys_fit, deg)
                p_std = p.convert()
                coeffs_inc = np.asarray(p_std.coef, dtype=np.float64)
                coeffs_desc = coeffs_inc[::-1]
            except Exception:
                continue
            
            # Уравнение y = sum c_i * x^i
            poly = np.poly1d(coeffs_desc)
            deg_used = poly.order
            terms = []
            for i, c in enumerate(coeffs_desc):
                power = deg_used - i
                c_str = f"{c:.4e}"
                if power == 0:
                    terms.append(c_str)
                elif power == 1:
                    terms.append(f"{c_str}*x")
                else:
                    terms.append(f"{c_str}*x^{power}")
            eq = " + ".join(terms)
            eq = f"level={thresh} contour={curve_count} deg={deg_used}: y = {eq}"
            equations.append(eq)
            curve_count += 1
            
            # Рисуем аппроксимированную кривую на результате (тонкой линией с акцентом)
            x_min, x_max = int(xs_clean.min()), int(xs_clean.max())
            xs_eval = np.linspace(x_min, x_max, max(200, x_max - x_min + 1))
            ys_eval = poly(xs_eval)
            pts_draw = np.column_stack((xs_eval, ys_eval)).astype(np.int32)
            cv2.polylines(result, [pts_draw], isClosed=False, color=(255, 100, 50), thickness=1, lineType=cv2.LINE_AA)
    
    # Fallback если нет кривых: простой профиль интенсивности
    if curve_count == 0:
        means = np.mean(enhanced, axis=0)  # По x теперь
        norm_means = (means - means.min()) / (means.max() - means.min() + 1e-8)
        ys_profile = (norm_means * (h_img - 1)).astype(np.int32)
        pts_draw = np.column_stack((np.arange(w_img), ys_profile)).astype(np.int32)
        cv2.polylines(result, [pts_draw], isClosed=False, color=(255, 100, 50), thickness=1, lineType=cv2.LINE_AA)
        equations.append("fallback_profile: y = mean_intensity_profile(x) (no polynomial fitted)")
    
    # 8. Сохранение PNG и TXT с уравнениями
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        name_stem = Path(base_name).stem if base_name else "graph"
        png_path = Path(out_dir) / f"graph_{name_stem}.png"
        txt_path = Path(out_dir) / f"graph_{name_stem}.txt"
        cv2.imwrite(str(png_path), result)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Source: {base_name}\n")
            f.write(f"Image size: {w_img}x{h_img}\n")
            f.write(f"Levels: {n_levels}\n")
            f.write(f"Contours fitted: {curve_count}\n\n")
            for line in equations:
                f.write(line + "\n")
    
    return result