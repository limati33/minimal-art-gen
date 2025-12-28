# processor/effects/mode_11_vectorize_svg.py
import os
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab, lab2rgb

def apply_vectorize(img, w, h, out_dir, base_name,
                    export_svg=True,        # ВКЛЮЧИЛ ПО УМОЛЧАНИЮ
                    svg_simplify=1.2,       # степень упрощения контура
                    svg_scale=1.0,          # масштаб координат
                    svg_fill_opacity=1.0,
                    median_blur_ksize=3):
    """
    Векторизует изображение (SLIC) и сохраняет результат в SVG и PNG.
    Возвращает растровое изображение для дальнейшей цепочки эффектов.
    """

    # безопасность: проверяем вход
    if img is None:
        return None

    img_h, img_w = img.shape[:2]
    # Если размеры не совпадают с требуемыми - ресайзим
    if w and h and (img_w != w or img_h != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
        w, h = int(w), int(h)

    # небольшое сглаживание исходника для лучшей сегментации
    blur = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    blur_float = np.clip(blur.astype(np.float32) / 255.0, 0.0, 1.0)

    # Параметры сегментации
    area = max(1, int(w) * int(h))
    n_segments = int(np.clip(area / 1200, 120, 1200))
    compactness = 10.0
    target_colors = int(np.clip(n_segments // 40, 4, 64))
    small_seg_threshold = int(np.clip(area / 5000, 20, 800))

    try:
        segments = slic(
            blur_float,
            n_segments=n_segments,
            compactness=compactness,
            sigma=1,
            start_label=0
        ).astype(np.int32)
    except Exception:
        # fallback: если SLIC упал, просто возвращаем размытый оригинал
        print("Warning: SLIC failed in vectorize.")
        return cv2.medianBlur(img, 3)

    # LAB и вычисление среднего цвета для сегментов
    lab = rgb2lab(blur_float)
    lab_flat = lab.reshape(-1, 3)
    seg_flat = segments.reshape(-1)

    unique_labels, inverse = np.unique(seg_flat, return_inverse=True)
    n_segs_actual = unique_labels.size
    inv = inverse
    counts = np.bincount(inv)
    counts_safe = np.where(counts == 0, 1, counts).astype(np.float64)

    sums_L = np.bincount(inv, weights=lab_flat[:, 0])
    sums_a = np.bincount(inv, weights=lab_flat[:, 1])
    sums_b = np.bincount(inv, weights=lab_flat[:, 2])

    seg_means_lab = np.vstack((sums_L / counts_safe,
                               sums_a / counts_safe,
                               sums_b / counts_safe)).T

    # Кластеризация цветов (уменьшение палитры)
    Z = seg_means_lab.astype(np.float32)
    K = min(max(2, target_colors), n_segs_actual)
    
    if n_segs_actual > K:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, labels_centers, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels_centers = labels_centers.flatten()
        centers = centers.astype(np.float32)
    else:
        labels_centers = np.arange(n_segs_actual, dtype=np.int32)
        centers = Z.astype(np.float32)

    centers_per_segment_lab = centers[labels_centers]

    # Слияние слишком мелких областей
    if small_seg_threshold > 0:
        inv_map = inv.reshape(h, w)
        seg_sizes = counts
        small_ids = np.where(seg_sizes < small_seg_threshold)[0]
        
        if small_ids.size:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            inv_map_u32 = inv_map.astype(np.int32)
            
            # Оптимизация: проходим только по существующим мелким сегментам
            # (этот цикл может быть долгим на 4K, но на обычных фото ок)
            for sid in small_ids:
                mask = (inv_map_u32 == sid).astype(np.uint8)
                if mask.sum() == 0: continue
                
                dil = cv2.dilate(mask, kernel, iterations=2).astype(bool)
                border_mask = dil & (~(mask.astype(bool)))
                
                if not np.any(border_mask): continue
                
                neighbor_indices = inv_map_u32[border_mask]
                if neighbor_indices.size == 0: continue
                
                # Находим самого частого соседа
                new_sid = np.bincount(neighbor_indices).argmax()
                centers_per_segment_lab[sid] = centers_per_segment_lab[new_sid]

    # Формируем итоговую картинку (растр)
    final_lab_flat = centers_per_segment_lab[inv]
    final_lab = final_lab_flat.reshape((h, w, 3))
    final_rgb = lab2rgb(final_lab)
    final_rgb_uint8 = np.clip((final_rgb * 255.0), 0, 255).astype(np.uint8)

    if median_blur_ksize and median_blur_ksize >= 3:
        # Убедимся, что ksize нечетный
        k = median_blur_ksize if median_blur_ksize % 2 == 1 else median_blur_ksize + 1
        final_rgb_uint8 = cv2.medianBlur(final_rgb_uint8, k)

    # -------------------------------------------------------------
    # ЭКСПОРТ ФАЙЛОВ (SVG + PNG)
    # -------------------------------------------------------------
    if export_svg and out_dir and base_name:
        os.makedirs(out_dir, exist_ok=True)
        
        svg_filename = f"{base_name}_vector.svg"
        png_filename = f"{base_name}_vector.png"
        
        svg_path = os.path.join(out_dir, svg_filename)
        png_path = os.path.join(out_dir, png_filename)

        # 1. Сохраняем PNG превью
        cv2.imwrite(png_path, cv2.cvtColor(final_rgb_uint8, cv2.COLOR_RGB2BGR))

        # 2. Генерируем SVG
        labels = np.unique(segments)
        entries = []
        
        # Сбор данных для SVG
        for lab_id in labels:
            mask = (segments == lab_id).astype(np.uint8)
            # Если после слияния мелких этот регион поменял цвет - берем новый цвет
            # Но геометрия осталась от segments (SLIC). 
            # Это упрощение: мелкие сегменты визуально сольются по цвету.
            
            area_pixels = int(mask.sum())
            
            lab_col = centers_per_segment_lab[lab_id].reshape(1, 1, 3)
            rgb_col = lab2rgb(lab_col)[0, 0]
            rgb_u8 = np.clip((rgb_col * 255.0), 0, 255).astype(np.uint8)
            hex_color = f"#{rgb_u8[0]:02x}{rgb_u8[1]:02x}{rgb_u8[2]:02x}"
            
            entries.append((lab_id, area_pixels, hex_color))

        # Сортировка: большие пятна рисуем первыми (фон), мелкие сверху
        entries.sort(key=lambda x: x[1], reverse=True)

        w_view = int(round(w * svg_scale))
        h_view = int(round(h * svg_scale))
        
        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_view}" height="{h_view}" viewBox="0 0 {w} {h}">',
            f'<rect width="100%" height="100%" fill="white" />' # Белая подложка
        ]

        for lab_id, area_px, hex_color in entries:
            mask = (segments == lab_id).astype(np.uint8) * 255
            if not np.any(mask): continue

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cnt.shape[0] < 3: continue
                
                # Упрощение кривых для SVG
                eps = float(svg_simplify) * max(w, h) / 1000.0
                approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
                
                if len(approx) < 3: continue
                
                pts = approx.reshape(-1, 2)
                # Игнорируем микро-шум
                if cv2.contourArea(pts) < 2.0: continue
                
                path_d = "M " + " L ".join(f"{x},{y}" for x, y in pts) + " Z"
                svg_lines.append(f'<path d="{path_d}" fill="{hex_color}" fill-opacity="{svg_fill_opacity}" stroke="none" />')

        svg_lines.append('</svg>')
        
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(svg_lines))

        print(f"   [Vector] Saved SVG: {svg_filename}")
        print(f"   [Vector] Saved PNG: {png_filename}")

    # Важно: возвращаем только изображение, чтобы не ломать pipeline
    return final_rgb_uint8