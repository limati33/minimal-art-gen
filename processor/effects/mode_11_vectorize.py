# processor/effects/mode_11_vectorize.py
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab, lab2rgb

def apply_vectorize(img, w, h, out_dir, base_name):
    # ----------------- ПАРАМЕТРЫ (подстрой под вкус) -----------------
    area = max(1, int(w) * int(h))
    n_segments = int(np.clip(area / 1200, 120, 1200))   # чем меньше — тем крупнее пятна
    compactness = 10.0                                  # 5..20 — влияет на форму сегментов
    # сколько итоговых цветов желаем примерно (будет адаптировано под сегменты)
    target_colors = int(np.clip(n_segments // 40, 4, 64))
    # минимальный размер сегмента (в пикселях). Мелкие сегменты будут поглощены соседями
    small_seg_threshold = int(np.clip(area / 5000, 20, 800))
    # постобработка медианой: 0 — отключено, 3..7 — мягкое сглаживание
    median_blur_ksize = 3
    # ----------------------------------------------------------------

    img_h, img_w = img.shape[:2]
    if (img_w, img_h) != (w, h):
        w, h = img_w, img_h

    # Немного сгладим исходник (убираем мелкие текстуры, сохраняем границы)
    blur = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # SLIC ожидает float [0..1]
    blur_float = np.clip(blur.astype(np.float32) / 255.0, 0.0, 1.0)

    # 1) SLIC: суперпиксели
    try:
        segments = slic(
            blur_float,
            n_segments=n_segments,
            compactness=compactness,
            sigma=1,
            start_label=0
        ).astype(np.int32)
    except Exception:
        # На случай проблем — fallback: просто вернуть слегка размытый оригинал
        return cv2.medianBlur(blur, 3)

    # 2) Переводим в LAB и готовим данные для bincount
    lab = rgb2lab(blur_float)  # float in expected range
    lab_flat = lab.reshape(-1, 3)
    seg_flat = segments.reshape(-1)

    # Получаем компактные индексы сегментов 0..N-1
    unique_labels, inverse = np.unique(seg_flat, return_inverse=True)
    n_segs_actual = unique_labels.size
    inv = inverse  # индекс каждого пикселя в 0..n_segs_actual-1

    # 3) Средний цвет (LAB) для каждого сегмента — через bincount (быстро)
    counts = np.bincount(inv)
    counts_safe = np.where(counts == 0, 1, counts).astype(np.float64)

    sums_L = np.bincount(inv, weights=lab_flat[:, 0])
    sums_a = np.bincount(inv, weights=lab_flat[:, 1])
    sums_b = np.bincount(inv, weights=lab_flat[:, 2])

    seg_means_lab = np.vstack((sums_L / counts_safe,
                               sums_a / counts_safe,
                               sums_b / counts_safe)).T  # (n_segs_actual, 3)

    # 4) Уменьшаем палитру (опционально) — простой kmeans если сегментов много
    # если сегментов мало — берём seg_means как есть
    from cv2 import kmeans as cv2_kmeans
    Z = seg_means_lab.astype(np.float32)
    K = min(max(2, target_colors), n_segs_actual)
    if n_segs_actual > K:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, labels_centers, centers = cv2_kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels_centers = labels_centers.flatten()
        centers = centers.astype(np.float32)
    else:
        # мало сегментов — каждая своя палитра
        labels_centers = np.arange(n_segs_actual, dtype=np.int32)
        centers = Z.astype(np.float32)

    # centers (LAB) -> цвет для каждого сегмента (по метке labels_centers)
    centers_per_segment_lab = centers[labels_centers]

    # 5) Слияние мелких сегментов: если сегмент меньше threshold, берём
    # наиболее частый сосед (по 3x3 окрестности) и присваиваем ему цвет.
    if small_seg_threshold > 0:
        inv_map = inv.reshape(h, w)  # индекс сегмента (0..n_segs_actual-1) для каждого пикселя
        seg_sizes = counts  # уже вычислены
        small_ids = np.where(seg_sizes < small_seg_threshold)[0]
        if small_ids.size:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # inv_map uint32 for bincount
            inv_map_u32 = inv_map.astype(np.int32)
            for sid in small_ids:
                # строим маску малого сегмента
                mask = (inv_map_u32 == sid).astype(np.uint8)
                if mask.sum() == 0:
                    continue
                # расширяем область и берём соседние сегменты
                dil = cv2.dilate(mask, kernel, iterations=2).astype(np.bool_)
                border_mask = dil & (~(mask.astype(bool)))
                if not np.any(border_mask):
                    continue
                neighbor_indices = inv_map_u32[border_mask]
                # частотный сосед
                new_sid = np.bincount(neighbor_indices).argmax()
                # присвоим цвет этому маленькому сегменту — просто переназначим его центр
                centers_per_segment_lab[sid] = centers_per_segment_lab[new_sid]
                # не меняем inv_map сам — достаточно изменить цвет сегмента
                # (поскольку в финале мы окрашиваем по index -> centers_per_segment_lab)
    
    # 6) Собираем итоговое изображение: для каждого пикселя берём цвет центра своего сегмента
    final_lab_flat = centers_per_segment_lab[inv]  # shape (h*w, 3)
    final_lab = final_lab_flat.reshape((h, w, 3))

    # Конвертация LAB->RGB (lab2rgb возвращает float 0..1)
    final_rgb = lab2rgb(final_lab)
    final_rgb_uint8 = np.clip((final_rgb * 255.0), 0, 255).astype(np.uint8)

    # 7) Небольшая постобработка — медианный фильтр, чтобы убрать «пиксели»
    if median_blur_ksize and median_blur_ksize >= 3 and median_blur_ksize % 2 == 1:
        final_rgb_uint8 = cv2.medianBlur(final_rgb_uint8, median_blur_ksize)

    return final_rgb_uint8
