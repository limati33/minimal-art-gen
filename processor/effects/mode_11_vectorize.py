# processor/effects/mode_11_vectorize.py
import cv2
import numpy as np
from skimage.segmentation import slic, find_boundaries

def apply_vectorize(img, w, h, out_dir, base_name):
    """
    Минималистичная векторизация (плоский арт).
    Возвращает uint8 RGB изображение (h, w, 3).

    - Автоматически подбирает число суперпикселей и цветов по размеру входа.
    - Использует SLIC для аккуратного выделения областей, затем KMeans на средних цветах сегментов.
    - Рисует чёрные контуры по границам сегментов (толщину можно менять).
    """

    # --- Параметры (можно подправить) ---
    # Авточисло сегментов в зависимости от площади
    area = max(1, int(w) * int(h))
    n_segments = int(np.clip(area / 1500, 100, 1000))   # примерно: 100..1000 сегментов
    compactness = 12.0                                  # компактность SLIC (форма vs цвет)
    n_colors = int(np.clip(n_segments / 50, 4, 12))     # количество итоговых цветов (4..12)
    line_thickness = 1                                  # толщина контура (px). повысь для грубых линий

    # Убедимся, что img — RGB uint8 и размеры совпадают
    if img is None:
        raise ValueError("apply_vectorize: img is None")
    img_h, img_w = img.shape[:2]
    if (img_w, img_h) != (w, h):
        # если переданные w,h не совпадают с реальными — используем реальные
        w, h = img_w, img_h

    # --- 1) Сглаживаем (убираем мелкие текстуры) ---
    # Bilateral даёт хороший баланс—сохраняет края
    blur = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # SLIC ожидает float image в диапазоне [0,1]
    blur_float = np.clip(blur.astype(np.float32) / 255.0, 0.0, 1.0)

    try:
        # --- 2) SLIC: суперпиксели ---
        segments = slic(
            blur_float,
            n_segments=n_segments,
            compactness=compactness,
            sigma=1,
            start_label=0
        ).astype(np.int32)
    except Exception:
        # Если SLIC по какой-то причине упал (редко), fallback — простой уменьшенный SLIC-подобный подход:
        # используем классический kmeans на всей картинке (медленнее/грубее).
        Z = blur.reshape((-1, 3)).astype(np.float32)
        Kf = max(4, min(12, n_colors))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels_full, centers_full = cv2.kmeans(Z, Kf, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        centers_full = np.uint8(centers_full)
        quant_full = centers_full[labels_full.flatten()].reshape((img_h, img_w, 3))
        # Возвращаем результат (быстрый вариант)
        return quant_full

    # --- 3) Средний цвет для каждого сегмента ---
    unique_labels = np.unique(segments)
    n_segs_actual = unique_labels.size

    # Быстро считать средние цвета: создаём массив и суммируем
    seg_colors = np.zeros((n_segs_actual, 3), dtype=np.float64)
    seg_counts = np.zeros((n_segs_actual,), dtype=np.int32)

    # Берём значения из blur (uint8)
    blur_uint8 = blur.reshape((-1, 3))
    seg_flat = segments.reshape(-1)
    # Map label value -> index in unique_labels
    label_to_index = {lab: idx for idx, lab in enumerate(unique_labels)}
    indices = np.vectorize(label_to_index.get)(seg_flat)
    for ch in range(3):
        channel_vals = blur_uint8[:, ch].astype(np.float64)
        # суммируем по сегментам
        for idx in range(n_segs_actual):
            mask_idx = (indices == idx)
            if mask_idx.any():
                seg_colors[idx, ch] = channel_vals[mask_idx].sum()
        # (we will divide by counts below)

    # посчитаем counts (делаем один проход, быстрее)
    for idx in range(n_segs_actual):
        seg_counts[idx] = np.count_nonzero(indices == idx)
    # избегаем деления на ноль
    seg_counts_safe = np.where(seg_counts == 0, 1, seg_counts)
    seg_colors = (seg_colors.T / seg_counts_safe).T
    seg_colors = np.clip(seg_colors, 0, 255).astype(np.uint8)  # (n_segs_actual, 3)

    # --- 4) KMeans на средних цветах сегментов (чтобы сократить палитру) ---
    Z_segs = seg_colors.astype(np.float32)
    K = max(2, min(12, n_colors))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    # cv2.kmeans ожидает non-empty and proper shape
    if Z_segs.shape[0] <= K:
        # слишком мало сегментов — просто используем seg_colors как палитру
        centers = Z_segs
        seg_to_center_labels = np.arange(Z_segs.shape[0], dtype=np.int32)
    else:
        _, seg_to_center_labels, centers = cv2.kmeans(Z_segs, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        seg_to_center_labels = seg_to_center_labels.flatten()
        centers = np.uint8(centers)

    # Для каждого сегмента — цвет из centers
    centers_per_segment = centers[seg_to_center_labels]  # (n_segs_actual, 3) if kmeans used, otherwise appropriate

    # --- 5) Собираем плоское изображение: для каждого пикселя — цвет своего сегмента ---
    # построим index_map: для каждый пикселя — индекс сегмента в 0..n_segs_actual-1
    index_map = np.zeros_like(segments, dtype=np.int32)
    for i, lab in enumerate(unique_labels):
        index_map[segments == lab] = i
    flat = centers_per_segment[index_map]  # shape (h, w, 3)
    flat = np.ascontiguousarray(flat.astype(np.uint8))

    return flat
