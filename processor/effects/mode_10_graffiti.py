import cv2
import numpy as np
import traceback
from pathlib import Path

# Попробуем импортировать MiniBatchKMeans (быстрее для видео)
try:
    from sklearn.cluster import MiniBatchKMeans as _MBK
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False
    _MBK = None


def apply_graffiti(
    img,
    w=None, h=None,
    out_dir=None, base_name=None,
    k_colors=6,
    overspray_strength=0.7,
    drip_strength=0.18,
    wall_grain=8,
    paint_saturation_boost=1.35,
    sharpness=1.15,
    texture_strength=0.12,
    # video-specific helpers
    prev_palette=None,         # передавайте centers из предыдущего кадра для стабильности
    return_palette=False,      # если True, вернёт (out, centers)
    fast_mode=True             # если True -> более быстрые параметры для видео
):
    """
    Устойчивый вариант apply_graffiti, подходящий для видео.
    - prev_palette: None или np.ndarray shape (k_colors,3) float32 — переиспользует палитру
    - return_palette: если True, вернёт (out, centers)
    - fast_mode: уменьшает количество спрея/размытия для скорости
    """

    try:
        if img is None:
            raise ValueError("Input image is None")

        # масштабирование
        if w and h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        # Защита: если кадр прилетает как путь или строка — попытаемся загрузить
        if isinstance(img, (str, Path)):
            img = cv2.imread(str(img))
            if img is None:
                raise ValueError(f"Cannot read image from path: {img}")

        # гарантируем 3 канала
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h_img, w_img = img.shape[:2]

        # Приводим картинку к float32 [0..1]
        img_f = img.astype(np.float32) / 255.0

        # Лёгкая коррекция насыщенности (в float)
        hsv = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * paint_saturation_boost, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        img_f = img.astype(np.float32) / 255.0

        # Подготовка данных для кластеризации
        flat = (img_f.reshape(-1, 3) * 255.0).astype(np.float32)  # 0..255 float32

        # Попробуем переиспользовать палитру (если передали)
        centers = None
        labels = None
        use_prev = False
        if prev_palette is not None:
            try:
                prev_palette = np.asarray(prev_palette, dtype=np.float32)
                if prev_palette.shape[1] == 3 and prev_palette.shape[0] >= k_colors:
                    # используем первые k_colors
                    centers = prev_palette[:k_colors].astype(np.float32)
                    use_prev = True
                else:
                    use_prev = False
            except Exception:
                use_prev = False

        if not use_prev:
            # кластеризация — prefer MiniBatchKMeans для скорости на видео
            try:
                if _HAVE_SKLEARN and _MBK is not None:
                    mbk = _MBK(n_clusters=k_colors, random_state=42, batch_size=1024, max_iter=50)
                    labs = mbk.fit_predict(flat)
                    centers = mbk.cluster_centers_.astype(np.float32)  # shape (k,3)
                    labels = np.asarray(labs, dtype=np.int32)
                else:
                    # fallback: cv2.kmeans (работает с float32)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
                    # cv2.kmeans требует sample count >= k, guard:
                    if flat.shape[0] < k_colors:
                        # если очень маленькое изображение — просто взять средний цвет
                        centers = np.mean(flat, axis=0, keepdims=True).astype(np.float32)
                        labels = np.zeros((flat.shape[0],), dtype=np.int32)
                    else:
                        _, labels32, centers32 = cv2.kmeans(flat, k_colors, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
                        centers = centers32.astype(np.float32)
                        labels = labels32.flatten().astype(np.int32)
            except Exception as e:
                # на случай странных типов — fallback: простая разметка по среднему цвета
                print("[apply_graffiti] clustering failed, fallback to global mean:", e)
                centers = np.mean(flat, axis=0, keepdims=True).astype(np.float32)
                labels = np.zeros((flat.shape[0],), dtype=np.int32)

        # Если labels ещё не заполнен (при prev_palette) — посчитаем ближайший центр (без полного KMeans)
        if labels is None:
            # вычисляем квадраты расстояний к центрам (может быть heavy, но обычно окончен)
            diff = flat[:, None, :] - centers[None, :, :]  # (N,k,3)
            d2 = np.sum(diff * diff, axis=2)  # (N,k)
            labels = np.argmin(d2, axis=1).astype(np.int32)

        # Safety: если centers имеют нечисловой тип — попробуем привести, иначе пересчитать из исходника
        try:
            centers = np.asarray(centers, dtype=np.float32)
            if centers.ndim != 2 or centers.shape[1] != 3:
                raise ValueError("centers shape wrong")
        except Exception:
            # fallback: mean color per cluster
            centers = np.zeros((k_colors, 3), dtype=np.float32)
            lab = labels.reshape(h_img, w_img)
            for i in range(k_colors):
                mask_i = (lab == i)
                if mask_i.any():
                    centers[i] = np.mean(img_f[mask_i], axis=0) * 255.0
                else:
                    centers[i] = np.mean(img_f, axis=(0,1)) * 255.0

        # создаём quantized для границ/визуализации (опционально)
        quantized = (centers[labels].reshape((h_img, w_img, 3))).astype(np.uint8)

        # Контуры на quantized
        gray_q = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_q, 80, 160)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        # Холсты
        paint_canvas = np.zeros((h_img, w_img, 3), dtype=np.float32)
        alpha_canvas = np.zeros((h_img, w_img), dtype=np.float32)

        # Обработка кластеров по размеру (labels -> counts)
        unique, counts = np.unique(labels, return_counts=True)
        sorted_indices = unique[np.argsort(-counts)]

        # Перебираем кластеры
        for cluster_idx in sorted_indices:
            cnt = int(counts[np.where(unique == cluster_idx)[0][0]])
            if cnt < 20:
                continue

            mask = (labels.reshape(h_img, w_img) == cluster_idx).astype(np.uint8) * 255

            # Морфология для сглаживания
            ksz = max(3, int(min(h_img, w_img) / 300))
            kernel = np.ones((ksz, ksz), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            base_soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=max(1.0, ksz * 0.6)) / 255.0
            hard_edge = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=1.2) / 255.0
            hard_edge = np.where(hard_edge > 0.6, (hard_edge - 0.6) / 0.4, 0.0)

            # Защита: цвет к float32
            try:
                col = np.asarray(centers[int(cluster_idx)], dtype=np.float32)
                if col.size != 3:
                    raise ValueError("bad color shape")
            except Exception:
                # fallback: средний цвет внутри маски
                mask_bool = mask.astype(bool)
                if mask_bool.any():
                    col = np.mean(img_f[mask_bool], axis=0).astype(np.float32) * 255.0
                else:
                    col = np.mean(img_f, axis=(0,1)).astype(np.float32) * 255.0

            # Color layer (float32)
            color_layer = np.empty((h_img, w_img, 3), dtype=np.float32)
            color_layer[..., 0] = col[0]
            color_layer[..., 1] = col[1]
            color_layer[..., 2] = col[2]

            # Нанесение краски (используем числа float32)
            paint_canvas = paint_canvas * (1 - base_soft[..., None]) + color_layer * base_soft[..., None] * (1.0 + 0.0)
            alpha_canvas = np.clip(alpha_canvas + base_soft * (1.0 - alpha_canvas), 0.0, 1.0)

            # Окантовка (темнее)
            outline_intensity = 0.35
            outline_color = np.clip(col * 0.75, 0, 255)
            paint_canvas = paint_canvas * (1 - hard_edge[..., None] * outline_intensity) + outline_color[None, None, :] * (hard_edge[..., None] * outline_intensity)

            # Overspray: микро + макро
            if fast_mode:
                micro_min = 150
                scale_div = 100  # меньше разбавление (быстрее)
            else:
                micro_min = 500
                scale_div = 40

            micro_n = int(max(micro_min, cnt // scale_div))
            ys = np.random.randint(0, h_img, micro_n)
            xs = np.random.randint(0, w_img, micro_n)

            # Vectorized inside test (mask values 0/255)
            inside = (mask[ys, xs] > 0).astype(np.float32)
            keep_prob = 0.25 + 0.75 * inside  # небольшие поправки
            keep = np.random.rand(micro_n) < keep_prob

            micro = np.zeros((h_img, w_img), dtype=np.float32)
            micro[ys[keep], xs[keep]] = 1.0
            micro = cv2.GaussianBlur(micro, (0, 0), sigmaX=1.2)

            halo = None
            try:
                dist_map = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
                halo = np.exp(-dist_map / 10.0) * (dist_map > 0).astype(np.float32)
                halo = cv2.GaussianBlur(halo, (0, 0), sigmaX=3.0)
            except Exception:
                halo = np.zeros((h_img, w_img), dtype=np.float32)

            spray = np.clip(micro * 0.8 + halo * 0.5, 0, 1.0) * overspray_strength
            paint_canvas += color_layer * spray[..., None] * 0.6  # 0.6 вместо 0.75 — чуть экономнее
            alpha_canvas = np.clip(alpha_canvas + spray * 0.25 * (1.0 - alpha_canvas), 0, 1.0)

            # Drips (реализованы векторно, защитно)
            if drip_strength > 0 and np.random.rand() < 0.65:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    if contour.size == 0:
                        continue
                    all_points = contour.reshape(-1, 2)
                    median_y = np.median(all_points[:, 1])
                    lower_points = all_points[all_points[:, 1] > median_y]

                    if len(lower_points) > 0:
                        k_pick = min(6, max(1, len(lower_points) // 3))
                        picks_idx = np.random.choice(len(lower_points), k_pick, replace=False)
                        picks = np.atleast_2d(lower_points[picks_idx]).reshape(-1, 2)

                        drip_layer = np.zeros((h_img, w_img), dtype=np.float32)
                        for p in picks:
                            px, py = int(p[0]), int(p[1])
                            length = np.random.randint(int(0.03 * h_img), int(0.1 * h_img))
                            width_base = np.random.randint(1, 3)
                            for dy in range(length):
                                yy = py + dy
                                if yy >= h_img:
                                    break
                                width = int(max(1, width_base * (1 - (dy / length) ** 2)))
                                wiggle = int(np.sin(dy / 4.0) * 2)
                                intensity = max(0.0, 1.0 - (dy / length) ** 1.4)
                                x1 = px + wiggle - width // 2
                                x2 = px + wiggle + width // 2
                                x1 = max(0, min(w_img - 1, x1))
                                x2 = max(0, min(w_img - 1, x2))
                                cv2.line(drip_layer, (x1, yy), (x2, yy), intensity, 1)
                        drip_layer = cv2.GaussianBlur(drip_layer, (0, 0), sigmaX=1.4)
                        drip_layer = np.clip(drip_layer * drip_strength * 1.1, 0, 1.0)
                        paint_canvas += color_layer * drip_layer[..., None] * 0.9
                        alpha_canvas = np.clip(alpha_canvas + drip_layer * 0.28 * (1.0 - alpha_canvas), 0, 1.0)

        # --- финализация вне цикла ---
        paint_u = paint_canvas / 255.0
        blur = cv2.GaussianBlur(paint_u, (0, 0), 1.0)
        paint_sharp = np.clip(paint_u + (paint_u - blur) * sharpness, 0, 1.0) * 255.0

        # texture (multi-scale)
        tex = np.zeros((h_img, w_img), dtype=np.float32)
        for scale in [wall_grain * 3, wall_grain * 2, wall_grain]:
            noise = np.random.normal(0, max(1.0, scale / 4.0), (max(4, h_img // 4), max(4, w_img // 4)))
            noise = cv2.resize(noise, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
            tex += noise / 3.0
        tex = cv2.GaussianBlur(tex, (0, 0), sigmaX=max(0.5, wall_grain / 3.0))
        tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-8) * 20 - 10

        # cracks
        cracks = np.zeros((h_img, w_img), dtype=np.float32)
        num_cracks = np.random.randint(2, 6) if fast_mode else np.random.randint(3, 8)
        for _ in range(num_cracks):
            start = (np.random.randint(0, w_img), np.random.randint(0, h_img))
            length = np.random.randint(max(10, h_img // 6), max(20, h_img // 3))
            angle = np.random.uniform(0, 2 * np.pi)
            for i in range(length):
                x = int(start[0] + i * np.cos(angle) + np.random.randint(-2, 3))
                y = int(start[1] + i * np.sin(angle) + np.random.randint(-2, 3))
                if 0 <= x < w_img and 0 <= y < h_img:
                    cracks[y, x] = 1.0
            angle += np.random.uniform(-0.2, 0.2)
        cracks = cv2.GaussianBlur(cracks, (0, 0), sigmaX=1.0)
        cracks = np.clip(cracks * 20, 0, 255).astype(np.uint8)

        wall_base = np.full((h_img, w_img, 3), 190, dtype=np.float32)
        wall_base += tex[..., None]
        wall_base = np.clip(wall_base, 100, 220)
        wall_base = cv2.subtract(wall_base.astype(np.uint8), np.repeat(cracks[..., None], 3, axis=2))

        paint_mod = paint_sharp.astype(np.float32) * (1.0 - texture_strength * (tex[..., None] / 20 + 0.5))
        alpha3 = np.repeat(alpha_canvas[..., None], 3, axis=2)
        out_f = (paint_mod * alpha3 + wall_base * (1.0 - alpha3)).astype(np.uint8)
        out = cv2.convertScaleAbs(out_f, alpha=1.08, beta=4)

        # save if needed
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            name_stem = Path(base_name).stem if base_name else "graffiti"
            cv2.imwrite(str(Path(out_dir) / f"graffiti_{name_stem}.png"), out)

        # return palette if requested (centers in 0..255 float32)
        if return_palette:
            return out, centers.astype(np.float32)

        return out

    except Exception as e:
        print(f"[apply_graffiti ERROR] base_name={base_name} -> {e}")
        traceback.print_exc()
        return img if img is not None else np.zeros((256, 256, 3), np.uint8)
