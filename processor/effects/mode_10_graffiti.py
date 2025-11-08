# processor/effects/mode_10_graffiti.py
import cv2
import numpy as np
from pathlib import Path
import traceback

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
    texture_strength=0.12
):
    try:
        if img is None:
            raise ValueError("Input image is None")
        
        # Масштабирование
        if w and h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        h_img, w_img = img.shape[:2]
        
        # Коррекция контраста и насыщенности
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * paint_saturation_boost, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # KMeans квантование
        data = img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k_colors, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)
        labels = labels.flatten()
        quantized = centers[labels].reshape((h_img, w_img, 3))
        
        # Контуры на quantized
        gray_q = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_q, 80, 160)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # Холсты
        paint_canvas = np.zeros((h_img, w_img, 3), dtype=np.float32)
        alpha_canvas = np.zeros((h_img, w_img), dtype=np.float32)
        
        # Обработка кластеров по размеру
        unique, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        for idx in sorted_indices:
            cluster_idx = unique[idx]
            cnt = counts[idx]
            if cnt < 50:
                continue
            mask = (labels.reshape(h_img, w_img) == cluster_idx).astype(np.uint8) * 255
            
            # Морфология для сглаживания
            ksz = max(3, int(min(h_img, w_img) / 200))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((ksz, ksz), np.uint8), iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((ksz, ksz), np.uint8), iterations=1)
            
            # Мягкая маска
            base_soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=ksz * 0.8) / 255.0
            
            # Жесткая маска для окантовки
            hard_edge = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=1.5) / 255.0
            hard_edge = np.where(hard_edge > 0.6, (hard_edge - 0.6) / 0.4, 0.0)  # Усиление краев
            
            col = centers[cluster_idx].astype(np.float32)
            
            # Вариация цвета
            color_variation = (np.random.rand(h_img, w_img, 3).astype(np.float32) - 0.5) * 20.0
            color_layer = np.clip(col[None, None, :] + color_variation, 0, 255)
            
            # Нанесение краски
            paint_canvas = paint_canvas * (1 - base_soft[..., None]) + color_layer * base_soft[..., None]
            alpha_canvas = np.clip(alpha_canvas + base_soft * (1.0 - alpha_canvas), 0.0, 1.0)
            
            # Окантовка
            outline_intensity = 0.4
            outline_color = np.clip(col * 0.8, 0, 255)  # Темнее для контура
            paint_canvas = paint_canvas * (1 - hard_edge[..., None] * outline_intensity) + outline_color[None, None, :] * (hard_edge[..., None] * outline_intensity)
            
            # Overspray: улучшенный с directional bias
            # Micro spray: random points with density based on mask
            micro_n = int(max(500, cnt // 40))
            ys = np.random.randint(0, h_img, micro_n)
            xs = np.random.randint(0, w_img, micro_n)
            inside = cv2.pointPolygonTest(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], (xs, ys), False) >= 0
            keep_prob = 0.3 + 0.7 * inside.astype(float)
            keep = np.random.rand(micro_n) < keep_prob
            micro = np.zeros((h_img, w_img), dtype=np.float32)
            micro[ys[keep], xs[keep]] = 1.0
            micro = cv2.GaussianBlur(micro, (0, 0), sigmaX=1.5)
            
            # Macro spray: halo with slight outward bias
            dist_map = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
            halo = np.exp(-dist_map / 10.0) * (dist_map > 0)
            halo = cv2.GaussianBlur(halo.astype(np.float32), (0, 0), sigmaX=5.0)
            
            spray = np.clip(micro * 0.8 + halo * 0.5, 0, 1.0) * overspray_strength
            paint_canvas += color_layer * spray[..., None] * 0.75
            alpha_canvas = np.clip(alpha_canvas + spray * 0.25 * (1.0 - alpha_canvas), 0, 1.0)
            
            # Drips: более реалистичные, от нижних краев
            if drip_strength > 0 and np.random.rand() < 0.7:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = contours[0]  # Largest contour
                    lower_points = contour[contour[:, :, 1] > np.median(contour[:, :, 1])]
                    if len(lower_points) > 0:
                        picks_idx = np.random.choice(len(lower_points), min(8, len(lower_points) // 2), replace=False)
                        picks = lower_points[picks_idx]
                        drip_layer = np.zeros((h_img, w_img), dtype=np.float32)
                        for p in picks:
                            px, py = p[0]
                            length = np.random.randint(int(0.04 * h_img), int(0.15 * h_img))
                            width_base = np.random.randint(2, 4)
                            for dy in range(length):
                                yy = py + dy
                                if yy >= h_img:
                                    break
                                width = width_base * (1 - (dy / length) ** 2)
                                wiggle = int(np.sin(dy / 5.0) * 2)
                                intensity = max(0.0, 1.0 - (dy / length) ** 1.5)
                                cv2.line(drip_layer, (px + wiggle - int(width/2), yy), (px + wiggle + int(width/2), yy), intensity, 1)
                        drip_layer = cv2.GaussianBlur(drip_layer, (0, 0), sigmaX=1.8)
                        drip_layer = np.clip(drip_layer * drip_strength * 1.2, 0, 1.0)
                        paint_canvas += color_layer * drip_layer[..., None] * 0.9
                        alpha_canvas = np.clip(alpha_canvas + drip_layer * 0.3 * (1.0 - alpha_canvas), 0, 1.0)
        
        # Sharpen
        paint_u = paint_canvas / 255.0
        blur = cv2.GaussianBlur(paint_u, (0, 0), 1.2)
        paint_sharp = np.clip(paint_u + (paint_u - blur) * sharpness, 0, 1.0) * 255.0
        
        # Улучшенная текстура стены: multi-scale noise для бетона
        tex = np.zeros((h_img, w_img), dtype=np.float32)
        for scale in [wall_grain * 4, wall_grain * 2, wall_grain]:
            noise = np.random.normal(0, scale / 4, (h_img // 4, w_img // 4))
            noise = cv2.resize(noise, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
            tex += noise / len([4,2,1])  # Normalize
        tex = cv2.GaussianBlur(tex, (0, 0), sigmaX=wall_grain / 3)
        tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-8) * 20 - 10  # +/- 10
        
        # Добавление трещин в бетоне
        cracks = np.zeros((h_img, w_img), dtype=np.float32)
        num_cracks = np.random.randint(3, 8)
        for _ in range(num_cracks):
            start = (np.random.randint(0, w_img), np.random.randint(0, h_img))
            length = np.random.randint(h_img // 4, h_img // 2)
            angle = np.random.uniform(0, 2*np.pi)
            for i in range(length):
                x = int(start[0] + i * np.cos(angle) + np.random.randint(-2, 3))
                y = int(start[1] + i * np.sin(angle) + np.random.randint(-2, 3))
                if 0 <= x < w_img and 0 <= y < h_img:
                    cracks[y, x] = 1.0
            angle += np.random.uniform(-0.2, 0.2)  # Wiggle
        cracks = cv2.GaussianBlur(cracks, (0, 0), sigmaX=1.0)
        cracks = np.clip(cracks * 20, 0, 255).astype(np.uint8)  # Темные трещины
        
        # Базовая стена
        wall_base = np.full((h_img, w_img, 3), 190, dtype=np.float32)  # Светло-серый бетон
        wall_base += tex[..., None]
        wall_base = np.clip(wall_base, 100, 220)
        wall_base = cv2.subtract(wall_base.astype(np.uint8), np.repeat(cracks[..., None], 3, axis=2))
        
        # Модификация краски по текстуре
        paint_mod = paint_sharp.astype(np.float32) * (1.0 - texture_strength * (tex[..., None] / 20 + 0.5))
        
        # Комбинирование
        alpha3 = np.repeat(alpha_canvas[..., None], 3, axis=2)
        out_f = (paint_mod * alpha3 + wall_base * (1.0 - alpha3)).astype(np.uint8)
        
        # Финальная коррекция
        out = cv2.convertScaleAbs(out_f, alpha=1.08, beta=4)
        
        # Сохранение
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            name_stem = Path(base_name).stem if base_name else "graffiti"
            cv2.imwrite(str(Path(out_dir) / f"graffiti_{name_stem}.png"), out)
        
        return out
    except Exception as e:
        print(f"[apply_graffiti ERROR] base_name={base_name} -> {e}")
        traceback.print_exc()
        return img if img is not None else np.zeros((256, 256, 3), np.uint8)