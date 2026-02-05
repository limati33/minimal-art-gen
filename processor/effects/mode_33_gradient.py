# processor/effects/mode_33_gradient.py
import cv2
import numpy as np
import os

def apply_gradient(img, w=None, h=None, out_dir=None, base_name=None):
    """
    Mode 33: Градиент по цветовым сегментам
    Контуры строятся ТОЛЬКО по цвету, а не по яркости
    """
    if img is None:
        return None

    # --- Resize ---
    h0, w0 = img.shape[:2]
    if w and h and (w0 != w or h0 != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    # --- Цветовая сегментация ---
    # MeanShift хорошо склеивает близкие цвета
    shifted = cv2.pyrMeanShiftFiltering(img, sp=12, sr=25)

    # LAB — лучшее пространство для цветовой близости
    lab = cv2.cvtColor(shifted, cv2.COLOR_BGR2LAB)

    # --- Квантование цветов (уменьшаем количество сегментов) ---
    Z = lab.reshape((-1, 3)).astype(np.float32)
    K = 12  # количество цветовых сегментов

    _, labels, centers = cv2.kmeans(
        Z, K, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        3,
        cv2.KMEANS_PP_CENTERS
    )

    labels = labels.reshape(lab.shape[:2])

    # Холст результата
    result = np.zeros_like(img)

    # --- Обрабатываем каждый цветовой сегмент ---
    for i in range(K):
        mask = (labels == i).astype(np.uint8) * 255

        # Чистим мусор
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue

            cnt_mask = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

            # --- Средний цвет ---
            mean_bgr = cv2.mean(img, mask=cnt_mask)[:3]
            pixel = np.uint8([[mean_bgr]])
            hls = cv2.cvtColor(pixel, cv2.COLOR_BGR2HLS)[0, 0]

            H, L, S = map(int, hls)
            L1 = min(255, L + 35)
            L2 = max(0, L - 35)

            bright = cv2.cvtColor(
                np.uint8([[[H, L1, S]]]), cv2.COLOR_HLS2BGR
            )[0, 0].astype(np.float32)

            dark = cv2.cvtColor(
                np.uint8([[[H, L2, S]]]), cv2.COLOR_HLS2BGR
            )[0, 0].astype(np.float32)

            # --- Направление градиента ---
            try:
                vx, vy, cx, cy = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            except:
                continue

            x, y, w_box, h_box = cv2.boundingRect(cnt)

            gx, gy = np.meshgrid(
                np.arange(w_box),
                np.arange(h_box)
            )

            rel_x = gx + x - cx
            rel_y = gy + y - cy

            proj = rel_x * vx + rel_y * vy

            roi_mask = cnt_mask[y:y+h_box, x:x+w_box]
            valid = proj[roi_mask > 0]
            if valid.size == 0:
                continue

            p_min, p_max = np.percentile(valid, [10, 90])
            denom = max(p_max - p_min, 1e-5)

            t = np.clip((proj - p_min) / denom, 0, 1)
            t = t[:, :, None]

            patch = dark * (1 - t) + bright * t
            patch = np.clip(patch, 0, 255).astype(np.uint8)

            roi = result[y:y+h_box, x:x+w_box]
            mask3 = cv2.merge([roi_mask]*3)
            result[y:y+h_box, x:x+w_box] = np.where(mask3 > 0, patch, roi)

    result = cv2.medianBlur(result, 3)

    if out_dir and base_name:
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(out_dir, f"{base_name}_mode33_gradient.png"),
            result
        )

    return result
