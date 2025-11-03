# processor/effects/mode_21_xray.py
import cv2
import numpy as np
from pathlib import Path

def apply_xray(img, w=None, h=None, out_dir=None, base_name=None):
    if img is None:
        raise ValueError("apply_xray: img is None")

    # масштабирование (если заданы)
    if w and h:
        try:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        except Exception:
            pass

    # базовая градация серого и CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)

    # инверсия: светлое = просвечивает
    inv = 255 - gray_eq
    inv = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h_img, w_img = inv.shape[:2]

    # LUT для рентген-палитры (можно менять контрольные точки)
    vals = np.arange(256, dtype=np.float32)
    xp = [0, 96, 192, 255]
    b_fp = [12, 140, 220, 255]
    g_fp = [6, 90, 180, 235]
    r_fp = [0, 20, 90, 180]
    B = np.interp(vals, xp, b_fp).astype(np.uint8)
    G = np.interp(vals, xp, g_fp).astype(np.uint8)
    R = np.interp(vals, xp, r_fp).astype(np.uint8)
    lut = np.stack([B, G, R], axis=1)
    colored = lut[inv]  # (h, w, 3)
    xray = colored.astype(np.float32)

    # глубинная маска (для натуральных вариаций просвечивания)
    depth_mask = cv2.GaussianBlur(gray_eq, (0,0), 14)
    depth_norm = cv2.normalize(depth_mask.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)[..., None]
    xray = xray * (0.9 + 0.35 * depth_norm)  # чуть усилить на "тонких" участках

    # ----------------------
    # BLOOM: локальное свечение
    # ----------------------
    # параметры для настройки
    bloom_strength = 0.18   # <-- увеличь, если хочешь сильнее свечения на белых участках
    bloom_blur = 40         # <-- размытие маски (чем больше — тем мягче область)
    bright_pct = 78         # <-- процентиль для выделения "ярких" пикселей

    # выделяем яркие участки по процентилю и размазываем
    thresh_val = int(np.percentile(inv, bright_pct))
    _, bright_mask = cv2.threshold(inv, thresh_val, 255, cv2.THRESH_BINARY)
    bloom_mask = cv2.GaussianBlur(bright_mask, (0,0), bloom_blur).astype(np.float32) / 255.0
    bloom_mask = np.clip(bloom_mask, 0.0, 1.0)[..., None]

    # применяем bloom локально (усиливаем яркие области пропорционально маске)
    xray = np.clip(xray * (1.0 + bloom_strength * bloom_mask), 0, 255)

    # ----------------------
    # царапины / пыль (тонкие светлые линии и точки)
    # ----------------------
    scratch_layer = np.zeros((h_img, w_img), dtype=np.float32)
    n_scratches = max(2, int(min(h_img, w_img) / 140))
    for i in range(n_scratches):
        x0 = np.random.randint(0, w_img)
        y0 = np.random.randint(0, h_img)
        angle = np.random.uniform(-0.6, 0.6)
        length = np.random.randint(int(0.35 * max(w_img, h_img)), int(0.85 * max(w_img, h_img)))
        x1 = int(np.clip(x0 + length * np.cos(angle), 0, w_img-1))
        y1 = int(np.clip(y0 + length * np.sin(angle), 0, h_img-1))
        thickness = np.random.randint(1, 3)
        cv2.line(scratch_layer, (x0, y0), (x1, y1), 1.0, thickness=thickness)
    n_dust = int((w_img * h_img) / 35000)
    for _ in range(n_dust):
        rx = np.random.randint(0, w_img)
        ry = np.random.randint(0, h_img)
        rr = np.random.randint(1, 3)
        cv2.circle(scratch_layer, (rx, ry), rr, 1.0, -1)
    scratch_layer = cv2.GaussianBlur(scratch_layer, (0,0), 2.0)
    scratch_layer = np.clip(scratch_layer, 0.0, 1.0)
    scratch_intensity = 0.22
    xray = np.clip(xray + (scratch_layer[..., None] * 255.0 * scratch_intensity), 0, 255)

    # ----------------------
    # зерно (плёночный шум)
    # ----------------------
    grain_sigma = 3.5
    noise = np.random.normal(0.0, grain_sigma, xray.shape).astype(np.float32)
    xray = np.clip(xray + noise, 0, 255)

    # ----------------------
    # легкая виньетка (слабая, чтобы не перекрывать локальный bloom)
    # ----------------------
    vignette_strength = 0.08  # маленькое значение: только для лёгкой глубины
    yv, xv = np.ogrid[:h_img, :w_img]
    cy, cx = h_img * 0.48, w_img * 0.5
    dist = np.sqrt((xv - cx)**2 + (yv - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    vign = 1.0 - (dist / (0.98 * max_dist))
    vign = np.clip(vign, 0, 1) ** 1.4
    vign = vign[..., None]
    # комбинируем: почти не затемняем, только немного контрастируем края
    xray = xray * (1.0 - vignette_strength + vignette_strength * vign) + 6.0 * (1.0 - vign)

    # финальная тонировка и контраст
    xray = np.clip(xray, 0, 255).astype(np.uint8).astype(np.float32)
    alpha = 1.03
    beta = -4
    xray = cv2.convertScaleAbs(xray, alpha=alpha, beta=beta).astype(np.float32)
    gamma = 0.97
    xray = ((xray / 255.0) ** gamma) * 255.0
    xray = np.clip(xray, 0, 255).astype(np.uint8)

    # сохранение (опционально)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        name_stem = Path(base_name).stem if base_name else "xray_result"
        cv2.imwrite(str(Path(out_dir) / f"xray_{name_stem}.png"), xray)

    return xray
