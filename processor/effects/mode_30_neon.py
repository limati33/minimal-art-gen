# processor/effects/mode_30_neon.py
import cv2
import numpy as np
import os

def apply_neon(img, w=None, h=None, out_dir=None, base_name=None):
    """
    Mode 30: Разноцветный неоновый градиентный контур.
    Контуры берут цвет из оригинального изображения, создавая переходы.
    Сама линия остаётся читаемой (ядро) + вокруг неё мягкое свечение (glow).
    """
    if img is None: 
        return None
    
    # 1. Ресайз
    img_h, img_w = img.shape[:2]
    if w and h and (img_w != w or img_h != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    # 2. Подготовка источника цвета (используем медианный фильтр)
    #    можно экспериментировать с bilateralFilter для более "чистых" цветов
    color_source = cv2.medianBlur(img, 7)

    # 3. Обнаружение резких границ (Canny)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred_gray, 50, 150)

    # ---------------- Параметры (подбирайте при необходимости) ----------------
    core_kernel = (2, 2)        # ядро для тонкого, но чёткого ядра линии
    soft_kernel = (5, 5)        # для более широкой мягкой маски перехода
    soft_blur_k = (5, 5)        # блюр маски для alpha-перехода (должен быть нечётным лучше)
    glow_k_small = (11, 11)     # маленькое свечение
    glow_k_large = (25, 25)     # большое свечение
    glow_strength_small = 0.5
    glow_strength_large = 0.5
    glow_overall_mul = 0.6      # уменьшает интенсивность свечения
    soft_weight = 0.5           # вклад мягкой цветной заливки
    core_weight = 1.0           # вклад чёткого ядра
    unsharp_amount = 0.5        # сколько вернуть резкости (0..1)
    # -------------------------------------------------------------------------

    # 1) Чёткое тонкое "ядро" — чтобы сохранить читаемость линий
    kernel_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, core_kernel)
    edges_core = cv2.dilate(edges, kernel_core, iterations=1)
    colored_core = cv2.bitwise_and(color_source, color_source, mask=edges_core)

    # 2) Мягкая маска — для плавного перехода (не заменяет ядро)
    kernel_soft = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, soft_kernel)
    edges_soft = cv2.dilate(edges, kernel_soft, iterations=1)
    edges_soft = cv2.GaussianBlur(edges_soft, soft_blur_k, 0)

    alpha = edges_soft.astype(np.float32) / 255.0
    alpha = cv2.merge([alpha, alpha, alpha])
    colored_soft = (color_source.astype(np.float32) * alpha).astype(np.uint8)
    # Небольшой локальный blur — чтобы убрать пиксельность в мягкой заливке
    colored_soft = cv2.GaussianBlur(colored_soft, (3, 3), 0)

    # 3) Свечение — делаем из мягкой заливки, но аккуратно (меньше размер и сила)
    glow_layer_1 = cv2.GaussianBlur(colored_soft, glow_k_small, 0)
    glow_layer_2 = cv2.GaussianBlur(colored_soft, glow_k_large, 0)
    total_glow = cv2.addWeighted(glow_layer_1, glow_strength_small,
                                 glow_layer_2, glow_strength_large, 0)
    total_glow = (total_glow * glow_overall_mul).astype(np.uint8)

    # 4) Композит: сначала смешиваем ядро + лёгкая мягкая заливка, затем добавляем свечение
    base = cv2.addWeighted(colored_core, core_weight, colored_soft, soft_weight, 0)
    final_result = cv2.addWeighted(base, 1.0, total_glow, 1.0, 0)

    # 5) Лёгкая "обратная" резкость (unsharp mask) — возвращаем читаемость без грубых краёв
    gauss = cv2.GaussianBlur(final_result, (3, 3), 0)
    final_result = cv2.addWeighted(final_result.astype(np.float32), 1.0 + unsharp_amount,
                                   gauss.astype(np.float32), -unsharp_amount, 0)
    final_result = np.clip(final_result, 0, 255).astype(np.uint8)

    # 6. Сохранение
    if out_dir and base_name:
        out_path = os.path.join(out_dir, f"{base_name}_mode30_gradient_neon.png")
        cv2.imwrite(out_path, final_result)
        print(f"  [Neon Gradient] Эффект сохранен -> {os.path.basename(out_path)}")

    return final_result
