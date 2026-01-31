# processor/effects/mode_29_filter.py
import os
import re
import cv2
import numpy as np

# ----------------------------
# ВНУТРЕННЕЕ СОСТОЯНИЕ ЭФФЕКТА (для видео)
# ----------------------------
_STATE = {
    "centers": None,          # все найденные центры (np.ndarray shape=(k,2))
    "active_centers": None,   # отобранные центры для видео (<= max_variants)
    "initialized": False,     # true после init для видео
    "sample_frame_used": None # сохранённый кадр, на котором считались площади
}


def reset_state():
    """Сброс внутреннего состояния — вызывать перед началом обработки нового видео."""
    _STATE["centers"] = None
    _STATE["active_centers"] = None
    _STATE["initialized"] = False
    _STATE["sample_frame_used"] = None


# ----------------------------
# ВСПОМОГАТЕЛИ
# ----------------------------
def _is_probably_video(base_name):
    """Небольшая эвристика: base_name вроде 'frame_123' -> video."""
    if not base_name:
        return False
    return re.search(r'frame[_\-]?\d+', base_name) is not None


def _compute_centers_on_frame(img, n_colors, min_saturation, min_value, max_sample=100_000):
    """
    Запускает kmeans на image и возвращает centers (в Lab a,b).
    Возвращает None если нет достаточного количества цветных пикселей.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    valid_mask = cv2.inRange(
        hsv,
        np.array([0, min_saturation, min_value], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8)
    )

    if cv2.countNonZero(valid_mask) < (img.shape[0] * img.shape[1] * 0.005):
        return None

    ab = lab[:, :, 1:3]
    masked_ab = ab[valid_mask > 0]

    if masked_ab.shape[0] == 0:
        return None

    if masked_ab.shape[0] > max_sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(masked_ab.shape[0], size=max_sample, replace=False)
        masked_ab = masked_ab[idx]

    data = np.float32(masked_ab)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    try:
        _, _, centers = cv2.kmeans(data, n_colors, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    except Exception as e:
        # На случай некорректного входа — вернуть None
        return None

    return centers  # shape (n_colors, 2) — a,b


def _area_for_center_on_frame(center, lab_ab, valid_mask, color_threshold):
    """
    Вычисляет «площадь» маски для данного центра на заданном кадре.
    lab_ab — (h,w,2) float32
    valid_mask — uint8 mask
    """
    dist = np.sqrt(np.sum((lab_ab - center) ** 2, axis=2))
    mask = np.clip((color_threshold - dist) / 5.0, 0.0, 1.0)
    mask[valid_mask == 0] = 0.0
    area = float(np.sum(mask)) / (lab_ab.shape[0] * lab_ab.shape[1])
    return area


def _lab_color_to_hex(center_ab):
    """Побочная: получить hex (rgb) приблизительного цвета из LAB (для имен файлов)."""
    p_lab = np.uint8([[[140, int(center_ab[0]), int(center_ab[1])]]])
    p_bgr = cv2.cvtColor(p_lab, cv2.COLOR_LAB2BGR)[0, 0]
    return f"{int(p_bgr[2]):02x}{int(p_bgr[1]):02x}{int(p_bgr[0]):02x}"


# ----------------------------
# ГЛАВНАЯ ФУНКЦИЯ
# ----------------------------
def apply_filter(
    img,
    w=None, h=None,
    out_dir=None, base_name=None,

    # Параметры эффекта (можно менять)
    n_colors=10,
    min_saturation=80,
    min_value=50,
    color_threshold=15,
    min_area_ratio=0.005,
    blur_mask=True,

    # Видео/фото поведение
    video_mode=None,        # None = определяем автоматически по base_name, True/False = принудительно
    max_variants=None,      # None = без лимита (фото), integer = максимум вариантов (видео: <=5)
    save_debug=False,       # Сохранять PNG для каждого варианта (когда применимо)
    return_multiple=False   # Для фото: вернуть список вариантов; для видео+True: вернуть list (до max_variants)
):
    """
    Универсальный фильтр для фото и видео.

    Возвращаемое значение:
      - Фото (video_mode=False): список numpy.ndarray с вариантами (можно выбрать первый)
      - Видео (video_mode=True, return_multiple=False): единый numpy.ndarray (лучший вариант)
      - Видео (video_mode=True, return_multiple=True): список (до max_variants) numpy.ndarray

    Совместимость: process_video ожидает, что функция принимает (img, w, h, out_dir, base_name).
    Поэтому video_mode и проч. можно не передавать — видео будет распознано по base_name.
    """
    if img is None:
        return None

    # Resize если переданы w,h
    img_h, img_w = img.shape[:2]
    if w and h and (img_w != w or img_h != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
        img_h, img_w = img.shape[:2]

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Детект видео или использовать переданный флаг
    if video_mode is None:
        video_mode = _is_probably_video(base_name)

    # Ограничение max_variants для видео (по условию — не более 5)
    if video_mode:
        if max_variants is None:
            max_variants = 5
        else:
            max_variants = min(5, int(max_variants))

    # Подготовка базовых матриц
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    valid_mask = cv2.inRange(
        hsv,
        np.array([0, min_saturation, min_value], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8)
    )

    gray_3ch = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_ab = lab[:, :, 1:3].astype(np.float32)
    img_f = img.astype(np.float32)
    gray_f = gray_3ch.astype(np.float32)

    # ----------------------------
    # ИНИЦИАЛИЗАЦИЯ ДЛЯ ВИДЕО (один раз)
    # ----------------------------
    if video_mode:
        if not _STATE["initialized"]:
            centers = _compute_centers_on_frame(img, n_colors, min_saturation, min_value)
            if centers is None:
                # ничего интересного — возвращаем серый кадр/вариант
                if return_multiple and not video_mode:
                    return [gray_3ch]
                return gray_3ch
            # вычисляем площади для каждого центра на этом кадре
            stats = []
            for c in centers:
                area = _area_for_center_on_frame(c, lab_ab, valid_mask, color_threshold)
                stats.append((area, c))
            # сортируем по убыванию площади
            stats.sort(key=lambda x: x[0], reverse=True)
            sorted_centers = [c for a, c in stats if a >= min_area_ratio]
            if not sorted_centers:
                # ничего значимого — вернём серый
                if return_multiple and not video_mode:
                    return [gray_3ch]
                return gray_3ch

            _STATE["centers"] = np.array(sorted_centers, dtype=np.float32)
            # active_centers — top N (max_variants) или все
            if max_variants is not None:
                _STATE["active_centers"] = _STATE["centers"][:max_variants].copy()
            else:
                _STATE["active_centers"] = _STATE["centers"].copy()
            _STATE["sample_frame_used"] = img.copy()
            _STATE["initialized"] = True

        centers_to_use = _STATE["active_centers"]
    else:
        # Фото: один раз считаем центры на текущем изображении
        centers = _compute_centers_on_frame(img, n_colors, min_saturation, min_value)
        if centers is None:
            return [gray_3ch] if return_multiple else gray_3ch
        centers_to_use = centers

    # ----------------------------
    # ПРИМЕНЕНИЕ: создаём варианты для каждого центра в centers_to_use
    # ----------------------------
    results = []  # список (area, res_img, center)
    for center in centers_to_use:
        # дистанция и маска
        dist = np.sqrt(np.sum((lab_ab - center) ** 2, axis=2))
        mask = np.clip((color_threshold - dist) / 5.0, 0.0, 1.0)
        mask[valid_mask == 0] = 0.0

        area = float(np.sum(mask)) / (img_w * img_h)
        if area < min_area_ratio:
            continue

        if blur_mask:
            mask = cv2.GaussianBlur(mask, (3, 3), 0)

        res = (img_f * mask[:, :, None] + gray_f * (1.0 - mask[:, :, None])).astype(np.uint8)

        results.append((area, res, center))

    if not results:
        return [gray_3ch] if return_multiple and not video_mode else (gray_3ch)

    # сортируем по площади (убывание)
    results.sort(key=lambda x: x[0], reverse=True)

    # Для фото — потенциально сохранить **все** варианты (или ограничить max_variants если передан)
    if not video_mode:
        # Если max_variants задан для фото — уважать его
        if max_variants is not None:
            results = results[:max_variants]
        # Сохранение debug-артов при желании
        if save_debug and out_dir and base_name:
            for _, res, center in results:
                hex_c = _lab_color_to_hex(center)
                out_path = os.path.join(out_dir, f"{base_name}_filter_{hex_c}.png")
                cv2.imwrite(out_path, res)
        # Возвращаем список изображений (варианты)
        imgs = [res for _, res, _ in results]
        return imgs if return_multiple or len(imgs) > 1 else imgs

    # --- Видео: вернуть один лучший или список до max_variants
    if video_mode:
        if return_multiple:
            imgs = [res for _, res, _ in results]
            # гарантируем длину <= max_variants (если был задан)
            if max_variants is not None:
                imgs = imgs[:max_variants]
            # опционально сохраняем дебаг-кадры (редко для видео)
            if save_debug and out_dir and base_name:
                for idx, res in enumerate(imgs):
                    center = results[idx][2]
                    hex_c = _lab_color_to_hex(center)
                    out_path = os.path.join(out_dir, f"{base_name}_filter_{hex_c}.png")
                    cv2.imwrite(out_path, res)
            return imgs
        else:
            # возвращаем лучший вариант (первый в списке)
            best_res = results[0][1]
            # при save_debug можем сохранять один кадр для контроля
            if save_debug and out_dir and base_name:
                center = results[0][2]
                hex_c = _lab_color_to_hex(center)
                out_path = os.path.join(out_dir, f"{base_name}_filter_{hex_c}.png")
                cv2.imwrite(out_path, best_res)
            return best_res


# ----------------------------
# НЕМНОГО ПРИМЕЧАНИЙ:
# - Для видео: вызови reset_state() перед началом обработки (process_video может сделать это).
# - Если process_video всё ещё вызывает эффект как fn(img,w,h,out_dir,base_name),
#   то видео будет распознано по base_name ('frame_123') автоматически.
# - Если хочешь получить до 5 видео-версий, в process_video нужно:
#     * на инициализации вызвать reset_state()
#     * при каждом кадре вызывать apply_filter(..., video_mode=True, return_multiple=True, max_variants=5)
#     * и писать каждый список-элемент в свой VideoWriter (или создавать writers динамически)
# - Для фото: вызов apply_filter(img, ...) вернёт список вариантов (по всем центрам) — их можно сохранить/показать.
# ----------------------------
