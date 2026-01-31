import cv2
import numpy as np
import re
from functools import lru_cache

# Кэши для карт / масок по размерам — чтобы не пересоздавать их для каждого кадра
_MAP_CACHE = {}
_MASK_CACHE = {}

def _extract_frame_idx_from_basename(base_name):
    """Попытка извлечь индекс кадра из base_name вида 'frame_123'"""
    if not base_name:
        return 0
    m = re.search(r'frame[_\-]?(\d+)', base_name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0

def _get_maps(w, h):
    # Возвращаем (base_map_x, base_map_y) из кэша или создаём
    key = (w, h)
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    _MAP_CACHE[key] = (map_x, map_y)
    return map_x, map_y

def _get_radial_mask(w, h):
    key = (w, h)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (w//2, h//2), int(np.sqrt(w*w + h*h)//2), 1, -1)
    mask = cv2.GaussianBlur(mask, (max(1, (w//3)|1), max(1, (w//3)|1)), 0)
    _MASK_CACHE[key] = mask
    return mask

def apply_underwater(img, w=None, h=None, out_dir=None, base_name=None):
    """
    Анимируемая версия "Abyss 2.0".
    - Если base_name содержит 'frame_<N>', используется N как индекс кадра (фаза).
    - Подходит в текущий pipeline: get_effect возвращает эту функцию и process_video
      вызывает её как fn(img, w, h, out_dir, base_name).
    """
    if img is None:
        return None

    # 1) Resize как раньше
    img_h, img_w = img.shape[:2]
    if w and h and (img_w != w or img_h != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    # Получаем индекс кадра (если есть) и фазу времени
    frame_idx = _extract_frame_idx_from_basename(base_name)
    # скорость анимации — регулируй множитель (0.12 — плавно, 0.3 — быстрее)
    phase = frame_idx * 0.12

    # 2) ПЛАСТИЧЕСКАЯ ДЕФОРМАЦИЯ (с анимацией)
    base_map_x, base_map_y = _get_maps(w, h)

    # смещения зависят от координат и фазы — так волны будут двигаться
    shift_x = (8.0 * np.sin(2 * np.pi * base_map_y / 150.0 + phase)
               + 4.0 * np.sin(2 * np.pi * base_map_y / 70.0 + phase * 1.4)
               + 2.0 * np.sin(2 * np.pi * base_map_x / 400.0 + phase * 0.6))
    shift_y = (5.0 * np.cos(2 * np.pi * base_map_x / 200.0 + phase * 1.1)
               + 2.0 * np.sin(2 * np.pi * base_map_y / 300.0 + phase * 0.9))

    map_x = (base_map_x + shift_x).astype(np.float32)
    map_y = (base_map_y + shift_y).astype(np.float32)

    submerged = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    # 3) Динамическое цветокорректирование (слегка «дышит»)
    submerged = submerged.astype(np.float32)
    # небольшая синусоидальная вариация интенсивностей
    blue_boost = 1.15 + 0.05 * np.sin(phase * 1.3)
    green_mul = 0.82 + 0.03 * np.sin(phase * 1.1 + 1.0)
    red_mul = 0.38 + 0.02 * np.sin(phase * 0.9 + 2.0)
    submerged[:, :, 0] *= blue_boost   # B
    submerged[:, :, 1] *= green_mul   # G
    submerged[:, :, 2] *= red_mul     # R (гасим)

    submerged = np.clip(submerged, 0, 255).astype(np.uint8)

    # 4) Динамическая, плавная каустика (без резкого шума)
    # Используем синусы по координатам + фазу — получается плавно меняющийся рисунок
    # Создаём паттерн на float, затем блюрим и нормализуем
    # NOTE: вычисления делаем на базе base_map_x/base_map_y (float32)
    caustic_pattern = (np.sin((base_map_x * 0.018 + base_map_y * 0.022) + phase * 2.2)
                       + 0.5 * np.sin((base_map_x * 0.035 - base_map_y * 0.012) + phase * 1.6))
    # Нормируем в 0..1
    caustic_norm = (caustic_pattern - caustic_pattern.min()) / (np.ptp(caustic_pattern) + 1e-8)
    caustic = (caustic_norm * 255).astype(np.uint8)

    # Блюрим, уменьшаем амплитуду (чтобы не было ярких пятен)
    k = max(3, (w // 20) | 1)
    caustic = cv2.GaussianBlur(caustic, (k, k), 0)
    caustic = cv2.normalize(caustic, None, 0, 50, cv2.NORM_MINMAX)

    # Добавляем каустику в светлые участки: суммируем B and G, R оставляем меньше
    b_add = caustic
    g_add = (caustic * 0.9).astype(np.uint8)
    r_add = (caustic * 0.1).astype(np.uint8)
    submerged = cv2.add(submerged, cv2.merge([b_add, g_add, r_add]))

    # 5) Радиационный туман (глубина) — центр чуть светлее, края темнее
    mask = _get_radial_mask(w, h)
    submerged = (submerged.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)

    return submerged
