# processor/effects/__init__.py
from .mode_1_poster import apply_poster
from .mode_2_smooth import apply_smooth
from .mode_3_comic import apply_comic
from .mode_4_paper import apply_paper
from .mode_5_grid import apply_grid
from .mode_6_dust import apply_dust
from .mode_7_chalk import apply_chalk
from .mode_8_ascii import apply_ascii
from .mode_9_plastic import apply_plastic
from .mode_10_neon import apply_neon
from .mode_11_chrome import apply_chrome
from .mode_12_candle import apply_candle
from .mode_13_hologram import apply_hologram
from .mode_14_watercolor import apply_watercolor
from .mode_15_blueprint import apply_blueprint
from .mode_16_map import apply_map
from .mode_17_mosaic import apply_mosaic
from .mode_18_dream import apply_dream
from .mode_19_pixel import apply_pixel
from .mode_20_fire import apply_fire


EFFECTS = {
    1: apply_poster,
    2: apply_smooth,
    3: apply_comic,
    4: apply_paper,
    5: apply_grid,
    6: apply_dust,
    7: apply_chalk,
    8: apply_ascii,
    9: apply_plastic,
    10: apply_neon,
    11: apply_chrome,
    12: apply_candle,
    13: apply_hologram,
    14: apply_watercolor,
    15: apply_blueprint,
    16: apply_map,
    17: apply_mosaic,
    18: apply_dream,
    19: apply_pixel,
    20: apply_fire,
}


def get_effect(mode):
    """Возвращает функцию эффекта по номеру, fallback — оригинал"""
    return EFFECTS.get(mode, lambda img, w, h, out_dir, base_name: img)