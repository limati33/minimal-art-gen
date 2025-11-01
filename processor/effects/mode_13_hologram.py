# processor/effects/mode_13_hologram.py
import cv2
import numpy as np
from utils.input_utils import print_progress
from utils.logging_utils import CYAN, RESET, log_error

def apply_hologram(img, w, h, out_dir, base_name, image_path=None, n_colors=None, blur_strength=None, mode=None):
    print_progress(3, prefix=f"{CYAN}Голограмма...{RESET} ")
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue_noise = np.random.randn(h, w) * 15
        hsv[..., 0] = (hsv[..., 0] + hue_noise * 2) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(1.1, 1.4, (h, w)), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * np.random.uniform(1.0, 1.25, (h, w)), 0, 255)
        base_frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        depth = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        depth = cv2.GaussianBlur(depth, (0, 0), 15)
        offset = 8
        rainbow_base = cv2.applyColorMap(np.linspace(0, 255, w, dtype=np.uint8), cv2.COLORMAP_HSV)
        rainbow_base = cv2.cvtColor(rainbow_base, cv2.COLOR_BGR2RGB)

        cols = np.arange(w)[None, :]
        rows = np.arange(h)[:, None]
        base_shift = ((depth[:, w // 2] * offset * 2 - offset).astype(np.int32))[:, None]
        phase = 0.0
        hue_shift = np.sin(rows / 20 + phase) * 15
        hsv_mod = hsv.copy()
        hsv_mod[..., 0] = (hsv_mod[..., 0] + hue_shift) % 180
        frame_color = cv2.cvtColor(hsv_mod.astype(np.uint8), cv2.COLOR_HSV2RGB)

        shift = base_shift + (np.sin(phase + rows / 30) * offset).astype(np.int32)[:, None]
        nx = np.empty((h, w), dtype=np.intp)
        for y in range(h):
            nx[y, :] = np.clip(cols + shift[y, 0], 0, w - 1)
        displaced = np.empty_like(frame_color)
        for c in range(3):
            displaced[..., c] = np.take_along_axis(frame_color[..., c], nx, axis=1)

        rainbow = np.roll(rainbow_base, int(5), axis=1)
        if displaced.shape != base_frame.shape:
            base_frame = cv2.resize(base_frame, (displaced.shape[1], displaced.shape[0]), interpolation=cv2.INTER_AREA)
        if displaced.shape != rainbow.shape:
            rainbow = cv2.resize(rainbow, (displaced.shape[1], displaced.shape[0]), interpolation=cv2.INTER_AREA)

        frame = cv2.addWeighted(displaced, 0.65, base_frame, 0.25, 0)
        frame = cv2.addWeighted(frame, 0.8, rainbow, 0.4, 0)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        bright = (gray > 180).astype(np.uint8) * 255
        bright = cv2.GaussianBlur(bright, (13, 13), 0)
        glow = np.full_like(frame, (210, 230, 255))
        frame = cv2.addWeighted(frame, 1.0, glow, 0.12, 0)
        frame = np.where(bright[..., None], np.clip(frame + 25, 0, 255), frame)

        noise = np.random.normal(0, 2.5, frame.shape).astype(np.int16)
        return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    except Exception as e:
        log_error("Ошибка при генерации голограммы", e, image_path, n_colors, blur_strength, mode)
        return img.copy()