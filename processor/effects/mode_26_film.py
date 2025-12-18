import cv2
import numpy as np

def apply_film(img, w=None, h=None, out_dir=None, base_name=None, **kwargs):
    if img is None: return None
    if w and h: img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    h_img, w_img = img.shape[:2]

    # Теплое тонирование
    img_f = img.astype(np.float32)
    img_f[:, :, 0] *= 0.88  # Blue вниз
    img_f[:, :, 2] *= 1.05  # Red вверх
    img = np.clip(img_f, 0, 255).astype(np.uint8)

    # Виньетка
    kernel_x = cv2.getGaussianKernel(w_img, w_img/1.5)
    kernel_y = cv2.getGaussianKernel(h_img, h_img/1.5)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    vignette = 0.4 + 0.6 * mask
    img = (img * vignette[:, :, np.newaxis]).astype(np.uint8)

    # Film Grain
    noise_sigma = max(1, min(20, w_img / 100))  # адаптивно по размеру
    noise = np.random.normal(0, noise_sigma, (h_img, w_img, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Bloom
    bloom_sigma = max(1.0, w_img / 500)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=bloom_sigma)
    img = cv2.addWeighted(img, 0.85, blur, 0.25, 0)

    return img
