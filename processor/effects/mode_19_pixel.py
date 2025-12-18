# processor/effects/mode_19_pixel.py
import cv2

def apply_pixel(img, w, h, out_dir, base_name, pixel_size=6):
    pixel_size = max(1, int(pixel_size))

    small_w = max(1, img.shape[1] // pixel_size)
    small_h = max(1, img.shape[0] // pixel_size)

    small = cv2.resize(
        img,
        (small_w, small_h),
        interpolation=cv2.INTER_NEAREST
    )

    return cv2.resize(
        small,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
