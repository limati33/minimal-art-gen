import cv2
import numpy as np

def apply_printer(img, w=None, h=None, out_dir=None, base_name=None, **kwargs):
    if img is None: return None
    if w and h: img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    h_img, w_img = img.shape[:2]

    dot_size = int(kwargs.get("dot_size", 5))
    shift_px = max(1, w_img // 640)  # пропорционально ширине

    b, g, r = cv2.split(img)
    M = np.float32([[1, 0, shift_px], [0, 1, 0]])
    r = cv2.warpAffine(r, M, (w_img, h_img))
    img_shifted = cv2.merge([b, g, r])

    gray = cv2.cvtColor(img_shifted, cv2.COLOR_BGR2GRAY)
    output = np.full_like(img, 245)

    for y in range(0, h_img, dot_size):
        for x in range(0, w_img, dot_size):
            roi = gray[y:y+dot_size, x:x+dot_size]
            mean_v = np.mean(roi)
            radius = int((1.0 - mean_v / 255.0) * (dot_size / 2) * 1.3)
            if radius > 0:
                color = img_shifted[y, x].tolist()
                cv2.circle(output, (x + dot_size//2, y + dot_size//2), radius, color, -1, cv2.LINE_AA)

    paper_noise = np.random.normal(0, 5, (h_img, w_img, 3)).astype(np.int16)
    output = np.clip(output.astype(np.int16) + paper_noise, 0, 255).astype(np.uint8)

    return output
