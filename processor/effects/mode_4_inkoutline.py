# processor/effects/mode_4_inkoutline
import cv2

def apply_inkoutline(img, w, h, out_dir, base_name, **kwargs):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 50, 120)

    edges = cv2.dilate(edges, None)
    edges_inv = 255 - edges

    base = cv2.bilateralFilter(img, 9, 75, 75)
    base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    ink = cv2.min(base_gray, edges_inv)
    return cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)