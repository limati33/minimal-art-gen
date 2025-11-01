# processor/effects/mode_11_chrome.py
import cv2

def apply_chrome(img, w, h, out_dir, base_name):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    grad = cv2.convertScaleAbs(grad)
    combined = cv2.addWeighted(gray, 0.6, grad, 0.8, 0)
    combined = cv2.equalizeHist(combined)
    chrome = cv2.applyColorMap(combined, cv2.COLORMAP_WINTER)
    return cv2.addWeighted(chrome, 0.95, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.05, 0)