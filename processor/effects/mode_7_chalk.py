# processor/effects/mode_7_chalk.py
import cv2
import numpy as np

def apply_chalk(img, w, h, out_dir, base_name):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 130)
    chalk = cv2.convertScaleAbs(img, alpha=0.95, beta=-10)
    grain = (np.random.normal(0, 18, img.shape)).astype(np.int16)
    chalk_g = np.clip(chalk.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    chalk_g = cv2.GaussianBlur(chalk_g, (3,3), 0)
    edges3 = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    mask_dark = (edges3[:,:,None] > 0).astype(np.uint8) * 30
    return np.clip(chalk_g.astype(np.int16) - mask_dark, 0, 255).astype(np.uint8)