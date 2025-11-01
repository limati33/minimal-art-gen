# processor/effects/mode_17_mosaic.py
import cv2
import numpy as np
import math
import random

def rotate_pts(pts, angle, cx, cy):
    ca = math.cos(angle); sa = math.sin(angle)
    out = []
    for (x, y) in pts:
        rx = x - cx; ry = y - cy
        nx = rx * ca - ry * sa + cx
        ny = rx * sa + ry * ca + cy
        out.append((int(round(nx)), int(round(ny))))
    return out

def apply_mosaic(img, w, h, out_dir, base_name):
    hq, wq = img.shape[:2]
    larger = max(wq, hq)
    desired_cell = 8
    base = max(1, desired_cell)
    cols = int(np.ceil(wq / base))
    rows = int(np.ceil(hq / base))
    abstract = np.zeros_like(img)
    shapes = ["circle", "square", "triangle", "diamond", "pentagon"]
    grid_color = (50, 50, 50)
    for r in range(rows+1):
        y = min(hq-1, r*base)
        cv2.line(abstract, (0, y), (wq-1, y), grid_color, 1)
    for c in range(cols+1):
        x = min(wq-1, c*base)
        cv2.line(abstract, (x, 0), (x, hq-1), grid_color, 1)
    for ry in range(rows):
        for cx in range(cols):
            x0 = cx * base
            y0 = ry * base
            x1 = min(wq, (cx+1)*base)
            y1 = min(hq, (ry+1)*base)
            bw, bh = x1-x0, y1-y0
            if bw <=0 or bh <=0: continue
            block = img[y0:y1, x0:x1]
            mean_col = tuple(map(int, block.mean(axis=(0,1)).astype(int)))
            lum = 0.299*mean_col[0] + 0.587*mean_col[1] + 0.114*mean_col[2]
            delta = int(max(20, min(80, min(255, 128 - (lum - 128)))))
            if lum > 130:
                shape_col = (max(0, mean_col[0]-delta), max(0, mean_col[1]-delta), max(0, mean_col[2]-delta))
            else:
                shape_col = (min(255, mean_col[0]+delta), min(255, mean_col[1]+delta), min(255, mean_col[2]+delta))
            shape = random.choice(shapes)
            cx_px = x0 + bw//2
            cy_px = y0 + bh//2
            scale = random.uniform(1.0, 1.25)
            if shape == "circle":
                rx = int((bw/2)*scale)
                ry_ = int((bh/2)*scale)
                cv2.ellipse(abstract, (cx_px, cy_px), (max(1,rx), max(1,ry_)), 0, 0, 360, shape_col, -1)
            else:
                hw = (bw/2)*scale
                hh = (bh/2)*scale
                if shape == "square":
                    pts = [(cx_px-hw, cy_px-hh), (cx_px+hw, cy_px-hh), (cx_px+hw, cy_px+hh), (cx_px-hw, cy_px+hh)]
                elif shape == "diamond":
                    pts = [(cx_px, cy_px-hh), (cx_px+hw, cy_px), (cx_px, cy_px+hh), (cx_px-hw, cy_px)]
                elif shape == "triangle":
                    pts = [(cx_px, cy_px-hh), (cx_px-hw, cy_px+hh), (cx_px+hw, cy_px+hh)]
                elif shape == "pentagon":
                    pts = []
                    sides = 5
                    for i in range(sides):
                        ang = 2*math.pi*i/sides - math.pi/2
                        px = cx_px + math.cos(ang)*hw
                        py = cy_px + math.sin(ang)*hh
                        pts.append((px, py))
                else:
                    pts = [(cx_px-hw, cy_px-hh), (cx_px+hw, cy_px-hh), (cx_px+hw, cy_px+hh), (cx_px-hw, cy_px+hh)]
                angle = random.uniform(-0.35, 0.35)
                pts_rot = rotate_pts(pts, angle, cx_px, cy_px)
                poly = np.array(pts_rot, dtype=np.int32)
                area = abs(cv2.contourArea(poly))
                if area < 4:
                    r = max(1, int(min(bw,bh)*0.25))
                    cv2.circle(abstract, (cx_px, cy_px), r, shape_col, -1)
                else:
                    cv2.fillConvexPoly(abstract, poly, shape_col)
    if max(hq, wq) > 200:
        noise = (np.random.randn(hq, wq, 1) * 6).astype(np.int16)
        abstract = np.clip(abstract.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return abstract