# processor/effects/mode_5_grid.py
import numpy as np

def apply_grid(img, w, h, out_dir, base_name):
    step = max(4, int(w / 100))
    grid = img.copy()
    for y in range(0, grid.shape[0], step):
        for x in range(0, grid.shape[1], step):
            if ((x//step) + (y//step)) % 2 == 0:
                grid[y:y+step, x:x+step] = (grid[y:y+step, x:x+step] * 0.88).astype(np.uint8)
    return grid