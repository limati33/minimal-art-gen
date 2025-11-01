# processor/effects/mode_8_ascii.py
import cv2
import numpy as np
from PIL import Image
import os

def apply_ascii(img, w, h, out_dir, base_name):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ascii_chars = np.array(list("@%#*+=-:. "))[::-1]
    cell_w, cell_h = 7, 14
    new_w, new_h = w // cell_w, h // cell_h
    if new_w == 0 or new_h == 0:
        return img

    small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    idx = (small.astype(np.float32) / 255 * (len(ascii_chars) - 1)).astype(np.int32)
    idx = np.clip(idx, 0, len(ascii_chars) - 1)

    color_ascii = np.full((h, w, 3), 255, dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            ch = ascii_chars[idx[i, j]]
            orig_rgb = img[i * cell_h, j * cell_w]
            color_bgr = (int(orig_rgb[2]), int(orig_rgb[1]), int(orig_rgb[0]))
            x = j * cell_w + 1
            y = i * cell_h + cell_h - 3
            cv2.putText(color_ascii, ch, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1, cv2.LINE_AA)

    bw_ascii = np.full((h, w, 3), 255, dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            ch = ascii_chars[idx[i, j]]
            brightness = int(small[i, j])
            color_bgr = (brightness, brightness, brightness)
            x = j * cell_w + 1
            y = i * cell_h + cell_h - 3
            cv2.putText(bw_ascii, ch, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1, cv2.LINE_AA)

    bw_path = os.path.join(out_dir, f"{base_name}_ascii_bw.png")
    Image.fromarray(bw_ascii).save(bw_path)
    print(f" ASCII (ЧБ): {bw_path}")

    txt_path = os.path.join(out_dir, f"{base_name}_ascii.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for i in range(new_h):
            line = ''.join(ascii_chars[idx[i, j]] for j in range(new_w))
            f.write(line + "\n")
    print(f" ASCII-текст: {txt_path}")

    return color_ascii