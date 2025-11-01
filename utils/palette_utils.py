# utils/palette_utils.py
from PIL import Image, ImageDraw
import os
from .logging_utils import CYAN, RESET

def show_palette(palette):
    print(f"\n{CYAN}Палитра цветов:{RESET}")
    for rgb in palette:
        r, g, b = map(int, rgb)
        print(f"\033[48;2;{r};{g};{b}m \033[0m", end=" ")
    print("\n")

def save_palette_image(palette, base_name, out_dir):
    n = len(palette)
    img = Image.new("RGB", (n * 50, 50))
    draw = ImageDraw.Draw(img)
    for i, rgb in enumerate(palette):
        draw.rectangle([i*50, 0, (i+1)*50, 50], fill=tuple(map(int, rgb)))
    path = os.path.join(out_dir, f"{base_name}_palette.png")
    img.save(path)
    return path