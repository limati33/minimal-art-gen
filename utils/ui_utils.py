# utils/ui_utils.py
import tkinter as tk
from tkinter import filedialog
from .logging_utils import log_error, CYAN, RESET

def select_images_via_dialog(multi=False):
    try:
        root = tk.Tk(); root.withdraw(); root.update()
        print(f"{CYAN}Выбор изображения...{RESET}")
        if multi:
            paths = filedialog.askopenfilenames(
                title="Выберите изображение(я)",
                filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff")]
            )
            return list(paths)
        else:
            path = filedialog.askopenfilename(title="Выберите изображение", filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff")])
            return [path] if path else []
    except Exception as e:
        log_error("Диалог", e)
        return []
    finally:
        try: root.destroy()
        except: pass