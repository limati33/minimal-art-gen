import tkinter as tk
from tkinter import filedialog
from .logging_utils import log_error, CYAN, RESET

def select_images_via_dialog(multi=False):
    try:
        root = tk.Tk(); root.withdraw(); root.update()
        print(f"{CYAN}Выбор файлов...{RESET}")
        
        # Обновленный список форматов
        file_types = [
            ("Media files", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.mp4 *.avi *.mov *.mkv *.webm"),
            ("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("All files", "*.*")
        ]

        if multi:
            paths = filedialog.askopenfilenames(title="Выберите файлы", filetypes=file_types)
            return list(paths)
        else:
            path = filedialog.askopenfilename(title="Выберите файл", filetypes=file_types)
            return [path] if path else []
    except Exception as e:
        log_error("Диалог", e)
        return []
    finally:
        try: root.destroy()
        except: pass