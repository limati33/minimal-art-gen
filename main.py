import os
import sys
import time
from utils.logging_utils import log_error
from utils.file_utils import resolve_shortcut
from utils.input_utils import ask_int, ask_float, parse_int_list, parse_mode_list
from utils.ui_utils import select_images_via_dialog
from utils.effects_table import show_effects_table
from processor.single_image import process_single
from processor.video import process_video
from processor.effects import EFFECTS
from colorama import Fore, Style

# Цвета
RESET = Style.RESET_ALL
BOLD = Style.BRIGHT
CYAN = Fore.CYAN
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
RED = Fore.RED
MAGENTA = Fore.MAGENTA
BLUE = Fore.BLUE

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}

def is_video(path):
    _, ext = os.path.splitext(path)
    return ext.lower() in VIDEO_EXTENSIONS


def main():
    print(f"\n{BOLD}{BLUE}МИНИМАЛИСТИЧНЫЙ АРТ-ГЕНЕРАТОР (ФОТО + ВИДЕО){RESET}")
    print(f"{MAGENTA}{'=' * 40}{RESET}")
    show_effects_table()
    
    args = sys.argv[1:]
    input_paths = []
    
    # --- Сбор файлов ---
    if args:
        for a in args:
            resolved = resolve_shortcut(a)
            if os.path.isfile(resolved):
                input_paths.append(resolved)
            else:
                print(f"{YELLOW}Внимание: '{a}' → не найден или не файл — пропущен.{RESET}")
    else:
        raw_paths = select_images_via_dialog(multi=True)
        for p in raw_paths:
            resolved = resolve_shortcut(p)
            if os.path.isfile(resolved):
                input_paths.append(resolved)
            else:
                print(f"{YELLOW}Пропущен: {p}{RESET}")
    
    if not input_paths:
        print(f"{RED}Файлы не выбраны. Выход.{RESET}")
        return
    
    print(f"{CYAN}Выбрано:{RESET} {len(input_paths)} файл(ов)")
    for i, p in enumerate(input_paths, 1):
        print(f"  {i}. {os.path.basename(p)}")
    
    # --- Ввод параметров ---
    ncolors_input = input(f"{YELLOW}Количество цветов (2–512 или 'all' или 'no limit'): {RESET}").strip()
    n_colors_list = parse_int_list(ncolors_input, 2, 512)

    # === FIX: Обработка режима "no limit" ===
    if n_colors_list is None:
        print("Режим: без ограничения количества цветов")
        n_colors_iter = [None]     # ← FIX (делаем итератор)
        n_colors_count = 1         # ← FIX
    elif not n_colors_list:
        return
    else:
        print("Будут использованы цвета:", n_colors_list)
        n_colors_iter = n_colors_list
        n_colors_count = len(n_colors_list)
    # =========================================

    scale = ask_float("Масштаб (1 = оригинал, 0.1–2.0): ", 0.1, 2.0)
    
    blur_input = input(f"{YELLOW}Размытие (0–5) [Видео использует 1 значение]: {RESET}").strip()
    blur_list = parse_int_list(blur_input, 0, 5)
    if not blur_list: return
    
    max_mode = max(EFFECTS.keys())
    modes_input = input(f"{YELLOW}Тип эффекта (1–{max_mode} или 'all', можно комбинации '9+12'): {RESET}").strip()
    modes_list = parse_mode_list(modes_input, 1, max_mode)
    if not modes_list: return
    
    # --- Запуск обработки ---
    start_time = time.time()
    
    for idx_path, path in enumerate(input_paths, 1):
        print(f"\n{BOLD}{BLUE}>>> Файл {idx_path}/{len(input_paths)}: {os.path.basename(path)}{RESET}")
        
        if is_video(path):

            # === FIX: корректный подсчёт задач ===
            total_v_tasks = n_colors_count * len(blur_list) * len(modes_list)
            # =====================================

            current_v_task = 0

            print(f"{MAGENTA}Режим ВИДЕО: Запланировано {total_v_tasks} вариант(а/ов).{RESET}")
            print(f"{YELLOW}ВНИМАНИЕ: Обработка видео долгая. Наберитесь терпения.{RESET}")

            for n_colors in n_colors_iter:   # ← FIX
                for blur_strength in blur_list:
                    for mode in modes_list:
                        current_v_task += 1
                        print(f"\n{BOLD}{CYAN}>>> Видео-вариант {current_v_task}/{total_v_tasks}{RESET}")
                        print(f"Параметры: Цветов={n_colors}, Размытие={blur_strength}, Эффект={mode}")
                        
                        try:
                            process_video(path, n_colors, scale, blur_strength, mode)
                        except Exception as e:
                            print(f"{RED}Ошибка видео: {e}{RESET}")
                            log_error("Ошибка видео", e, path, n_colors, blur_strength, mode)
                
        else:
            # Фото
            # === FIX: корректный подсчёт задач ===
            local_tasks = n_colors_count * len(blur_list) * len(modes_list)
            # ======================================
            completed_local = 0
            
            for n_colors in n_colors_iter:   # ← FIX
                for blur_strength in blur_list:
                    for mode in modes_list:
                        completed_local += 1
                        print(f"\n{MAGENTA}> Фото-арт {completed_local}/{local_tasks}: C={n_colors} B={blur_strength} M={mode}{RESET}")
                        try:
                            process_single(path, n_colors, scale, blur_strength, mode)
                        except Exception as e:
                            print(f"{RED}[Ошибка] {e}{RESET}")
                            log_error("Ошибка фото", e, path, n_colors, blur_strength, mode)

    total_time = time.time() - start_time
    mins, secs = divmod(int(total_time), 60)
    print(f"\n\n{GREEN}ВСЕ ЗАДАЧИ ВЫПОЛНЕНЫ!{RESET} ({mins} мин {secs} сек)")


if __name__ == "__main__":
    main()
