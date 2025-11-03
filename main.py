import os
import sys
import time
from utils.logging_utils import log_error
from utils.file_utils import resolve_shortcut
from utils.input_utils import ask_int, ask_float, parse_int_list
from utils.ui_utils import select_images_via_dialog
from utils.effects_table import show_effects_table
from processor.single_image import process_single
from colorama import Fore, Style
from processor.effects import EFFECTS

# Цвета
RESET = Style.RESET_ALL
BOLD = Style.BRIGHT
CYAN = Fore.CYAN
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
RED = Fore.RED
MAGENTA = Fore.MAGENTA
BLUE = Fore.BLUE

def main():
    print(f"\n{BOLD}{BLUE}МИНИМАЛИСТИЧНЫЙ АРТ-ГЕНЕРАТОР{RESET}")
    print(f"{MAGENTA}{'=' * 40}{RESET}")
    show_effects_table()
    
    args = sys.argv[1:]
    image_paths = []
    if args:
        for a in args:
            resolved = resolve_shortcut(a)
            if os.path.isfile(resolved):
                image_paths.append(resolved)
            else:
                print(f"{YELLOW}Внимание: '{a}' → не найден или не файл — пропущен.{RESET}")
        if not image_paths:
            print(f"{RED}Нет корректных файлов в аргументах.{RESET}")
            return
    else:
        raw_paths = select_images_via_dialog(multi=True)
        for p in raw_paths:
            resolved = resolve_shortcut(p)
            if os.path.isfile(resolved):
                image_paths.append(resolved)
            else:
                print(f"{YELLOW}Пропущен (не файл): {p}{RESET}")
    
    if not image_paths:
        print(f"{RED}Файлы не выбраны. Выход.{RESET}")
        return
    
    print(f"{CYAN}Выбрано:{RESET} {', '.join(os.path.basename(p) for p in image_paths)}")
    
    ncolors_input = input(f"{YELLOW}Количество цветов (2–128), например: 4,8,12 или 4-12 или (all): {RESET}").strip()
    n_colors_list = parse_int_list(ncolors_input, 2, 128)
    if not n_colors_list:
        print(f"{RED}Ни одного корректного значения для 'Количество цветов'. Выход.{RESET}")
        return
    
    scale = ask_float("Масштаб (1 = оригинал, 0.1–2.0): ", 0.1, 2.0)
    
    blur_input = input(f"{YELLOW}Размытие (0–5), например: 0,1,2 или 0-3 или (all): {RESET}").strip()
    blur_list = parse_int_list(blur_input, 0, 5)
    if not blur_list:
        print(f"{RED}Ни одного корректного значения для 'Размытие'. Выход.{RESET}")
        return
    
    # Автоматическое определение диапазона эффектов
    max_mode = max(EFFECTS.keys())

    modes_input = input(f"{YELLOW}Тип эффекта (1–{max_mode}), например: 2,7,11,16 или 2-5 или (all): {RESET}").strip()
    modes_list = parse_int_list(modes_input, 1, max_mode)

    if not modes_list:
        print(f"{RED}Ни одного корректного значения для 'Тип эффекта'. Выход.{RESET}")
        return
    
    total = len(image_paths) * len(n_colors_list) * len(blur_list) * len(modes_list)
    print(f"\n{CYAN}Запустим {total} задач(и):{RESET}")
    print(f" • Файлов: {len(image_paths)}")
    print(f" • Цветов: {', '.join(map(str, n_colors_list))}")
    print(f" • Размытие: {', '.join(map(str, blur_list))}")
    print(f" • Типы: {', '.join(map(str, modes_list))}")
    print(f" • Масштаб: {scale}\n")
    
    completed = 0
    start_time = time.time()
    
    for idx_path, path in enumerate(image_paths, 1):
        for idx_color, n_colors in enumerate(n_colors_list, 1):
            for idx_blur, blur_strength in enumerate(blur_list, 1):
                for idx_mode, mode in enumerate(modes_list, 1):
                    completed += 1
                    percent = completed / total * 100
                    bar_len = 20
                    filled = int(bar_len * completed // total)
                    bar = f"{GREEN}{'█' * filled}{RESET}{'.' * (bar_len - filled)}"
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed if completed > 0 else 0
                    eta = avg_time * (total - completed)
                    mins, secs = divmod(int(eta), 60)
                    eta_str = f"{mins} мин {secs} сек" if mins > 0 else f"{secs} сек"
                    print(f"\r"
                          f"{BLUE}Файл {idx_path}/{len(image_paths)} | "
                          f"Цвета {idx_color}/{len(n_colors_list)} | "
                          f"Размытие {idx_blur}/{len(blur_list)} | "
                          f"Эффект {idx_mode}/{len(modes_list)}{RESET} | "
                          f"{bar} {YELLOW}{percent:5.1f}%{RESET} | "
                          f"{CYAN}{avg_time:.1f} сек/задача{RESET} | "
                          f"{MAGENTA}ETA: {eta_str}{RESET}",
                          end="", flush=True)
                    
                    print(f"\n\n{BOLD}{MAGENTA}> {os.path.basename(path)} — colors={n_colors}, blur={blur_strength}, mode={mode}{RESET}")
                    
                    try:
                        process_single(path, n_colors, scale, blur_strength, mode)
                    except Exception as e:
                        print(f"\n{RED}[ОШИБКА] {e}{RESET}")
                        log_error("Ошибка при обработке", e, path, n_colors, blur_strength, mode)
                        print(f"{YELLOW}Пропускаем...{RESET}")
    
    total_time = time.time() - start_time
    mins, secs = divmod(int(total_time), 60)
    print(f"\n\n{GREEN}ГОТОВО!{RESET} Обработано {total} задач за {mins} мин {secs} сек.")

if __name__ == "__main__":
    main()
