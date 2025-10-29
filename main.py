import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
import os
import datetime
import time
import tkinter as tk
from tkinter import filedialog
from colorama import init, Fore, Style
# === Инициализация цветов терминала ===
init(autoreset=True)
RESET = Style.RESET_ALL
BOLD = Style.BRIGHT
CYAN = Fore.CYAN
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
RED = Fore.RED
MAGENTA = Fore.MAGENTA
BLUE = Fore.BLUE
# === Утилиты ввода ===
def ask_int(prompt, min_v, max_v):
    while True:
        try:
            v = int(input(f"{YELLOW}{prompt}{RESET}"))
            if min_v <= v <= max_v:
                return v
            print(f"{RED}Введите число от {min_v} до {max_v}.{RESET}")
        except ValueError:
            print(f"{RED}Нужно целое число!{RESET}")
def ask_float(prompt, min_v, max_v):
    while True:
        try:
            v = float(input(f"{YELLOW}{prompt}{RESET}"))
            if min_v <= v <= max_v:
                return v
            print(f"{RED}Введите значение от {min_v} до {max_v}.{RESET}")
        except ValueError:
            print(f"{RED}Нужно число!{RESET}")
# === UI / Файловая логика ===
def select_images_via_dialog(multi=False):
    root = tk.Tk()
    root.withdraw()
    print(f"{CYAN}Выбор изображения...{RESET}")
    if multi:
        paths = filedialog.askopenfilenames(
            title="Выберите изображение(я)",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff")]
        )
        return list(paths)
    else:
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff")]
        )
        return [path] if path else []
# === Визуализация палитры в терминале ===
def show_palette(palette):
    print(f"\n{CYAN}Палитра цветов:{RESET}")
    for rgb in palette:
        r, g, b = rgb
        # ANSI 24-bit фон: \033[48;2;R;G;Bm
        print(f"\033[48;2;{r};{g};{b}m \033[0m", end=" ")
    print("\n")
def save_palette_image(palette, base_name, out_dir):
    n = len(palette)
    palette_img = Image.new("RGB", (n * 50, 50))
    draw = ImageDraw.Draw(palette_img)
    for i, rgb in enumerate(palette):
        draw.rectangle([i * 50, 0, (i + 1) * 50, 50], fill=tuple(rgb))
    path = os.path.join(out_dir, f"{base_name}_palette.png")
    palette_img.save(path)
    return path
# === Обработка одного изображения ===
def process_single(image_path, n_colors, scale, blur_strength, mode):
    # Загрузка и подготовка
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    h, w = img.shape[:2]
    if scale != 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    if blur_strength > 0:
        img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)
    # Кластеризация
    print(f"\n{CYAN}[INFO]{RESET} Кластеризация ({n_colors} цветов) для {os.path.basename(image_path)} ...")
    start = time.time()
    pixels = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=5)
    labels = kmeans.fit_predict(pixels)
    palette = np.uint8(kmeans.cluster_centers_)
    quantized = palette[labels].reshape(img.shape)
    # Эффекты типов
    if mode == 2:
        quantized = cv2.bilateralFilter(quantized, 9, 75, 75)
    elif mode == 3:
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edges_inv = cv2.bitwise_not(edges)
        edges_inv = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2RGB)
        quantized = cv2.bitwise_and(quantized, edges_inv)
    duration = time.time() - start
    print(f"{GREEN}Кластеризация завершена за {duration:.2f} сек.{RESET}")
    # Сохранение результатов (используем исходное имя + приписки)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("output_minimal_art", f"{base_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    art_path = os.path.join(out_dir, f"{base_name}_minimal.png")
    Image.fromarray(quantized).save(art_path)
    palette_path = save_palette_image(palette, base_name, out_dir)
    # Сохраняем настройки в txt
    mode_names = {1: "Постеризация", 2: "Плавные пятна", 3: "Комикс"}
    txt_path = os.path.join(out_dir, f"{base_name}_settings.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Минималистичный арт-генератор ===\n")
        f.write(f"Исходный файл: {image_path}\n")
        f.write(f"Дата: {datetime.datetime.now()}\n\n")
        f.write(f"Количество цветов: {n_colors}\n")
        f.write(f"Масштаб: {scale}\n")
        f.write(f"Размытие: {blur_strength}\n")
        f.write(f"Тип: {mode_names.get(mode, mode)}\n")
        f.write(f"Время выполнения: {duration:.2f} сек\n\n")
        f.write("Палитра (RGB):\n")
        for rgb in palette:
            f.write(f" {tuple(rgb)}\n")
    # Вывод итогов
    print(f"\n{GREEN}Сохранено:{RESET}")
    print(f" Результат: {art_path}")
    print(f" Палитра: {palette_path}")
    print(f" Настройки: {txt_path}")
    # --- Сравнение оригинал + результат ---
    orig = Image.open(image_path).convert("RGB")
    if scale != 1.0:
        w, h = orig.size
        orig = orig.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    orig_np = np.array(orig)
    combined = np.hstack((orig_np, quantized))
    compare_path = os.path.join(out_dir, f"{base_name}_compare.png")
    Image.fromarray(combined).save(compare_path)
    print(f" Сравнение: {compare_path}")
    # Показываем палитру в терминале
    show_palette(palette)
# === Главная логика приложения ===
def main():
    print(f"\n{BOLD}{BLUE}МИНИМАЛИСТИЧНЫЙ АРТ-ГЕНЕРАТОР{RESET}")
    print(f"{MAGENTA}{'=' * 40}{RESET}")
    # Если есть аргументы в командной строке — используем их как пути к файлам
    if image_paths:
        print(f"{CYAN}Выбран файл:{RESET} {', '.join(os.path.basename(p) for p in image_paths)}")
    args = sys.argv[1:]
    image_paths = []
    if args:
        # фильтруем несуществующие пути
        for a in args:
            if os.path.isfile(a):
                image_paths.append(a)
            else:
                print(f"{YELLOW}Внимание: '{a}' не найден или это не файл — пропущен.{RESET}")
        if not image_paths:
            print(f"{RED}Нет корректных файлов в аргументах.{RESET}")
    else:
        # если аргументов нет — открываем диалог (позволим выбрать несколько файлов)
        image_paths = select_images_via_dialog(multi=True)
    if not image_paths:
        print(f"{RED}Файлы не выбраны. Выход.{RESET}")
        return
    # Общие параметры (запрашиваем один раз и применяем ко всем выбранным файлам)
    n_colors = ask_int("Количество цветов (2–32): ", 2, 32)
    scale = ask_float("Масштаб (1 = оригинал, 0.5 = в 2 раза меньше): ", 0.1, 2)
    blur_strength = ask_int("Размытие (0 = выкл, 1–5): ", 0, 5)
    print(f"\n{CYAN}Типы минимализма:{RESET}")
    print(f" {BOLD}1{RESET} — Простая постеризация")
    print(f" {BOLD}2{RESET} — Плавные пятна (мягкий арт)")
    print(f" {BOLD}3{RESET} — Комикс (жёсткие контуры)")
    mode = ask_int("Выберите тип (1/2/3): ", 1, 3)
    # Обрабатываем каждый файл по очереди
    for path in image_paths:
        process_single(path, n_colors, scale, blur_strength, mode)
if __name__ == "__main__":
    main()