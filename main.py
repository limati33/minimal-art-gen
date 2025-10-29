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
        print(f"\033[48;2;{r};{g};{b}m   \033[0m", end=" ")
    print("\n")

def save_palette_image(palette, base_name, out_dir):
    n = len(palette)
    palette_img = Image.new("RGB", (n * 50, 50))
    draw = ImageDraw.Draw(palette_img)
    for i, rgb in enumerate(palette):
        draw.rectangle([i * 50, 0, (i + 1) * 50, 50], fill=tuple(map(int, rgb)))
    path = os.path.join(out_dir, f"{base_name}_palette.png")
    palette_img.save(path)
    return path

def sort_palette(palette):
    return sorted(palette, key=lambda c: 0.3*c[0] + 0.59*c[1] + 0.11*c[2])

# === Таблица эффектов (показывается перед выбором режима) ===
def show_effects_table():
    effects = [
        (1,  "Постеризация",    "Чистое упрощение цветов, без фильтров.",               "Очень быстро"),
        (2,  "Плавные пятна",   "Мягкие переходы, как глянец.",                         "Средне"),
        (3,  "Комикс",          "Контуры и пятна, эффект рисовки.",                     "Средне"),
        (4,  "Бумага",          "Бумажный вырез, неровные края.",                       "Средне"),
        (5,  "Сетка",           "Половинная тонировка, сеточный узор.",                 "Очень быстро"),
        (6,  "Пыль",            "Небольшой шум и мягкость.",                            "Очень быстро"),
        (7,  "Мел",             "Эффект рисования мелом: мягкие размытые границы, будто по доске.", "Медленно"),
        (8,  "Разбитое стекло", "Имитация трещин и бликов.",                            "Средне"),
        (9,  "Пластик",         "Глянцевый, яркий, немного игрушечный.",                "Средне"),
        (10, "Неон",            "Тёмное + яркие светящиеся контуры.",                   "Средне"),
        (11, "Хром",            "Металлический блеск и холодные оттенки.",              "Средне"),
        (12, "Свеча",           "Тёплое свечение, мягкий свет.",                        "Очень быстро"),
        (13, "Мох",             "Зелёный налёт, природная мягкость.",                   "Средне"),
        (14, "Акварель",        "Плавные переходы, кистевая мягкость.",                 "Медленно"),
        (15, "Технический",     "Чертёж, белые линии на синем фоне.",                   "Очень быстро"),
        (16, "Карта",           "Топографическая палитра, как карта высот.",            "Очень быстро"),
        (17, "Зеркало",         "Симметрия и отражения.",                               "Очень быстро"),
        (18, "Сон",             "Плывущая, искажённая структура.",                      "Медленно"),
        (19, "Пиксель-Арт",     "Чистая пикселизация (ретро).",                        "Очень быстро"),
        (20, "Огненный",        "Тёплое свечение, как пламя.",                          "Средне"),
    ]

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}СПИСОК ДОСТУПНЫХ ЭФФЕКТОВ{RESET}")
    print(f"{CYAN}{'-'*60}{RESET}")
    print(f"{BOLD}{'№':<3} {'Название':<15} {'Описание':<35} {'Скорость'}{RESET}")
    print(f"{CYAN}{'-'*60}{RESET}")
    for n, name, desc, speed in effects:
        print(f"{BOLD}{n:<3}{RESET} {name:<15} {desc:<35} {speed}")
    print(f"{CYAN}{'='*60}{RESET}\n")

# === Обработка одного изображения ===
def process_single(image_path, n_colors, scale, blur_strength, mode):
    # Загрузка и подготовка
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    h, w = img.shape[:2]
    if scale != 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    if blur_strength > 0:
        k = (blur_strength * 2 + 1) | 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Кластеризация
    print(f"\n{CYAN}[INFO]{RESET} Кластеризация ({n_colors} цветов) для {os.path.basename(image_path)} ...")
    start = time.time()
    pixels = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=5)
    labels = kmeans.fit_predict(pixels)
    palette = np.uint8(kmeans.cluster_centers_)
    palette = np.array(sort_palette(palette))
    quantized = palette[labels].reshape(img.shape)

    # === ЭФФЕКТЫ ТИПОВ ===
    if mode == 1:  # Простая постеризация
        pass

    elif mode == 2:  # Плавные пятна
        quantized = cv2.bilateralFilter(quantized, 9, 75, 75)

    elif mode == 3:  # Комикс
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edges_inv = cv2.bitwise_not(edges)
        edges_inv = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2RGB)
        quantized = cv2.bitwise_and(quantized, edges_inv)

    elif mode == 4:  # Бумага (Cutout)
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 100)
        edges = cv2.dilate(edges, None)
        mask = cv2.bitwise_not(edges)
        quantized = cv2.bitwise_and(quantized, quantized, mask=mask)
        quantized = cv2.detailEnhance(quantized, sigma_s=10, sigma_r=0.15)

    elif mode == 5:  # Сетка (GridTone)
        step = 8
        for y in range(0, quantized.shape[0], step):
            for x in range(0, quantized.shape[1], step):
                if (x + y) // step % 2 == 0:
                    quantized[y:y+step, x:x+step] = (quantized[y:y+step, x:x+step] * 0.9).astype(np.uint8)

    elif mode == 6:  # Пыль (Dustlight)
        noise = np.random.normal(0, 10, quantized.shape).astype(np.int16)
        noisy = np.clip(quantized.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        quantized = cv2.GaussianBlur(noisy, (3, 3), 0)

    elif mode == 7:  # Стекло (Frost)
        hq, wq = quantized.shape[:2]
        offset = 5
        distorted = np.zeros_like(quantized)
        for y in range(hq):
            for x in range(wq):
                nx = min(wq-1, max(0, x + np.random.randint(-offset, offset+1)))
                ny = min(hq-1, max(0, y + np.random.randint(-offset, offset+1)))
                distorted[y, x] = quantized[ny, nx]
        quantized = cv2.GaussianBlur(distorted, (3, 3), 0)

    elif mode == 8:  # Разбитое стекло
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
        cracks = cv2.Canny(gray, 100, 200)
        cracks = cv2.dilate(cracks, None, iterations=2)
        cracks_rgb = cv2.cvtColor(cracks, cv2.COLOR_GRAY2RGB)
        quantized = cv2.addWeighted(quantized, 0.9, cracks_rgb, 0.4, 0)

    elif mode == 9:  # Пластик (Plastify)
        plast = cv2.bilateralFilter(quantized, 7, 100, 100)
        plast = cv2.convertScaleAbs(plast, alpha=1.2, beta=10)
        highlight = cv2.GaussianBlur(plast, (9, 9), 0)
        quantized = cv2.addWeighted(plast, 1.0, highlight, 0.2, 0)

    elif mode == 10:  # Неон (NeonDream)
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        glow = cv2.dilate(edges, None, iterations=2)
        glow = cv2.cvtColor(glow, cv2.COLOR_GRAY2BGR)
        neon = np.zeros_like(quantized)
        neon[..., 0] = np.clip(glow[..., 0] * 2, 0, 255)
        neon[..., 2] = np.clip(glow[..., 2] * 2, 0, 255)
        dark = (quantized * 0.3).astype(np.uint8)
        quantized = cv2.addWeighted(dark, 1.0, neon, 0.7, 0)

    elif mode == 11:  # Хром (Chrome)
        grad = cv2.morphologyEx(quantized, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
        grad_gray = cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY) if grad.ndim == 3 else grad
        grad_clip = cv2.convertScaleAbs(grad_gray, alpha=1.4, beta=20)
        quantized = cv2.applyColorMap(grad_clip, cv2.COLORMAP_WINTER)

    elif mode == 12:  # Свеча (Candlelight)
        overlay = np.full_like(quantized, (40, 20, 0))
        blurred = cv2.GaussianBlur(quantized, (15,15), 30)
        warm = cv2.addWeighted(blurred, 0.9, overlay, 0.4, 0)
        quantized = cv2.addWeighted(quantized, 0.8, warm, 0.6, 0)

    elif mode == 13:  # Мох (Moss)
        green = np.random.normal(0, 30, quantized.shape).astype(np.int16)
        moss = np.clip(quantized.astype(np.int16) + green, 0, 255).astype(np.uint8)
        moss[..., 1] = np.clip(moss[..., 1] + 20, 0, 255)
        quantized = cv2.bilateralFilter(moss, 9, 50, 50)

    elif mode == 14:  # Акварель (WaterBloom)
        soft = cv2.edgePreservingFilter(quantized, flags=1, sigma_s=50, sigma_r=0.6)
        quantized = cv2.detailEnhance(soft, sigma_s=15, sigma_r=0.4)

    elif mode == 15:  # Технический (Blueprint)
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 120)
        inv = cv2.bitwise_not(edges)
        blue = cv2.applyColorMap(inv, cv2.COLORMAP_OCEAN)
        quantized = cv2.addWeighted(quantized, 0.2, blue, 1.0, 0)

    elif mode == 16:  # Карта (Terrain)
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
        quantized = cv2.applyColorMap(gray, cv2.COLORMAP_TERRAIN)

    elif mode == 17:  # Зеркало (MirrorSplit)
        hq, wq = quantized.shape[:2]
        half = wq // 2
        left = quantized[:, :half]
        mirror = cv2.flip(left, 1)
        quantized = np.hstack((left, mirror))

    elif mode == 18:  # Сон (Dreamwarp)
        offset = 10
        warped = np.zeros_like(quantized)
        for y in range(quantized.shape[0]):
            for x in range(quantized.shape[1]):
                nx = min(quantized.shape[1]-1, max(0, x + np.random.randint(-offset, offset+1)))
                ny = min(quantized.shape[0]-1, max(0, y + np.random.randint(-offset, offset+1)))
                warped[y, x] = quantized[ny, nx]
        quantized = cv2.medianBlur(warped, 5)

    elif mode == 19:  # Пиксель-Арт (RetroGrid)
        scale_factor = 0.1
        small = cv2.resize(quantized, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        quantized = cv2.resize(small, (quantized.shape[1], quantized.shape[0]), interpolation=cv2.INTER_NEAREST)

    elif mode == 20:  # Огненный (Ember)
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
        glow = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
        quantized = cv2.addWeighted(quantized, 0.5, glow, 0.8, 0)

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
    mode_names = {
        1:"Постеризация",2:"Плавные пятна",3:"Комикс",4:"Бумага",5:"Сетка",6:"Пыль",
        7:"Мел",8:"Разбитое стекло",9:"Пластик",10:"Неон",11:"Хром",12:"Свеча",
        13:"Мох",14:"Акварель",15:"Технический",16:"Карта",17:"Зеркало",18:"Сон",
        19:"Пиксель-Арт",20:"Огненный"
    }
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
            f.write(f" {tuple(map(int, rgb))}\n")

    # Вывод итогов
    print(f"\n{GREEN}Сохранено:{RESET}")
    print(f" Результат: {art_path}")
    print(f" Палитра: {palette_path}")
    print(f" Настройки: {txt_path}")

    # --- Сравнение оригинал + результат ---
    orig = Image.open(image_path).convert("RGB")
    if scale != 1.0:
        w0, h0 = orig.size
        orig = orig.resize((int(w0 * scale), int(h0 * scale)), Image.Resampling.LANCZOS)
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

    # Показываем таблицу эффектов
    show_effects_table()

    # Получаем файлы (аргументы CLI или диалог)
    args = sys.argv[1:]
    image_paths = []
    if args:
        for a in args:
            if os.path.isfile(a):
                image_paths.append(a)
            else:
                print(f"{YELLOW}Внимание: '{a}' не найден или это не файл — пропущен.{RESET}")
        if not image_paths:
            print(f"{RED}Нет корректных файлов в аргументах.{RESET}")
            return
    else:
        image_paths = select_images_via_dialog(multi=True)

    # Вывод выбранных файлов
    if image_paths:
        print(f"{CYAN}Выбран файл:{RESET} {', '.join(os.path.basename(p) for p in image_paths)}")
    else:
        print(f"{RED}Файлы не выбраны. Выход.{RESET}")
        return

    # Параметры
    n_colors = ask_int("Количество цветов (2–32): ", 2, 32)
    scale = ask_float("Масштаб (1 = оригинал, 0.5 = в 2 раза меньше): ", 0.1, 2)
    blur_strength = ask_int("Размытие (0 = выкл, 1–5): ", 0, 5)

    mode = ask_int("Выберите тип (1–20): ", 1, 20)

    # Обрабатываем каждый файл по очереди
    for path in image_paths:
        process_single(path, n_colors, scale, blur_strength, mode)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{RED}[ОШИБКА]{RESET} {e}\n")
        import traceback
        traceback.print_exc()
        print(f"\n{YELLOW}Программа завершилась с ошибкой. Нажмите Enter, чтобы выйти...{RESET}")
        input()
