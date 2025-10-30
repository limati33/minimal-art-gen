import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
import os
import datetime
import textwrap
import time
import tkinter as tk
from tkinter import filedialog
from colorama import init, Fore, Style
import math
import random

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
        r, g, b = tuple(map(int, rgb))
        print(f"\033[48;2;{r};{g};{b}m \033[0m", end=" ")
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

# === Таблица эффектов с подсказками ===
def show_effects_table():
    effects = [
        (1,  "Постеризация",    "Чистое упрощение цветов, без фильтров.",              "Очень быстро", "Используй малое число цветов для графичных результатов"),
        (2,  "Плавные пятна",   "Мягкие переходы, глянец.",                            "Средне",       "Подходит для портретов — увеличь bilateral для более плавного блеска"),
        (3,  "Комикс",          "Контуры + сглаженные пятна — рисунок тушью.",         "Средне",       "Уменьши резкость контуров, если слишком агрессивно"),
        (4,  "Бумага (Cutout)", "Вырезанные слои, неровные края (как мел/ножницы).",   "Средне",       "Подходит для силуэтов — попробуй разное порога Canny"),
        (5,  "Сетка",           "Половинная тонировка/рубрикация — ретро-штриховка.",  "Очень быстро", "Шаг сетки влияет на «тон»"),
        (6,  "Пыль",            "Мягкий шум + рассеянный свет в ярких зонах.",         "Очень быстро", "Для эффекта — увеличь уровень шума и лёгкий blur"),
        (7,  "Мел",             "Рисование мелом: мягкие, немного сухие границы.",     "Медленно",     "Поиграй с detail / контрастом для «пыльного» мелка"),
        (8,  "Разбитое стекло", "Трещины поверх картинки + локальная дисторсия.",      "Средне",       "Трещины генерируются дополнительно — регенерация каждый запуск"),
        (9,  "Пластик",         "Глянцевый, игрушечный, яркие блики.",                 "Средне",       "Добавь specular highlight mask для усиления блеска"),
        (10, "Неон",            "Тёмный фон + светящиеся контуры (голубой/пурпур).",   "Средне",       "Попробуй разные неоновые палитры — cyan/magenta"),
        (11, "Хром",            "Металлический блеск, холодные тона (синее сяйво).",   "Средне",       "Используем градиенты + колормэп для контроля оттенка"),
        (12, "Свеча",           "Тёплое боковое освещение, золотистые тени.",          "Очень быстро", "Подойдёт для сцен с тёплым настроением"),
        (13, "Мох",             "Зелёный налёт, мягкая текстура.",                     "Средне",       "Добавляет зелёные шумовые слои и смягчение"),
        (14, "Акварель",        "Мягкие растекающиеся переходы, как кисть и вода.",    "Медленно",     "Убираем резкие контуры, ставим сильный edge-preserve + blur"),
        (15, "Технический",     "Чертёж: белые линии на синем фоне (blueprint).",      "Очень быстро", "Линии — белые, фон — синий; инвертируем каналы правильно"),
        (16, "Карта",           "Топографическая палитра — уровни как карта высот.",   "Очень быстро", "Подходит для пейзажей/спутника"),
        (17, "Мозаика",         "Набор случайных фигур в сетке.",                      "Среднее",      "Чем больше клеток — тем плотнее мозаика"),
        (18, "Сон",             "Плывущая, слегка искажённая структура.",              "Медленно",     "Искажения случайны — будут разные каждый раз"),
        (19, "Пиксель-Арт",     "Чистая пикселизация (ретро).",                        "Очень быстро", "scale_factor ~0.05..0.2"),
        (20, "Огненный",        "Тёплое свеча/пламя — мягкий glow в тёплых тонах.",    "Средне",       "Glow делается по яркостной маске, не просто контраст"),
    ]
    w_num, w_name, w_desc, w_speed = 4, 20, 45, 14
    wrap_desc = 45
    wrap_hint = 40
    print(f"\n{CYAN}{'='*110}{RESET}")
    print(f"{BOLD}{BLUE}СПИСОК ДОСТУПНЫХ ЭФФЕКТОВ (с подсказками){RESET}")
    print(f"{CYAN}{'-'*110}{RESET}")
    header = f"{BOLD}{MAGENTA}{'№':<{w_num}}{CYAN}{'Название':<{w_name}}{YELLOW}{'Описание':<{w_desc}}{GREEN}{'Скорость':<{w_speed}}{RESET}Подсказка"
    print(header)
    print(f"{CYAN}{'-'*110}{RESET}")
    for n, name, desc, speed, hint in effects:
        desc_lines = textwrap.wrap(desc, wrap_desc)
        hint_lines = textwrap.wrap(hint, wrap_hint)
        max_lines = max(len(desc_lines), len(hint_lines))
        for i in range(max_lines):
            num_str = f"{BOLD}{MAGENTA}{n:<{w_num}}{RESET}" if i == 0 else " "*w_num
            name_str = f"{CYAN}{name:<{w_name}}{RESET}" if i == 0 else " "*w_name
            desc_str = f"{YELLOW}{desc_lines[i]:<{w_desc}}{RESET}" if i < len(desc_lines) else " "*w_desc
            speed_str = f"{GREEN}{speed:<{w_speed}}{RESET}" if i == 0 else " "*w_speed
            hint_str = f"{hint_lines[i]}" if i < len(hint_lines) else ""
            print(f"{num_str}{name_str}{desc_str}{speed_str}{hint_str}")
    print(f"{CYAN}{'='*110}{RESET}\n")

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

    # Кластеризация — ИСПРАВЛЕНО: без сортировки
    print(f"\n{CYAN}[INFO]{RESET} Кластеризация ({n_colors} цветов) для {os.path.basename(image_path)} ...")
    start = time.time()
    pixels = img.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=n_colors, random_state=None, n_init=10)
    labels = kmeans.fit_predict(pixels)
    palette = np.uint8(kmeans.cluster_centers_)  # ← БЕЗ sort_palette
    quantized = palette[labels].reshape(img.shape).astype(np.uint8)

    # === ЭФФЕКТЫ (без изменений) ===
    if mode == 2:
        quantized = cv2.bilateralFilter(quantized, 9, 75, 75)
        quantized = cv2.convertScaleAbs(quantized, alpha=1.05, beta=6)
    elif mode == 3:
        smooth = cv2.bilateralFilter(quantized, 9, 75, 75)
        gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 140)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        edges_rgb = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2BGR)
        quantized = cv2.bitwise_and(smooth, edges_rgb)
    elif mode == 4:
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        edges = cv2.Canny(mask, 50, 150)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        mask_inv = cv2.bitwise_not(edges)
        out = cv2.bitwise_and(quantized, quantized, mask=mask_inv)
        shadow = cv2.GaussianBlur(out, (5,5), 0)
        quantized = cv2.addWeighted(out, 1.0, shadow, 0.08, 0)
    elif mode == 5:
        step = max(4, int(w / 100))
        grid = quantized.copy()
        for y in range(0, grid.shape[0], step):
            for x in range(0, grid.shape[1], step):
                if ((x//step) + (y//step)) % 2 == 0:
                    grid[y:y+step, x:x+step] = (grid[y:y+step, x:x+step] * 0.88).astype(np.uint8)
        quantized = grid
    elif mode == 6:
        noise = np.random.normal(0, 8, quantized.shape).astype(np.int16)
        noisy = np.clip(quantized.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(noisy, cv2.COLOR_RGB2GRAY)
        bright = cv2.threshold(gray, np.percentile(gray, 85), 255, cv2.THRESH_BINARY)[1]
        bright = cv2.GaussianBlur(bright, (21,21), 0)
        bright = (bright / 255.0)[:, :, None]
        bloom = (noisy.astype(np.float32) * (0.6 + 0.4 * bright)).clip(0,255).astype(np.uint8)
        quantized = cv2.GaussianBlur(bloom, (3,3), 0)
    elif mode == 7:
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 130)
        chalk = cv2.convertScaleAbs(quantized, alpha=0.95, beta=-10)
        grain = (np.random.normal(0, 18, quantized.shape)).astype(np.int16)
        chalk_g = np.clip(chalk.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        chalk_g = cv2.GaussianBlur(chalk_g, (3,3), 0)
        edges3 = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        mask_dark = (edges3[:,:,None] > 0).astype(np.uint8) * 30
        quantized = np.clip(chalk_g.astype(np.int16) - mask_dark, 0, 255).astype(np.uint8)
    elif mode == 8:
        hq, wq = quantized.shape[:2]
        cracks = np.zeros((hq, wq), dtype=np.uint8)
        n_cracks = max(3, min(12, int(min(hq,wq)/100)))
        center_x = np.random.randint(wq//3, wq*2//3)
        center_y = np.random.randint(hq//3, hq*2//3)
        for i in range(n_cracks):
            angle = np.random.uniform(0, 2*np.pi)
            length = np.random.randint(min(hq,wq)//3, min(hq,wq))
            x2 = int(center_x + np.cos(angle) * length)
            y2 = int(center_y + np.sin(angle) * length)
            cv2.line(cracks, (center_x, center_y), (x2, y2), 255, thickness=np.random.randint(1,3))
            if np.random.rand() > 0.5:
                ax = int(center_x + np.cos(angle + 0.3) * (length//2))
                ay = int(center_y + np.sin(angle + 0.3) * (length//2))
                cv2.line(cracks, (ax, ay), (x2, y2), 200, thickness=1)
        cracks = cv2.dilate(cracks, np.ones((3,3), np.uint8), iterations=1)
        cracks_rgb = cv2.cvtColor(cracks, cv2.COLOR_GRAY2BGR)
        mask = cracks.astype(bool)
        displaced = quantized.copy()
        ys, xs = np.where(mask)
        for (y, x) in zip(ys, xs):
            dx = int((x - center_x) * 0.02)
            dy = int((y - center_y) * 0.02)
            nx = np.clip(x + dx, 0, wq-1)
            ny = np.clip(y + dy, 0, hq-1)
            displaced[y,x] = quantized[ny, nx]
        quantized = cv2.addWeighted(quantized, 0.9, displaced, 0.1, 0)
        lines = cv2.cvtColor(cracks, cv2.COLOR_GRAY2BGR)
        lines = cv2.GaussianBlur(lines, (3,3), 0)
        quantized = cv2.addWeighted(quantized, 0.85, lines, 0.6, 0)
    elif mode == 9:
        plast = cv2.bilateralFilter(quantized, 7, 90, 90)
        plast = cv2.convertScaleAbs(plast, alpha=1.15, beta=8)
        gray = cv2.cvtColor(plast, cv2.COLOR_RGB2GRAY)
        spec = cv2.threshold(gray, np.percentile(gray, 92), 255, cv2.THRESH_BINARY)[1]
        spec = cv2.GaussianBlur(spec, (31,31), 0)
        spec = (spec / 255.0)[:, :, None]
        highlight = (255 * (0.6 * spec)).astype(np.uint8)
        quantized = np.clip(plast.astype(np.int16) + highlight.astype(np.int16), 0, 255).astype(np.uint8)
    elif mode == 10:
        dark = (quantized * 0.25).astype(np.uint8)
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        glow = np.zeros_like(quantized)
        glow[..., 0] = cv2.GaussianBlur(edges, (9,9), 0)
        glow[..., 2] = cv2.GaussianBlur(edges, (15,15), 0)
        glow = cv2.blur(glow, (7,7))
        glow = cv2.convertScaleAbs(glow * 2)
        quantized = cv2.addWeighted(dark, 1.0, glow, 0.9, 0)
    elif mode == 11:
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
        grad = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        grad = cv2.convertScaleAbs(grad)
        combined = cv2.addWeighted(gray, 0.6, grad, 0.8, 0)
        combined = cv2.equalizeHist(combined)
        chrome = cv2.applyColorMap(combined, cv2.COLORMAP_WINTER)
        quantized = cv2.addWeighted(chrome, 0.95, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.05, 0)
    elif mode == 12:
        overlay = np.full_like(quantized, (20, 30, 60))
        blurred = cv2.GaussianBlur(quantized, (15,15), 30)
        warm = cv2.addWeighted(blurred, 0.85, overlay, 0.25, 0)
        quantized = cv2.addWeighted(quantized, 0.75, warm, 0.6, 0)
    elif mode == 13:
        green = np.random.normal(0, 25, quantized.shape).astype(np.int16)
        moss = np.clip(quantized.astype(np.int16) + green, 0, 255).astype(np.uint8)
        moss[..., 1] = np.clip(moss[..., 1] + 18, 0, 255)
        quantized = cv2.bilateralFilter(moss, 9, 60, 60)
    elif mode == 14:
        soft = cv2.edgePreservingFilter(quantized, flags=1, sigma_s=60, sigma_r=0.6)
        color_flow = cv2.GaussianBlur(soft, (9,9), 3)
        quantized = cv2.addWeighted(soft, 0.6, color_flow, 0.4, 0)
    elif mode == 15:
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 120)
        lines = cv2.GaussianBlur(edges, (3,3), 0)
        lines = cv2.threshold(lines, 40, 255, cv2.THRESH_BINARY)[1]
        lines_rgb = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)
        blue_bg = np.full_like(quantized, (10, 60, 140))
        white_lines = cv2.bitwise_and(255 - lines_rgb, np.full_like(lines_rgb, 255))
        white_lines = (white_lines > 0).astype(np.uint8) * 255
        quantized = cv2.addWeighted(blue_bg, 1.0, white_lines, 1.0, 0)
    elif mode == 16:
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
        gray8 = cv2.convertScaleAbs(gray)
        cmap = cv2.__dict__.get('COLORMAP_TERRAIN', None)
        if cmap is None:
            cmap = cv2.COLORMAP_JET
        quantized = cv2.applyColorMap(gray8, cmap)

    #-----------------
    elif mode == 17:
        def rotate_pts(pts, angle, cx, cy):
            ca = math.cos(angle); sa = math.sin(angle)
            out = []
            for (x, y) in pts:
                rx = x - cx; ry = y - cy
                nx = rx * ca - ry * sa + cx
                ny = rx * sa + ry * ca + cy
                out.append((int(round(nx)), int(round(ny))))
            return out

        hq, wq = quantized.shape[:2]
        larger = max(wq, hq)
        desired_cell = 8  # базовый желаемый размер клетки в пикселях
        base = max(1, desired_cell)  # минимальный размер клетки = 1 пиксель

        cols = int(np.ceil(wq / base))
        rows = int(np.ceil(hq / base))


        abstract = np.zeros_like(quantized)
        shapes = ["circle", "square", "triangle", "diamond", "pentagon"]

        # Сначала рисуем сетку
        grid_color = (50, 50, 50)  # цвет линий сетки (темный)
        for r in range(rows+1):
            y = min(hq-1, r*base)
            cv2.line(abstract, (0, y), (wq-1, y), grid_color, 1)
        for c in range(cols+1):
            x = min(wq-1, c*base)
            cv2.line(abstract, (x, 0), (x, hq-1), grid_color, 1)

        # Затем заполняем клетки фигурами
        for ry in range(rows):
            for cx in range(cols):
                x0 = cx * base
                y0 = ry * base
                x1 = min(wq, (cx+1)*base)
                y1 = min(hq, (ry+1)*base)
                bw, bh = x1-x0, y1-y0
                if bw <=0 or bh <=0:
                    continue

                block = quantized[y0:y1, x0:x1]
                mean_col = tuple(map(int, block.mean(axis=(0,1)).astype(int)))

                # Цвет фигуры контрастный к среднему
                lum = 0.299*mean_col[0] + 0.587*mean_col[1] + 0.114*mean_col[2]
                delta = int(max(20, min(80, min(255, 128 - (lum - 128)))))
                if lum > 130:
                    shape_col = (max(0, mean_col[0]-delta), max(0, mean_col[1]-delta), max(0, mean_col[2]-delta))
                else:
                    shape_col = (min(255, mean_col[0]+delta), min(255, mean_col[1]+delta), min(255, mean_col[2]+delta))

                # Выбор фигуры
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
                        pts = [(cx_px-hw, cy_px-hh), (cx_px+hw, cy_px-hh),
                               (cx_px+hw, cy_px+hh), (cx_px-hw, cy_px+hh)]
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
                        pts = [(cx_px-hw, cy_px-hh), (cx_px+hw, cy_px-hh),
                               (cx_px+hw, cy_px+hh), (cx_px-hw, cy_px+hh)]

                    angle = random.uniform(-0.35, 0.35)
                    pts_rot = rotate_pts(pts, angle, cx_px, cy_px)
                    poly = np.array(pts_rot, dtype=np.int32)

                    area = abs(cv2.contourArea(poly))
                    if area < 4:
                        r = max(1, int(min(bw,bh)*0.25))
                        cv2.circle(abstract, (cx_px, cy_px), r, shape_col, -1)
                    else:
                        cv2.fillConvexPoly(abstract, poly, shape_col)

        # Лёгкая зернистость
        if max(hq, wq) > 200:
            noise = (np.random.randn(hq, wq, 1) * 6).astype(np.int16)
            abstract = np.clip(abstract.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        quantized = abstract

    #----------------- 

    elif mode == 18:
        offset = max(4, int(min(w,h) * 0.01))
        warped = np.zeros_like(quantized)
        for y in range(quantized.shape[0]):
            dx = int(np.sin(y / 20.0) * offset)
            for x in range(quantized.shape[1]):
                nx = np.clip(x + dx + np.random.randint(-offset, offset+1), 0, quantized.shape[1]-1)
                ny = np.clip(y + np.random.randint(-offset, offset+1), 0, quantized.shape[0]-1)
                warped[y, x] = quantized[ny, nx]
        quantized = cv2.GaussianBlur(warped, (3,3), 0)
    elif mode == 19:
        scale_factor = max(0.04, min(0.2, 32.0 / max(w, h)))
        small = cv2.resize(quantized, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        quantized = cv2.resize(small, (quantized.shape[1], quantized.shape[0]), interpolation=cv2.INTER_NEAREST)
    elif mode == 20:
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
        bright = cv2.threshold(gray, np.percentile(gray, 70), 255, cv2.THRESH_BINARY)[1]
        bright_blur = cv2.GaussianBlur(bright, (21,21), 0)
        bright_mask = (bright_blur / 255.0)[:, :, None]
        warm = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
        glow = (warm.astype(np.float32) * (0.8 * bright_mask)).clip(0,255).astype(np.uint8)
        quantized = cv2.addWeighted(quantized, 0.6, glow, 0.9, 0)
        sparks = (np.random.rand(*gray.shape) > 0.995).astype(np.uint8) * 255
        sparks = cv2.GaussianBlur(sparks, (5,5), 0)
        sparks_rgb = cv2.cvtColor(sparks, cv2.COLOR_GRAY2BGR)
        quantized = cv2.addWeighted(quantized, 1.0, sparks_rgb, 0.15, 0)

    duration = time.time() - start
    print(f"{GREEN}Кластеризация завершена за {duration:.2f} сек.{RESET}")

    # === Сохранение ===
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir_name = f"{base_name}_{timestamp}_c{n_colors}_b{blur_strength}_m{mode}"
    out_dir = os.path.join("output_minimal_art", out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    art_path = os.path.join(out_dir, f"{base_name}_minimal.png")
    Image.fromarray(quantized).save(art_path)
    palette_path = save_palette_image(palette, base_name, out_dir)

    mode_names = {1:"Постеризация",2:"Плавные пятна",3:"Комикс",4:"Бумага",5:"Сетка",6:"Пыль",
                  7:"Мел",8:"Разбитое стекло",9:"Пластик",10:"Неон",11:"Хром",12:"Свеча",
                  13:"Мох",14:"Акварель",15:"Технический",16:"Карта",17:"Мозаика",18:"Сон",
                  19:"Пиксель-Арт",20:"Огненный"}
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

    print(f"\n{GREEN}Сохранено:{RESET}")
    print(f" Результат: {art_path}")
    print(f" Палитра: {palette_path}")
    print(f" Настройки: {txt_path}")

    orig = Image.open(image_path).convert("RGB")
    if scale != 1.0:
        w0, h0 = orig.size
        orig = orig.resize((int(w0 * scale), int(h0 * scale)), Image.Resampling.LANCZOS)
    orig_np = np.array(orig)
    try:
        if orig_np.shape[0] == quantized.shape[0] and orig_np.shape[1] == quantized.shape[1]:
            combined = np.hstack((orig_np, quantized))
        else:
            q_resized = cv2.resize(quantized, (orig_np.shape[1], orig_np.shape[0]), interpolation=cv2.INTER_AREA)
            combined = np.hstack((orig_np, q_resized))
    except Exception:
        combined = quantized
    compare_path = os.path.join(out_dir, f"{base_name}_compare.png")
    Image.fromarray(combined).save(compare_path)
    print(f" Сравнение: {compare_path}")

    show_palette(palette)

# === Парсер списков ===
def parse_int_list(s: str, min_v: int, max_v: int):
    if not s:
        return []
    s = s.strip().lower()
    if s == "all":
        return list(range(min_v, max_v + 1))
    vals = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                a, b = part.split("-", 1)
                a = int(a.strip())
                b = int(b.strip())
                if a > b:
                    a, b = b, a
                for v in range(a, b + 1):
                    if min_v <= v <= max_v:
                        vals.add(v)
            except Exception:
                continue
        else:
            try:
                v = int(part)
                if min_v <= v <= max_v:
                    vals.add(v)
            except Exception:
                continue
    return sorted(vals)

# === Основной цикл ===
def main():
    print(f"\n{BOLD}{BLUE}МИНИМАЛИСТИЧНЫЙ АРТ-ГЕНЕРАТОР{RESET}")
    print(f"{MAGENTA}{'=' * 40}{RESET}")
    show_effects_table()

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
    if not image_paths:
        print(f"{RED}Файлы не выбраны. Выход.{RESET}")
        return
    print(f"{CYAN}Выбрано:{RESET} {', '.join(os.path.basename(p) for p in image_paths)}")

    ncolors_input = input(f"{YELLOW}Количество цветов (2–32), например: 4,8,12 или 4-12: {RESET}").strip()
    n_colors_list = parse_int_list(ncolors_input, 2, 32)
    if not n_colors_list:
        print(f"{RED}Ни одного корректного значения для 'Количество цветов'. Выход.{RESET}")
        return

    scale = ask_float("Масштаб (1 = оригинал, 0.1–2.0): ", 0.1, 2.0)

    blur_input = input(f"{YELLOW}Размытие (0–5), например: 0,1,2 или 0-3: {RESET}").strip()
    blur_list = parse_int_list(blur_input, 0, 5)
    if not blur_list:
        print(f"{RED}Ни одного корректного значения для 'Размытие'. Выход.{RESET}")
        return

    modes_input = input(f"{YELLOW}Тип эффекта (1–20), например: 2,7,11,16 или 2-5: {RESET}").strip()
    modes_list = parse_int_list(modes_input, 1, 20)
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

    for path in image_paths:
        for n_colors in n_colors_list:
            for blur_strength in blur_list:
                for mode in modes_list:
                    print(f"\n{BOLD}{MAGENTA}▶ {os.path.basename(path)} — colors={n_colors}, blur={blur_strength}, mode={mode}{RESET}")
                    try:
                        process_single(path, n_colors, scale, blur_strength, mode)
                    except Exception as e:
                        print(f"\n{RED}[ОШИБКА при обработке]{RESET} {e}")
                        import traceback
                        traceback.print_exc()
                        print(f"{YELLOW}Продолжаю со следующей комбинацией...{RESET}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{RED}[ОШИБКА]{RESET} {e}\n")
        import traceback
        traceback.print_exc()
        print(f"\n{YELLOW}Программа завершилась с ошибкой. Нажмите Enter, чтобы выйти...{RESET}")
        input()