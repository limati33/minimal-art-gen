import cv2
import numpy as np
from PIL import Image
import os
import time
import gc
from sklearn.cluster import MiniBatchKMeans
from utils.file_utils import resolve_shortcut
from utils.input_utils import print_progress
from utils.palette_utils import save_palette_image, show_palette
from utils.logging_utils import (
    log_error, OUTPUT_DIR, EFFECT_NAMES,
    GREEN, RESET, YELLOW, CYAN, RED, MAGENTA, BLUE
)
from processor.effects import get_effect


def process_single(image_path, n_colors, scale, blur_strength, mode):
    path = resolve_shortcut(image_path)
    base_name = os.path.splitext(os.path.basename(path))[0]
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(
        OUTPUT_DIR, f"{base_name}_{timestamp}_c{n_colors}_b{blur_strength}_m{mode}"
    )
    os.makedirs(out_dir, exist_ok=True)

    try:
        print_progress(1, prefix="Загрузка... ")
        start_time = time.time()
        img_pil = Image.open(path).convert("RGB")
        img = np.array(img_pil)
        h, w = img.shape[:2]

        if scale != 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        if blur_strength > 0:
            k = (blur_strength * 2 + 1) | 1
            img = cv2.GaussianBlur(img, (k, k), 0)

        print_progress(2, prefix=f"Кластеризация ({n_colors})... ")
        pixels = img.reshape(-1, 3).astype(np.float32)
        kmeans = MiniBatchKMeans(n_clusters=n_colors, batch_size=300_000, random_state=42)
        kmeans.fit(pixels)
        palette = np.uint8(kmeans.cluster_centers_)
        quantized = palette[kmeans.labels_].reshape(img.shape)
        del pixels
        gc.collect()

        print_progress(3, prefix="Эффект... ")
        quantized = get_effect(mode)(quantized, w, h, out_dir, base_name)

        print_progress(4, prefix="Сохранение... ")
        duration = time.time() - start_time
        art_path = os.path.join(out_dir, f"{base_name}_minimal.png")
        Image.fromarray(quantized).save(art_path)

        # Сохранение палитры
        palette_path = save_palette_image(palette, base_name, out_dir)

        # Создание compare
        create_compare(path, art_path, out_dir, base_name)

        # Создание текстового отчёта
        report_path = os.path.join(out_dir, f"{base_name}_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Минималистичный арт-генератор ===\n")
            f.write(f"Исходный файл: {path}\n")
            f.write(f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Количество цветов: {n_colors}\n")
            f.write(f"Масштаб: {scale}\n")
            f.write(f"Размытие: {blur_strength}\n")
            f.write(f"Тип: {EFFECT_NAMES.get(mode, 'Неизвестно')}\n")
            f.write(f"Время выполнения: {duration:.2f} сек\n\n")
            f.write("Палитра (RGB):\n")
            for color in palette:
                f.write(f" {tuple(color)}\n")

        print_progress(5, prefix="Готово! ")
        print(f"\n{GREEN}Сохранено:{RESET}")
        print(f" Результат: {art_path}")
        print(f" Палитра: {palette_path}")
        print(f" Отчёт: {report_path}")
        print(f" Сравнение: {os.path.join(out_dir, f'{base_name}_compare.png')}")
        show_palette(palette)

    except Exception as e:
        log_error("process_single", e, image_path, n_colors, blur_strength, mode)
        raise


def create_compare(original_path, result_path, out_dir, base_name):
    """Создаёт изображение сравнения: ориентация подбирается автоматически."""
    orig = Image.open(original_path).convert("RGB")
    res = Image.open(result_path).convert("RGB")

    ow, oh = orig.size
    rw, rh = res.size

    # Приведение по масштабу (высота одинакова для горизонтального компоновки)
    if ow / oh >= 1.1:  # альбомная
        # склеиваем по вертикали: оригинал сверху, арт снизу
        new_w = max(ow, rw)
        new_h = oh + rh
        compare = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        compare.paste(orig, ((new_w - ow) // 2, 0))
        compare.paste(res, ((new_w - rw) // 2, oh))
    else:  # портретная или квадратная
        # склеиваем по горизонтали: оригинал слева, арт справа
        new_w = ow + rw
        new_h = max(oh, rh)
        compare = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        compare.paste(orig, (0, (new_h - oh) // 2))
        compare.paste(res, (ow, (new_h - rh) // 2))

    compare_path = os.path.join(out_dir, f"{base_name}_compare.png")
    compare.save(compare_path)
    return compare_path
