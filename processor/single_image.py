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


def process_single(image_path, n_colors, scale, blur_strength, mode, out_dir=None, return_report=True):
    """
    Обрабатывает одиночное изображение, сохраняет результат, палитру и отчёт.
    Если n_colors is None -> автоматически определяет количество цветов по выборке.
    Возвращает словарь (интерфейс совпадает с process_video, где применимо).
    """
    try:
        path = resolve_shortcut(image_path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        # В имени папки безопасно подставляем строковое представление n_colors
        n_colors_label = "nolimit" if n_colors is None else str(n_colors)
        if isinstance(mode, (list, tuple)):
            mode_label = "+".join(str(int(m)) for m in mode)
        else:
            mode_label = str(int(mode))
        if out_dir is None:
            out_dir = os.path.join(OUTPUT_DIR, f"{base_name}_{timestamp}_c{n_colors_label}_b{blur_strength}_m{mode_label}")
        os.makedirs(out_dir, exist_ok=True)

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

        # --- Если режим "no limit", быстро считаем число уникальных цветов по выборке ---
        if n_colors is None:
            # Работать с uint8 — быстро и экономно по памяти
            flat_uint8 = img.reshape(-1, 3)
            total_pixels = flat_uint8.shape[0]

            # Размер выборки: не больше 200k пикселей
            sample_limit = 200_000
            if total_pixels > sample_limit:
                rng = np.random.default_rng(42)
                # выборка индексов без повторений
                idx = rng.choice(total_pixels, size=sample_limit, replace=False)
                sample = flat_uint8[idx]
                sample_size = sample_limit
            else:
                sample = flat_uint8
                sample_size = total_pixels

            # Считаем уникальные цвета в выборке
            try:
                unique_colors = np.unique(sample, axis=0)
                unique_count = unique_colors.shape[0]
            except Exception:
                # На всякий случай fallback — считаем через Python set (медленнее, но надёжно)
                tuples = [tuple(c) for c in sample]
                unique_count = len(set(tuples))
                unique_colors = None

            # Ограничиваем диапазон (1..128). Если хотите минимум 2, поменяйте 1->2.
            max_allowed = 128
            n_colors_auto = max(1, min(unique_count, max_allowed))

            print(f"{YELLOW}Авто-режим: по выборке {sample_size} пикселей найдено ~{unique_count} уникальных цветов; "
                  f"будет использовано {n_colors_auto} цвет(ов).{RESET}")

            # Присваиваем найденное число цветов для дальнейшей кластеризации
            n_colors = n_colors_auto

            # освобождаем временные объекты
            del flat_uint8, sample
            if 'unique_colors' in locals():
                del unique_colors
            gc.collect()

        # --- Кластеризация ---
        print_progress(2, prefix=f"Кластеризация ({n_colors})... ")
        pixels = img.reshape(-1, 3).astype(np.float32)

        # Безопасная проверка n_colors на случай неверных значений
        if not isinstance(n_colors, int) or n_colors < 1:
            n_colors = max(1, int(n_colors) if isinstance(n_colors, (int, float)) else 8)

        kmeans = MiniBatchKMeans(n_clusters=n_colors, batch_size=300_000, random_state=42)
        kmeans.fit(pixels)
        palette = np.uint8(kmeans.cluster_centers_)
        quantized = palette[kmeans.labels_].reshape(img.shape)
        del pixels
        gc.collect()

        print_progress(3, prefix="Эффект... ")

        # Поддерживаем mode в виде int или tuple/list (последовательность)
        def apply_mode_sequence(img_in, mode_item):
            """
            Возвращает изображение после применения одного эффекта (если mode_item int),
            или последовательности (если tuple/list) — применяются по порядку.
            Если эффект вернул список (несколько картинок), возвращаем этот список сразу.
            """
            img = img_in
            # единичный режим
            if isinstance(mode_item, int):
                fn = get_effect(mode_item)
                return fn(img, w, h, out_dir, base_name)
            # последовательность режимов
            elif isinstance(mode_item, (list, tuple)):
                for m in mode_item:
                    fn = get_effect(m)
                    img = fn(img, w, h, out_dir, base_name)
                    # если эффект вернул список/несколько артов — прерываем и возвращаем как есть
                    if isinstance(img, list):
                        return img
                return img
            else:
                # неверный тип — возвращаем исходник
                return img_in

        # применяем
        result_img = apply_mode_sequence(quantized, mode)

        # если эффект вернул несколько изображений (сам сохранил) — обрабатываем как раньше
        if isinstance(result_img, list):
            print(f"{YELLOW}Эффект вернул несколько изображений, сохранение пропущено.{RESET}")
            return {
                "out_path": None,
                "palette_path": None,
                "report_path": None,
                "duration": 0.0,
                "bitrate_kbps": None,
                "processed_frames": 0,
                "peak_mem_mb": None
            }
        quantized = result_img
        print_progress(4, prefix="Сохранение... ")
        duration = time.time() - start_time
        art_path = os.path.join(out_dir, f"{base_name}_minimal.png")
        Image.fromarray(quantized).save(art_path)

        palette_path = save_palette_image(palette, base_name, out_dir)
        compare_path = create_compare(path, art_path, out_dir, base_name)

        report_path = None
        if return_report:
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
                    f.write(f" {tuple(int(c) for c in color)}\n")

        print_progress(5, prefix="Готово! ")
        print(f"\n{GREEN}Сохранено:{RESET}")
        print(f" Результат: {art_path}")
        print(f" Палитра: {palette_path}")
        print(f" Отчёт: {report_path}")
        print(f" Сравнение: {compare_path}")

        return {
            "out_path": art_path,
            "palette_path": palette_path,
            "report_path": report_path,
            "duration": duration,
            "bitrate_kbps": None,
            "processed_frames": 1,
            "peak_mem_mb": None
        }

    except Exception as e:
        # Передаём сам объект исключения, чтобы log_error мог корректно читать exc.args
        log_error("process_single", e, image_path, n_colors, blur_strength, mode)
        return {"error": str(e)}


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
