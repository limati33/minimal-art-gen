import cv2
import numpy as np
import os
import time
import subprocess # для запуска FFmpeg
from sklearn.cluster import MiniBatchKMeans
from utils.input_utils import print_progress
# Импортируем всё необходимое для логирования и отчёта
from utils.logging_utils import (
    OUTPUT_DIR, GREEN, RESET, YELLOW,
    EFFECT_NAMES,
    log_error
)
from utils.palette_utils import save_palette_image
from processor.effects import get_effect

# Доп. зависимости для контроля памяти (необязательная, graceful fallback)
try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:
    _HAVE_PSUTIL = False


def _human_size(bytes_val):
    """Простая функция для красивого отображения байт."""
    if bytes_val is None:
        return "N/A"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:3.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"


# ---- Вспомогательные функции для режима/цепочек эффектов ----
def apply_mode_sequence(img_in, mode_item, w, h, out_dir, base_name):
    """
    Применяет один режим (int) или последовательность режимов (tuple/list) по порядку.
    Возвращает итоговый кадр (np.uint8) или список (если эффект вернул список).
    """
    img = img_in
    if isinstance(mode_item, int):
        fn = get_effect(mode_item)
        return fn(img, w, h, out_dir, base_name)

    if isinstance(mode_item, (list, tuple)):
        for m in mode_item:
            fn = get_effect(m)
            img = fn(img, w, h, out_dir, base_name)
            if isinstance(img, list):
                return img
        return img

    # некорректный тип — просто возвращаем исходник
    return img_in


def format_mode_tag(mode_item):
    """Возвращает короткую метку для имени папки: m9 или m9+12"""
    if isinstance(mode_item, (list, tuple)):
        return "m" + "+".join(str(int(m)) for m in mode_item)
    return "m" + str(int(mode_item))


def format_effect_name(mode_item):
    """Человекочитаемое имя для отчёта: 'Пластик + Свеча'"""
    if isinstance(mode_item, (list, tuple)):
        return " + ".join(EFFECT_NAMES.get(int(m), f"Эффект{m}") for m in mode_item)
    return EFFECT_NAMES.get(int(mode_item), f"Эффект{mode_item}")


# ---- Основная функция ----
def process_video(video_path, n_colors, scale, blur_strength, mode):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Формирование тега режима / имени папки
    mode_tag = format_mode_tag(mode)
    effect_name_tag = f"_c{n_colors}_b{blur_strength}_{mode_tag}"
    out_dir = os.path.join(OUTPUT_DIR, f"{base_name}_{timestamp}{effect_name_tag}_VIDEO")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {video_path}")

    # Свойства видео
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Новый размер
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Временный файл (AVI)
    temp_path = os.path.join(out_dir, f"{base_name}_temp.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(temp_path, fourcc, fps if fps > 0 else 30.0, (new_w, new_h))

    print(f"\n{YELLOW}Обработка видео: {total_frames} кадров, {fps:.2f} FPS{RESET}")
    print(f"Размер: {orig_w}x{orig_h} -> {new_w}x{new_h}")

    # --- ЭТАП 1: Подбор / обучение палитры (если n_colors == None — авто) ---
    print_progress(10, prefix="Анализ палитры... ")

    # Пытаемся взять кадр из середины; fallback — первый
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2 if total_frames > 0 else 0)
    ret, sample_frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, sample_frame = cap.read()
    if not ret:
        raise IOError("Не удалось прочитать ни одного кадра для анализа палитры.")

    # ресайз образца
    if scale != 1.0:
        sample_frame = cv2.resize(sample_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Если n_colors == None — автоматически определяем по выборке (как в process_single)
    if n_colors is None:
        flat = sample_frame.reshape(-1, 3)
        total_pixels = flat.shape[0]
        sample_limit = 200_000
        if total_pixels > sample_limit:
            rng = np.random.default_rng(42)
            idx = rng.choice(total_pixels, size=sample_limit, replace=False)
            sample = flat[idx]
            sample_size = sample_limit
        else:
            sample = flat
            sample_size = total_pixels

        try:
            unique_colors = np.unique(sample, axis=0)
            unique_count = unique_colors.shape[0]
        except Exception:
            tuples = [tuple(c) for c in sample]
            unique_count = len(set(tuples))
            unique_colors = None

        max_allowed = 128
        n_colors_auto = max(1, min(unique_count, max_allowed))
        print(f"{YELLOW}Авто-режим (video): по выборке {sample_size} пикселей найдено ~{unique_count} уник. цветов; будет использовано {n_colors_auto}.{RESET}")
        n_colors = n_colors_auto

        # очистка
        del flat, sample
        if 'unique_colors' in locals():
            del unique_colors
        gc.collect()

    # Обучаем KMeans один раз (палитра для всего видео)
    pixels = sample_frame.reshape(-1, 3).astype(np.float32)
    try:
        kmeans = MiniBatchKMeans(n_clusters=n_colors, batch_size=100_000, random_state=42, n_init='auto')
    except TypeError:
        kmeans = MiniBatchKMeans(n_clusters=n_colors, batch_size=100_000, random_state=42, n_init=3)
    kmeans.fit(pixels)
    palette = np.uint8(kmeans.cluster_centers_)

    # Возвращаемся в начало
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- ЭТАП 2: Покадровая обработка ---
    start_time = time.time()
    processed_count = 0
    from collections import deque
    times = deque(maxlen=30)
    peak_mem = 0
    proc = psutil.Process(os.getpid()) if _HAVE_PSUTIL else None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_start = time.time()

            # ресайз + blur
            if scale != 1.0:
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if blur_strength > 0:
                k = int((blur_strength * 2 + 1)) | 1
                frame = cv2.GaussianBlur(frame, (k, k), 0)

            # квантование через заранее обученную палитру
            frame_pixels = frame.reshape(-1, 3).astype(np.float32)
            labels = kmeans.predict(frame_pixels)
            quantized = palette[labels].reshape(frame.shape)

            # применяем эффект/цепочку эффектов
            art_frame = apply_mode_sequence(quantized, mode, new_w, new_h, out_dir, f"frame_{processed_count}")

            # если эффект вернул список — берем первый (или fallback)
            if isinstance(art_frame, list):
                art_frame = art_frame[0] if art_frame else quantized

            if art_frame is None:
                art_frame = quantized

            # приводим к корректному виду для VideoWriter
            if art_frame.dtype != np.uint8:
                art_frame = np.clip(art_frame, 0, 255).astype(np.uint8)
            if art_frame.shape[1] != new_w or art_frame.shape[0] != new_h:
                art_frame = cv2.resize(art_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            out.write(art_frame)
            processed_count += 1

            # память
            if proc is not None:
                try:
                    mem = proc.memory_info().rss
                    if mem > peak_mem:
                        peak_mem = mem
                except Exception:
                    pass

            frame_time = time.time() - frame_start
            times.append(frame_time)
            elapsed = time.time() - start_time
            avg_frame_time = sum(times) / len(times) if len(times) > 0 else (elapsed / processed_count if processed_count else 0.0)
            fps_proc = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
            if processed_count % 10 == 0 or processed_count == 1:
                percent = (processed_count / total_frames) * 100 if total_frames > 0 else 0.0
                eta = (total_frames - processed_count) * avg_frame_time if (total_frames > 0 and avg_frame_time > 0) else 0
                print(f"\rОбработано: {processed_count}/{total_frames} [{percent:.1f}%] | FPS: {fps_proc:.2f} | ETA: {int(eta)}с", end="")

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Прервано пользователем. Сохраняем то, что успели...{RESET}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    duration = time.time() - start_time

    # --- ЭТАП 3: Финальное сжатие FFmpeg ---
    out_path = os.path.join(out_dir, f"{base_name}_processed.mp4")

    if os.path.exists(temp_path) and processed_count > 0:
        print(f"\n{YELLOW}--- ФИНАЛЬНОЕ СЖАТИЕ (FFmpeg) ---{RESET}")
        ffmpeg_command = [
            'ffmpeg', '-i', temp_path,
            '-c:v', 'libx264', '-crf', '18',
            '-preset', 'fast', '-pix_fmt', 'yuv420p',
            out_path, '-y'
        ]
        try:
            print("Запуск FFmpeg...")
            subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"{GREEN}Сжатие завершено! Итоговый файл: {out_path}{RESET}")
            os.remove(temp_path)
        except FileNotFoundError:
            log_error("FFmpeg не найден. Сжатие невозможно.", "Установите FFmpeg и добавьте в PATH.", None, n_colors, blur_strength, mode)
            os.rename(temp_path, out_path)
        except subprocess.CalledProcessError as e:
            log_error("Ошибка при выполнении FFmpeg.", str(e), None, n_colors, blur_strength, mode)
            os.rename(temp_path, out_path)

    # финальный размер/битрейт
    final_size = None
    bitrate_kbps = None
    if os.path.exists(out_path):
        try:
            final_size = os.path.getsize(out_path)
            if duration > 0 and processed_count > 0:
                bitrate_kbps = (final_size * 8) / duration / 1000.0
        except Exception:
            final_size = None
            bitrate_kbps = None

    # сохраняем палитру и отчёт
    palette_path = save_palette_image(palette, f"{base_name}_palette", out_dir)

    report_path = os.path.join(out_dir, f"{base_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Минималистичный арт-генератор (ВИДЕО) ===\n")
        f.write(f"Исходный файл: {video_path}\n")
        f.write(f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Количество цветов: {n_colors}\n")
        f.write(f"Масштаб: {scale}\n")
        f.write(f"Размытие: {blur_strength}\n")
        f.write(f"Тип: {format_effect_name(mode)} ({mode_tag})\n")
        f.write(f"Исходное разрешение: {orig_w}x{orig_h}\n")
        f.write(f"Итоговое разрешение: {new_w}x{new_h}\n")
        f.write(f"Общее количество кадров: {total_frames}\n")
        f.write(f"Итоговая длительность (обработка): {duration:.2f} сек\n")
        f.write(f"FPS исходное/обработки: {fps:.2f} / { (processed_count / duration) if duration>0 else 0.0 :.2f}\n\n")
        if _HAVE_PSUTIL:
            f.write(f"Пиковая память (RSS): {peak_mem / (1024*1024):.2f} MB\n")
        else:
            f.write("Пиковая память (RSS): psutil не установлен (N/A)\n")
        f.write(f"Итоговый файл: {out_path}\n")
        f.write(f"Итоговый размер файла: { _human_size(final_size) }\n")
        if bitrate_kbps is not None:
            f.write(f"Прибл. битрейт (H.264/CRF 18): {bitrate_kbps:.1f} kbit/s\n\n")
        else:
            f.write("Прибл. битрейт: N/A kbit/s\n\n")

        f.write("Палитра (RGB):\n")
        for color in palette:
            f.write(f" {tuple(int(c) for c in color)}\n")

    print(f"\n{GREEN}Готово!{RESET}")
    print(f" Результат: {out_path}")
    print(f" Палитра: {palette_path}")
    print(f" Отчёт: {report_path}")
