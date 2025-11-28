import cv2
import numpy as np
import os
import time
import subprocess # НОВЫЙ ИМПОРТ: для запуска FFmpeg
from sklearn.cluster import MiniBatchKMeans
from utils.input_utils import print_progress
# Импортируем все необходимое для логирования и отчета
from utils.logging_utils import (
    OUTPUT_DIR, GREEN, RESET, YELLOW,
    EFFECT_NAMES,
    log_error # <<< ДОБАВЛЕНО для исправления NameError
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


def process_video(video_path, n_colors, scale, blur_strength, mode):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # --- ИЗМЕНЕНИЕ: Формирование имени папки с параметрами ---
    effect_name_tag = f"_c{n_colors}_b{blur_strength}_m{mode}"
    out_dir = os.path.join(
        OUTPUT_DIR, 
        f"{base_name}_{timestamp}{effect_name_tag}_VIDEO"
    )
    os.makedirs(out_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {video_path}")

    # Получаем свойства видео
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    # Расчет нового размера
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
    # Настройка сохранения видео (ЗАПИСЫВАЕМ В БОЛЬШОЙ ВРЕМЕННЫЙ ФАЙЛ)
    # Используем AVI для временного файла, чтобы избежать проблем с mp4v и крупными битрейтами
    temp_path = os.path.join(out_dir, f"{base_name}_temp.avi") 
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # MJPG - быстрый, но большой кодек
    out = cv2.VideoWriter(temp_path, fourcc, fps if fps > 0 else 30.0, (new_w, new_h))

    print(f"\n{YELLOW}Обработка видео: {total_frames} кадров, {fps:.2f} FPS{RESET}")
    print(f"Размер: {orig_w}x{orig_h} -> {new_w}x{new_h}")

    # --- ЭТАП 1: Обучение цветовой модели (чтобы не было мерцания) ---
    print_progress(10, prefix="Анализ палитры... ")
    
    # Берем кадр из середины для лучшего определения цветов
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2 if total_frames > 0 else 0)
    ret, sample_frame = cap.read()
    if not ret: # Если не вышло, пробуем первый
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, sample_frame = cap.read()
    if not ret:
        raise IOError("Не удалось прочитать ни одного кадра для анализа палитры.")

    # Подготовка модели KMeans
    if scale != 1.0:
        sample_frame = cv2.resize(sample_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Обучаем KMeans один раз
    pixels = sample_frame.reshape(-1, 3).astype(np.float32)
    # Оберегаем n_init для совместимости со старыми версиями sklearn
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

    # Для скользящего ETA
    from collections import deque
    times = deque(maxlen=30)

    # Memory tracking
    peak_mem = 0
    proc = psutil.Process(os.getpid()) if _HAVE_PSUTIL else None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # 1. Ресайз
            if scale != 1.0:
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 2. Размытие
            if blur_strength > 0:
                k = int((blur_strength * 2 + 1)) | 1
                frame = cv2.GaussianBlur(frame, (k, k), 0)
            
            # 3. Квантование цветов (используем уже обученный kmeans для скорости и стабильности)
            frame_pixels = frame.reshape(-1, 3).astype(np.float32)
            labels = kmeans.predict(frame_pixels)
            quantized = palette[labels].reshape(frame.shape)

            # 4. Эффект
            # УНИФИКАЦИЯ ИНТЕРФЕЙСА: передаем n_colors, blur_strength, mode
            art_frame = get_effect(mode)(
                quantized, new_w, new_h, out_dir, f"frame_{processed_count}",
                n_colors=n_colors, blur_strength=blur_strength, mode=mode 
            )
            
            # Если эффект вернул что-то странное (список), берем первый элемент
            if isinstance(art_frame, list):
                art_frame = art_frame[0] if art_frame else quantized

            # Запись кадра
            if art_frame is None:
                art_frame = quantized
            # Убедимся, что формат корректный для VideoWriter (uint8, BGR, размер совпадает)
            if art_frame.dtype != np.uint8:
                art_frame = np.clip(art_frame, 0, 255).astype(np.uint8)
            if art_frame.shape[1] != new_w or art_frame.shape[0] != new_h:
                art_frame = cv2.resize(art_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            out.write(art_frame)
            
            processed_count += 1

            # Memory sample
            if proc is not None:
                try:
                    mem = proc.memory_info().rss
                    if mem > peak_mem:
                        peak_mem = mem
                except Exception:
                    pass

            # Update sliding ETA
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
    
    # --- ЭТАП 3: ФИНАЛЬНОЕ СЖАТИЕ FFmpeg (H.264) ---
    out_path = os.path.join(out_dir, f"{base_name}_processed.mp4")
    
    if os.path.exists(temp_path) and processed_count > 0:
        print(f"\n{YELLOW}--- ФИНАЛЬНОЕ СЖАТИЕ (FFmpeg) ---{RESET}")
        
        # CRF (Constant Rate Factor) 18 - почти неразличимое качество от оригинала
        # Кодек libx264 (H.264)
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
            os.remove(temp_path) # Удаляем большой временный файл
        except FileNotFoundError:
            log_error("FFmpeg не найден. Сжатие невозможно.", "Установите FFmpeg и убедитесь, что он добавлен в PATH.", None, n_colors, blur_strength, mode)
            os.rename(temp_path, out_path) # Переименовываем временный файл, если FFmpeg не сработал
        except subprocess.CalledProcessError as e:
            log_error("Ошибка при выполнении FFmpeg.", str(e), None, n_colors, blur_strength, mode)
            os.rename(temp_path, out_path)
            
    # Расчет финального размера и битрейта для отчета
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

    # --- БЛОК: Сохранение результатов и отчета ---

    # Сохраняем палитру
    palette_path = save_palette_image(palette, f"{base_name}_palette", out_dir)
    
    # Создание текстового отчёта
    report_path = os.path.join(out_dir, f"{base_name}_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Минималистичный арт-генератор (ВИДЕО) ===\n")
        f.write(f"Исходный файл: {video_path}\n")
        f.write(f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Количество цветов: {n_colors}\n")
        f.write(f"Масштаб: {scale}\n")
        f.write(f"Размытие: {blur_strength}\n")
        f.write(f"Тип: {EFFECT_NAMES.get(mode, 'Неизвестно')} (Mode {mode})\n")
        f.write(f"Исходное разрешение: {orig_w}x{orig_h}\n")
        f.write(f"Итоговое разрешение: {new_w}x{new_h}\n")
        f.write(f"Общее количество кадров: {total_frames}\n")
        f.write(f"Итоговая длительность (обработка): {duration:.2f} сек\n")
        f.write(f"FPS исходное/обработки: {fps:.2f} / { (processed_count / duration) if duration>0 else 0.0 :.2f}\n\n")
        # Память
        if _HAVE_PSUTIL:
            f.write(f"Пиковая память (RSS): {peak_mem / (1024*1024):.2f} MB\n")
        else:
            f.write("Пиковая память (RSS): psutil не установлен (N/A)\n")
        # Битрейт и размер
        f.write(f"Итоговый файл: {out_path}\n")
        f.write(f"Итоговый размер файла: { _human_size(final_size) }\n")
        if bitrate_kbps is not None:
            f.write(f"Прибл. битрейт (H.264/CRF 18): {bitrate_kbps:.1f} kbit/s\n\n") # Обновляем метку
        else:
            f.write("Прибл. битрейт: N/A kbit/s\n\n")

        f.write("Палитра (RGB):\n")
        for color in palette:
            f.write(f" {tuple(int(c) for c in color)}\n")

    print(f"\n{GREEN}Готово!{RESET}")
    print(f" Результат: {out_path}")
    print(f" Палитра: {palette_path}")
    print(f" Отчёт: {report_path}")