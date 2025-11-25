import cv2
import numpy as np
import os
import time
from sklearn.cluster import MiniBatchKMeans
from utils.input_utils import print_progress
from utils.logging_utils import OUTPUT_DIR, GREEN, RESET, YELLOW
from processor.effects import get_effect

def process_video(video_path, n_colors, scale, blur_strength, mode):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(OUTPUT_DIR, f"{base_name}_{timestamp}_VIDEO")
    os.makedirs(out_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {video_path}")

    # Получаем свойства видео
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Расчет нового размера
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
    # Настройка сохранения видео
    out_path = os.path.join(out_dir, f"{base_name}_processed.mp4")
    # Используем mp4v кодек (работает на большинстве систем)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(out_path, fourcc, fps, (new_w, new_h))

    print(f"\n{YELLOW}Обработка видео: {total_frames} кадров, {fps:.2f} FPS{RESET}")
    print(f"Размер: {orig_w}x{orig_h} -> {new_w}x{new_h}")

    # --- ЭТАП 1: Обучение цветовой модели (чтобы не было мерцания) ---
    print_progress(10, prefix="Анализ палитры... ")
    
    # Берем кадр из середины для лучшего определения цветов
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, sample_frame = cap.read()
    if not ret: # Если не вышло, пробуем первый
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, sample_frame = cap.read()
    
    # Подготовка модели KMeans
    if scale != 1.0:
        sample_frame = cv2.resize(sample_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Обучаем KMeans один раз
    pixels = sample_frame.reshape(-1, 3).astype(np.float32)
    kmeans = MiniBatchKMeans(n_clusters=n_colors, batch_size=100_000, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    palette = np.uint8(kmeans.cluster_centers_)
    
    # Возвращаемся в начало
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # --- ЭТАП 2: Покадровая обработка ---
    start_time = time.time()
    processed_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Ресайз
            if scale != 1.0:
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 2. Размытие
            if blur_strength > 0:
                k = (blur_strength * 2 + 1) | 1
                frame = cv2.GaussianBlur(frame, (k, k), 0)
            
            # 3. Квантование цветов (используем уже обученный kmeans для скорости и стабильности)
            # Оптимизация: predict для всего кадра может быть медленным.
            # Для ускорения можно использовать просто ближайший цвет, но predict точнее.
            frame_pixels = frame.reshape(-1, 3).astype(np.float32)
            labels = kmeans.predict(frame_pixels)
            quantized = palette[labels].reshape(frame.shape)

            # 4. Эффект
            # Передаем заглушку вместо out_dir, чтобы эффекты не спамили файлами каждый кадр
            # Если эффект ТРЕБУЕТ сохранения файлов, это может вызвать проблемы, 
            # но большинство эффектов просто возвращают массив.
            art_frame = get_effect(mode)(quantized, new_w, new_h, out_dir, f"frame_{processed_count}")

            # Если эффект вернул что-то странное (список), берем первый элемент или пропускаем
            if isinstance(art_frame, list):
                art_frame = art_frame[0] if art_frame else quantized

            # Запись кадра
            out.write(art_frame)
            
            processed_count += 1
            
            # Обновление прогресса каждые 10 кадров
            if processed_count % 10 == 0:
                percent = (processed_count / total_frames) * 100
                elapsed = time.time() - start_time
                fps_proc = processed_count / elapsed if elapsed > 0 else 0
                eta = (total_frames - processed_count) / fps_proc if fps_proc > 0 else 0
                print(f"\rОбработано: {processed_count}/{total_frames} [{percent:.1f}%] | FPS: {fps_proc:.1f} | ETA: {int(eta)}с", end="")

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Прервано пользователем. Сохраняем то, что успели...{RESET}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"\n{GREEN}Видео сохранено: {out_path}{RESET}")
    
    # Сохраняем палитру для справки
    from utils.palette_utils import save_palette_image
    save_palette_image(palette, f"{base_name}_palette", out_dir)