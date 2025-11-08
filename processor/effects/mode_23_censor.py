import cv2
import numpy as np
import os
import traceback

# Цвета для логов
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def apply_censor(
    img,
    w=None, h=None,
    out_dir=None, base_name=None,
):
    """
    Эффект "цензуры" — несколько стилей закрытия глаз.
    Работает как с человеческими лицами, так и с аниме.
    Если глаз не найдено -> печать "Нету глаз!" и возвращаем оригинал.
    """
    try:
        if img is None:
            raise ValueError("Input image is None")

        # --- масштаб ---
        if w and h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        h_img, w_img = img.shape[:2]

        # ensure out_dir exists (safe to call even if None)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes_all = []

        # === 1. Попытка использовать аниме-каскад ===
        anime_cascade_path = os.path.join(
            os.path.dirname(__file__), "data", "lbpcascade_animeface.xml"
        )
        if os.path.exists(anime_cascade_path):
            anime_cascade = cv2.CascadeClassifier(anime_cascade_path)
            faces = anime_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
            )
            for (x, y, w0, h0) in faces:
                eyes_all.append((x, y + h0 // 4, w0, h0 // 3))
            if eyes_all:
                print(f"{GREEN}[Anime]{RESET} найдено лиц: {len(eyes_all)}")

        # === 2. Если аниме не найдено — fallback на обычные каскады ===
        if not eyes_all:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (fx, fy, fw, fh) in faces:
                roi_gray = gray[fy:fy + fh, fx:fx + fw]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    eyes_all.append((fx + ex, fy + ey, ew, eh))
            if eyes_all:
                print(f"{CYAN}[Human]{RESET} найдено глаз: {len(eyes_all)}")

        # === 3. Если вообще ничего не найдено ===
        if not eyes_all:
            msg = f"{YELLOW}Нету глаз!{RESET}"
            print(msg)
            if out_dir and base_name:
                try:
                    txt_path = os.path.join(out_dir, f"{base_name}_no_eyes.txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write("Нету глаз!\n")
                    print(f"Saved: {txt_path}")
                except Exception as e:
                    print(f"Failed to write no-eyes file: {e}")
            return img

        results = {}

        # ============= 1. BLACK STRIKE =============
        bs_img = img.copy()
        for (x, y, ew, eh) in eyes_all:
            y_mid = y + eh // 2
            thickness = max(2, int(eh * 1.2))
            for i in range(-thickness//2, thickness//2 + 1):
                offset = np.random.randint(-3, 4)
                cv2.line(bs_img,
                         (x - int(ew * 0.1), y_mid + i + offset),
                         (x + ew + int(ew * 0.1), y_mid + i + offset),
                         (0, 0, 0), 1, cv2.LINE_AA)
            n_spl = max(6, ew // 6)
            for _ in range(n_spl):
                sx = np.random.randint(x - ew//6, x + ew + ew//6)
                sy = np.random.randint(y_mid - thickness, y_mid + thickness)
                r = np.random.randint(1, max(2, thickness//3))
                cv2.circle(bs_img, (sx, sy), r, (0,0,0), -1)
        results["bs"] = bs_img

        # ============= 2. RED CROSS =============
        rc_img = img.copy()
        for (x, y, ew, eh) in eyes_all:
            cx, cy = x + ew // 2, y + eh // 2
            len_cross = int(max(ew, eh) * 1.6)
            color = (0, 0, 255)
            thickness = max(1, int(min(ew, eh) * 0.18))
            for _ in range(2):
                offx, offy = np.random.randint(-3, 4, 2)
                cv2.line(rc_img,
                         (cx - len_cross//2 + offx, cy - len_cross//2 + offy),
                         (cx + len_cross//2 + offx, cy + len_cross//2 + offy),
                         color, thickness, cv2.LINE_AA)
                offx, offy = np.random.randint(-3, 4, 2)
                cv2.line(rc_img,
                         (cx + len_cross//2 + offx, cy - len_cross//2 + offy),
                         (cx - len_cross//2 + offx, cy + len_cross//2 + offy),
                         color, thickness, cv2.LINE_AA)
        results["rc"] = rc_img

        # ============= 3. PIXEL BLOCK =============
        px_img = img.copy()
        for (x, y, ew, eh) in eyes_all:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w_img, x+ew), min(h_img, y+eh)
            roi = px_img[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            k = max(2, min(max(1, (x1-x0)//8), (y1-y0)//8))
            small = cv2.resize(roi, (k, k), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST)
            px_img[y0:y1, x0:x1] = pixelated
        results["px"] = px_img

        # ============= 4. BLUR BAND =============
        bl_img = img.copy()
        for (x, y, ew, eh) in eyes_all:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w_img, x+ew), min(h_img, y+eh)
            roi = bl_img[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            ksz = max(15, int(min((x1-x0), (y1-y0)) * 0.6))
            blurred = cv2.GaussianBlur(roi, (ksz|1, ksz|1), ksz/3)
            bl_img[y0:y1, x0:x1] = blurred
        results["bl"] = bl_img

        # ============= 5. SPRAY PAINT =============
        sp_img = img.copy()
        for (x, y, ew, eh) in eyes_all:
            center_x = x + ew // 2
            center_y = y + eh // 2
            mask = np.zeros((h_img, w_img), np.uint8)
            cv2.ellipse(mask, (center_x, center_y), (ew // 2, eh // 2), 0, 0, 360, 255, -1)
            spray = np.zeros_like(mask, np.float32)
            niter = max(200, int((ew*eh) / 20))
            for _ in range(niter):
                xs = np.random.randint(center_x - ew, center_x + ew)
                ys = np.random.randint(center_y - eh, center_y + eh)
                if 0 <= xs < w_img and 0 <= ys < h_img:
                    val = np.random.rand()
                    spray[ys, xs] += val * (1.0 if mask[ys, xs] > 0 else 0.25)
            spray = cv2.GaussianBlur(spray, (13, 13), 7)
            maxv = spray.max()
            if maxv <= 1e-6:
                continue
            spray_u = (spray / maxv * 255).astype(np.uint8)
            color = (
                np.random.randint(20, 220),
                np.random.randint(20, 220),
                np.random.randint(20, 220),
            )
            color_layer = np.zeros_like(sp_img)
            color_layer[:] = color
            sp_img[spray_u > 90] = color_layer[spray_u > 90]
        results["sp"] = sp_img

        # ============= 6. INK SPLATTER =============
        in_img = img.copy()
        for (x, y, ew, eh) in eyes_all:
            blob = np.zeros((h_img, w_img), np.uint8)
            center = (x + ew//2, y + eh//2)
            for _ in range(100):
                ox = np.random.randint(-ew, ew)
                oy = np.random.randint(-eh, eh)
                r = np.random.randint(2, max(3, int(min(ew, eh) * 0.15)))
                cx = center[0] + ox
                cy = center[1] + oy
                if 0 <= cx < w_img and 0 <= cy < h_img:
                    cv2.circle(blob, (cx, cy), r, 255, -1)
            blob = cv2.GaussianBlur(blob, (9, 9), 5)
            in_img[blob > 80] = (0, 0, 0)
        results["in"] = in_img

        # ============= 7. TAPE (CENSORED label) =============
        tp_img = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for (x, y, ew, eh) in eyes_all:
            label = "CENSORED"
            tx = max(0, x - 10)
            ty = y + eh // 2
            tw = min(w_img - tx - 1, ew + 20)
            th = int(min(h_img - 1, max(10, eh * 1.2)))
            cv2.rectangle(tp_img, (tx, ty - th//2), (tx + tw, ty + th//2), (20, 20, 20), -1)
            cv2.putText(tp_img, label, (tx + 6, ty + int(th*0.15)),
                        font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        results["tp"] = tp_img

        # --- сохранение всех ---
        if out_dir and base_name:
            for key, res in results.items():
                try:
                    path = os.path.join(out_dir, f"{base_name}_{key}.png")
                    cv2.imwrite(path, res)
                    print(f"Saved: {path}")
                except Exception as e:
                    print(f"Failed to save {key}: {e}")

        return img

    except Exception as e:
        print(f"[apply_censor ERROR] {e}")
        traceback.print_exc()
        return img
