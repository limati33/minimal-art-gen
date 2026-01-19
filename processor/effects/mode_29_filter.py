# processor/effects/mode_29_filter.py
import os
import cv2
import numpy as np

def apply_filter(img, w=None, h=None, out_dir=None, base_name=None,
                     n_colors=10,          
                     min_saturation=80,    
                     min_value=50,         
                     color_threshold=15,   
                     min_area_ratio=0.005, 
                     blur_mask=True):     

    if img is None: return None
    
    img_h, img_w = img.shape[:2]
    if w and h and (img_w != w or img_h != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    if out_dir: os.makedirs(out_dir, exist_ok=True)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray_3ch = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    valid_mask = cv2.inRange(hsv, np.array([0, min_saturation, min_value]), 
                                  np.array([180, 255, 255]))
    
    if cv2.countNonZero(valid_mask) < (img_w * img_h * 0.005):
        return gray_3ch

    ab_channels = lab[:, :, 1:3]
    masked_ab = ab_channels[valid_mask > 0]
    
    if len(masked_ab) > 100000:
        masked_ab = masked_ab[np.random.choice(len(masked_ab), 100000, replace=False)]

    training_data = np.float32(masked_ab)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(training_data, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Список для хранения всех созданных изображений
    all_results = []
    
    img_f, gray_f = img.astype(np.float32), gray_3ch.astype(np.float32)
    ab_f = ab_channels.astype(np.float32)

    for i, center in enumerate(centers):
        dist = np.sqrt(np.sum((ab_f - center)**2, axis=2))
        mask = np.clip((color_threshold - dist) / 5.0, 0.0, 1.0)
        mask[valid_mask == 0] = 0.0
        
        area_ratio = np.sum(mask) / (img_w * img_h)
        if area_ratio < min_area_ratio: continue

        if blur_mask: mask = cv2.GaussianBlur(mask, (3, 3), 0)

        res = (img_f * mask[:,:,None] + gray_f * (1.0 - mask[:,:,None])).astype(np.uint8)

        p_lab = np.uint8([[[140, center[0], center[1]]]])
        p_bgr = cv2.cvtColor(p_lab, cv2.COLOR_LAB2BGR)[0,0]
        hex_c = "{:02x}{:02x}{:02x}".format(p_bgr[2], p_bgr[1], p_bgr[0])
        
        if out_dir and base_name:
            out_filename = f"{base_name}_filter_{hex_c}.png"
            cv2.imwrite(os.path.join(out_dir, out_filename), res)
            print(f"  [Filter] Сохранен: #{hex_c} ({area_ratio*100:.1f}%)")
            
            # Сохраняем результат и его "вес" (площадь), чтобы выбрать лучший для возврата
            all_results.append((area_ratio, res))

    if not all_results:
        return gray_3ch

    # Сортируем по площади (от большего к меньшему)
    all_results.sort(key=lambda x: x[0], reverse=True)

    # ВОЗВРАЩАЕМ САМЫЙ БОЛЬШОЙ ЦВЕТ (как массив пикселей, чтобы не ломать скрипт)
    # При этом ВСЕ остальные цвета уже сохранены на диске командой imwrite выше
    return all_results[0][1]