# processor/effects/mode_stainedglass_bigregions.py
import cv2
import numpy as np
from pathlib import Path

def apply_stainedglass(img, w=None, h=None, out_dir=None, base_name=None, **kwargs):
    if img is None:
        return None

    # параметры
    desired_regions = int(kwargs.get("desired_regions", 8))
    gradient_strength = float(kwargs.get("gradient_strength", 0.45))
    lead_thickness = int(kwargs.get("lead_thickness", 3))
    saturation_boost = float(kwargs.get("saturation_boost", 1.2))
    save_aux = bool(kwargs.get("save_aux", False))
    max_iter_merge = int(kwargs.get("max_iter_merge", 2000))

    # размеры / ресайз
    if w and h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    h_img, w_img = img.shape[:2]

    # начальный шаг сетки (чтобы получить достаточно крупных фасетов)
    grid_step = kwargs.get("grid_step", None)
    if grid_step is None:
        grid_step = max(16, min(w_img, h_img) // 12)

    # размытие для стабильных цветов
    img_blur = cv2.medianBlur(img, 5)

    # --- 1) точки и Subdiv2D Voronoi как базис ---
    points = []
    for y in range(0, h_img, grid_step):
        for x in range(0, w_img, grid_step):
            nx = x + np.random.randint(0, max(1, grid_step))
            ny = y + np.random.randint(0, max(1, grid_step))
            nx = min(w_img - 1, nx)
            ny = min(h_img - 1, ny)
            points.append((nx, ny))

    rect = (0, 0, w_img, h_img)
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        try:
            subdiv.insert(p)
        except Exception:
            pass

    try:
        facets, _ = subdiv.getVoronoiFacetList([])
    except Exception:
        # fallback: простая крупная пикселизация
        small = cv2.resize(img_blur, (max(1, w_img//grid_step), max(1, h_img//grid_step)), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        return out

    # фильтруем корректные полигоны
    polygons = []
    for f in facets:
        try:
            poly = np.array(f, dtype=np.int32)
            if poly.ndim == 2 and poly.shape[0] >= 3:
                poly[:,0] = np.clip(poly[:,0], 0, w_img-1)
                poly[:,1] = np.clip(poly[:,1], 0, h_img-1)
                polygons.append(poly)
        except Exception:
            continue

    if not polygons:
        small = cv2.resize(img_blur, (max(1, w_img//grid_step), max(1, h_img//grid_step)), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        return out

    # --- 2) facet_map: для каждого пикселя - id фасета ---
    facet_map = np.full((h_img, w_img), -1, dtype=np.int32)
    for i, poly in enumerate(polygons):
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly, 255)
        facet_map[mask == 255] = i

    # удалить пустые индексы и перенумеровать
    uniq = np.unique(facet_map)
    uniq = uniq[uniq >= 0]
    remap = {old: new for new, old in enumerate(uniq)}
    new_map = -1 * np.ones_like(facet_map)
    for old, new in remap.items():
        new_map[facet_map == old] = new
    facet_map = new_map
    n_facets = int(facet_map.max()) + 1

    # --- 3) средний цвет и площадь каждой фасеты ---
    facet_means = np.zeros((n_facets, 3), dtype=np.float32)
    facet_areas = np.zeros((n_facets,), dtype=np.int32)
    for fid in range(n_facets):
        mask = (facet_map == fid).astype(np.uint8)
        area = int(mask.sum())
        facet_areas[fid] = area
        if area > 0:
            facet_means[fid] = cv2.mean(img_blur, mask=mask)[:3]
        else:
            facet_means[fid] = np.array([127.,127.,127.], dtype=np.float32)

    # --- 4) adjacency graph между фасетами (соседство) ---
    neighbors = [set() for _ in range(n_facets)]
    # сравним смещения (right, down) чтобы собрать пары границ
    a = facet_map
    right = a[:, :-1]
    left = a[:, 1:]
    mask_diff = (right != left)
    ys, xs = np.where(mask_diff)
    for y, x in zip(ys, xs):
        i = int(a[y, x])
        j = int(a[y, x+1])
        if i >= 0 and j >= 0 and i != j:
            neighbors[i].add(j)
            neighbors[j].add(i)
    down = a[:-1, :]
    up = a[1:, :]
    mask_diff2 = (down != up)
    ys, xs = np.where(mask_diff2)
    for y, x in zip(ys, xs):
        i = int(a[y, x])
        j = int(a[y+1, x])
        if i >= 0 and j >= 0 and i != j:
            neighbors[i].add(j)
            neighbors[j].add(i)

    # --- 5) начальные регионы = фасеты (каждому facet назначен region_id) ---
    facet_to_region = np.arange(n_facets, dtype=np.int32)
    region_means = facet_means.copy()
    region_areas = facet_areas.copy()
    # region adjacency (based on facets adjacency, but keyed by region ids)
    region_neighbors = {i: set(neighbors[i]) for i in range(n_facets)}

    # merge loop: пока число регионов > desired_regions, сливаем наиболее похожие соседние регионы
    current_regions = set(range(n_facets))
    iter_count = 0
    # helper to compute color distance squared
    def color_dist2(i, j):
        d = region_means[i] - region_means[j]
        return float(np.dot(d, d))

    while len(current_regions) > desired_regions and iter_count < max_iter_merge:
        iter_count += 1
        # находим пару соседних регионов с минимальным цветовым расстоянием
        best_pair = None
        best_cost = float('inf')
        # iterate regions and their neighbors
        for r in list(current_regions):
            neighs = region_neighbors.get(r, set())
            for nb in neighs:
                if nb not in current_regions or r == nb:
                    continue
                cost = color_dist2(r, nb)
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (r, nb)
        if best_pair is None:
            break
        r1, r2 = best_pair
        # merge smaller into larger to reduce updates
        if region_areas[r1] < region_areas[r2]:
            r1, r2 = r2, r1  # ensure r1 bigger

        # perform merge: r2 -> r1
        # update facet_to_region
        facet_to_region[facet_to_region == r2] = r1
        # update area and mean (weighted)
        total_area = region_areas[r1] + region_areas[r2]
        if total_area > 0:
            region_means[r1] = (region_means[r1] * region_areas[r1] + region_means[r2] * region_areas[r2]) / total_area
        region_areas[r1] = total_area
        # update neighbors: union, remove self refs and r2
        nb_union = (region_neighbors.get(r1, set()) | region_neighbors.get(r2, set())) - {r1, r2}
        region_neighbors[r1] = nb_union
        # remove r2 from neighbors and replace with r1
        for nb in list(nb_union):
            region_neighbors.setdefault(nb, set())
            region_neighbors[nb].discard(r2)
            region_neighbors[nb].add(r1)
        # delete r2
        if r2 in region_neighbors:
            del region_neighbors[r2]
        if r2 in current_regions:
            current_regions.discard(r2)
        # keep region_means[r2]/area for potential later (no need)
        # safety: if we merged enough break
        if len(current_regions) <= desired_regions:
            break

    # --- 6) построим финальную карту region_map: pixel -> final region id (0..R-1) ---
    # перезапишем facet_map -> region indices via facet_to_region
    # но facet_to_region currently maps facet_id -> region_id (which may be large numbers and sparse)
    # мы уменьшим region ids до компактного 0..R-1
    # сначала для каждой facet -> region
    facet_region = facet_to_region.copy()
    # создаём pixel_region
    pixel_region = -1 * np.ones_like(facet_map)
    mask_valid = facet_map >= 0
    pixel_region[mask_valid] = facet_region[facet_map[mask_valid]]
    unique_regions = np.unique(pixel_region[mask_valid])
    unique_regions = np.array([r for r in unique_regions if r >= 0])
    remap_regions = {old: new for new, old in enumerate(unique_regions)}
    region_count = len(unique_regions)
    region_map = -1 * np.ones_like(pixel_region)
    for old, new in remap_regions.items():
        region_map[pixel_region == old] = new

    # recompute region_means by aggregated pixels (more accurate)
    final_region_means = np.zeros((region_count, 3), dtype=np.float32)
    for rid in range(region_count):
        mask = (region_map == rid)
        if mask.sum() > 0:
            final_region_means[rid] = cv2.mean(img_blur, mask=mask.astype(np.uint8))[:3]
        else:
            final_region_means[rid] = np.array([127.,127.,127.], dtype=np.float32)

    # --- 7) внутри каждой region делаем градиент (radial по центру региона) ---
    out = np.zeros_like(img, dtype=np.uint8)
    yy, xx = np.mgrid[0:h_img, 0:w_img]
    for rid in range(region_count):
        mask = (region_map == rid)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        # координаты центра региона (centroid)
        cx = int(xs.mean())
        cy = int(ys.mean())
        # расстояние до центра (нормализуем по max расстояние внутри региона)
        dsq = (xx[mask] - cx).astype(np.float32)**2 + (yy[mask] - cy).astype(np.float32)**2
        maxd = max(1.0, float(dsq.max())**0.5)
        dnorm = (np.sqrt(dsq) / maxd)  # 0..1
        # два цвета: base и slightly shifted (lighter or darker)
        base = final_region_means[rid]
        # shift sign random to have some variety
        delta = (np.random.uniform(-0.12, 0.12, size=(3,))).astype(np.float32)
        alt = np.clip(base * (1.0 + delta), 0, 255)
        # t = dnorm, blend = base*(1-t*gs) + alt*(t*gs) where gs = gradient_strength
        gs = gradient_strength
        t = dnorm
        col_pixels = (base[None,:] * (1.0 - t[:,None]*gs) + alt[None,:] * (t[:,None]*gs))
        col_pixels = np.clip(col_pixels, 0, 255).astype(np.uint8)
        out[ys, xs] = col_pixels

    # --- 8) контуры между регионами (lead lines) ---
    # boundary where any neighbour pixel has different region id
    region_uint = region_map.copy().astype(np.int32)
    boundary = np.zeros((h_img, w_img), dtype=np.uint8)
    # compare shifts
    for dy, dx in ((0,1),(1,0),(0,-1),(-1,0)):
        shifted = np.full_like(region_uint, -1)
        if dy == 0 and dx == 1:
            shifted[:, :-1] = region_uint[:, 1:]
        elif dy == 1 and dx == 0:
            shifted[:-1, :] = region_uint[1:, :]
        elif dy == 0 and dx == -1:
            shifted[:, 1:] = region_uint[:, :-1]
        else:
            shifted[1:, :] = region_uint[:-1, :]
        boundary |= ((shifted != region_uint) & (region_uint >= 0) & (shifted >= 0)).astype(np.uint8)

    # thicken the boundary and darken
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lead_thickness, lead_thickness))
    bmask = cv2.dilate(boundary, kernel, iterations=1).astype(np.uint8)
    out[bmask.astype(bool)] = np.array([18,18,18], dtype=np.uint8)

    # --- 9) текстура + насыщенность + bloom ---
    # легкий шум
    noise = (np.random.normal(0, 6, (h_img, w_img, 3))).astype(np.int16)
    out = cv2.add(out.astype(np.int16), noise, dtype=cv2.CV_8U)
    # насыщенность
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1] * saturation_boost, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # bloom
    glow = cv2.GaussianBlur(out, (0,0), sigmaX=3)
    out = cv2.addWeighted(out, 0.82, glow, 0.28, 0)

    # сохранить если нужно
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(out_dir)/f"stainedglass_big_{Path(base_name).stem}.png"), out)
        if save_aux:
            viz_fac = ((facet_map % 256)).astype(np.uint8)
            cv2.imwrite(str(Path(out_dir)/f"stainedglass_big_facets_{Path(base_name).stem}.png"), cv2.cvtColor(viz_fac, cv2.COLOR_GRAY2BGR))
            viz_reg = ((region_map % 256)).astype(np.uint8)
            cv2.imwrite(str(Path(out_dir)/f"stainedglass_big_regions_{Path(base_name).stem}.png"), cv2.cvtColor(viz_reg, cv2.COLOR_GRAY2BGR))

    return out
