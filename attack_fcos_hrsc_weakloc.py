import argparse
import json
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from mmdetection.mmdet.apis.inference import init_detector as mmdetection_init_detector
from mmdetection.mmdet.apis.inference import (
    inference_single_attack_single_init as fcos_attack_init,
)
from mmdetection.mmdet.apis.inference import (
    inference_single_attack_single_mt as fcos_attack_mt,
)


def to_tensor(img, device="cuda:0"):
    assert isinstance(img, np.ndarray), "img must be ndarray"
    tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().to(device)
    return tensor.unsqueeze(0)


def parse_float_expr(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)


def ensure_cv2_image(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.dim() == 4:
            img = img.squeeze(0)
        if img.dim() == 3:
            img = img.permute(1, 2, 0).contiguous().numpy()
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be numpy array or tensor")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def clamp_box(box, h, w):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def normalize_map(arr):
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(
        arr, nan=0.0, posinf=0.0, neginf=0.0
    )  # 闃叉NaN瀵艰嚧鍙鍖栧穿婧?
    vmax = float(np.max(arr))
    if vmax <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return arr / vmax


def stretch_map(arr, low_percentile=90.0, high_percentile=99.8):
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(vals, low_percentile))
    hi = float(np.percentile(vals, high_percentile))
    if hi <= lo + 1e-12:
        return normalize_map(arr)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0)

# 构建边缘强度热力图
def build_fallback_weak_map(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    weak = cv2.magnitude(gx, gy)
    return normalize_map(weak)


def pick_salient_points(roi_map, percentile=85, min_points=40):
    flat = roi_map.reshape(-1)
    if flat.size == 0:
        return None, None
    if float(np.max(flat)) <= 1e-12:
        return None, None
    k = max(int(flat.size * (100.0 - percentile) / 100.0), min_points)
    k = min(k, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    ys, xs = np.unravel_index(idx, roi_map.shape)
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    weights = flat[idx].astype(np.float32)
    return pts, weights


def weighted_pca(points, weights=None):
    if points is None or len(points) < 3:
        return None, None
    points = points.astype(np.float32)
    if weights is None:
        center = np.mean(points, axis=0)
        centered = points - center
        cov = np.cov(centered, rowvar=False)
    else:
        weights = np.maximum(weights.astype(np.float32), 1e-6)
        weights = weights / np.sum(weights)
        center = np.sum(points * weights[:, None], axis=0)
        centered = points - center
        cov = (centered * weights[:, None]).T @ centered
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    major_v = eig_vecs[:, int(np.argmax(eig_vals))]
    if major_v[0] < 0:
        major_v = -major_v
    minor_v = np.array([-major_v[1], major_v[0]], dtype=np.float32)
    return major_v.astype(np.float32), minor_v.astype(np.float32)


def largest_connected_component(binary_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return binary_mask.astype(np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def build_object_prior_from_weak_map(roi, weak_percentile=85):
    vals = roi[roi > 0]
    if vals.size < 10:
        return None
    thr = float(np.percentile(vals, weak_percentile))
    obj = (roi >= thr).astype(np.uint8)
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    obj = cv2.morphologyEx(
        obj, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2
    )
    obj = largest_connected_component(obj)
    if np.sum(obj) < 20:
        return None
    return obj


def score_grid_cells(points, values, major_v, minor_v, origin, grid_n=4):
    proj_major = (points - origin) @ major_v
    proj_minor = (points - origin) @ minor_v
    major_min, major_max = float(np.min(proj_major)), float(np.max(proj_major))
    minor_min, minor_max = float(np.min(proj_minor)), float(np.max(proj_minor))
    major_span = max(1e-6, major_max - major_min)
    minor_span = max(1e-6, minor_max - minor_min)
    scores = np.zeros((grid_n, grid_n), dtype=np.float32)
    counts = np.zeros((grid_n, grid_n), dtype=np.float32)
    maj_idx = np.clip(
        ((proj_major - major_min) / major_span * grid_n).astype(np.int32), 0, grid_n - 1
    )
    min_idx = np.clip(
        ((proj_minor - minor_min) / minor_span * grid_n).astype(np.int32), 0, grid_n - 1
    )
    for i in range(points.shape[0]):
        r = int(maj_idx[i])
        c = int(min_idx[i])
        scores[r, c] += float(values[i])
        counts[r, c] += 1.0
    scores = scores / np.maximum(counts, 1.0)
    return scores


def build_weakloc_regions(
    boxes,
    weak_map,
    head_tail_ratio=0.22,
    middle_len_ratio=0.42,
    strip_thickness_ratio=0.22,
    weak_percentile=85,
    min_points=40,
    grid_n=4,
):
    h, w = weak_map.shape[:2]
    head_mask = np.zeros((h, w), dtype=np.uint8)
    tail_mask = np.zeros((h, w), dtype=np.uint8)
    middle_mask = np.zeros((h, w), dtype=np.uint8)
    union_mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        cb = clamp_box(box[:4], h, w)
        if cb is None:
            continue
        x1, y1, x2, y2 = cb
        roi = weak_map[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        obj_mask = build_object_prior_from_weak_map(
            roi, weak_percentile=weak_percentile
        )
        if obj_mask is None:
            pts, weights = pick_salient_points(
                roi, percentile=weak_percentile, min_points=min_points
            )
            if pts is None or len(pts) < 3:
                # fallback to bbox geometry
                bw, bh = x2 - x1, y2 - y1
                if bw >= bh:
                    center_thickness = max(2, int(bh * strip_thickness_ratio))
                    center_len = max(4, int(bw * middle_len_ratio))
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    middle_mask[
                        max(y1, cy - center_thickness // 2) : min(
                            y2, cy + center_thickness // 2
                        ),
                        max(x1, cx - center_len // 2) : min(x2, cx + center_len // 2),
                    ] = 1
                    head_mask[y1:y2, x1 : x1 + max(2, int(bw * head_tail_ratio))] = 1
                    tail_mask[y1:y2, x2 - max(2, int(bw * head_tail_ratio)) : x2] = 1
                else:
                    center_thickness = max(2, int(bw * strip_thickness_ratio))
                    center_len = max(4, int(bh * middle_len_ratio))
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    middle_mask[
                        max(y1, cy - center_len // 2) : min(y2, cy + center_len // 2),
                        max(x1, cx - center_thickness // 2) : min(
                            x2, cx + center_thickness // 2
                        ),
                    ] = 1
                    head_mask[y1 : y1 + max(2, int(bh * head_tail_ratio)), x1:x2] = 1
                    tail_mask[y2 - max(2, int(bh * head_tail_ratio)) : y2, x1:x2] = 1
                union_mask[y1:y2, x1:x2] = np.clip(union_mask[y1:y2, x1:x2] + 1, 0, 1)
                continue
            obj_pts, obj_weights = pts, weights
        else:
            ys, xs = np.where(obj_mask > 0)
            obj_pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
            obj_weights = roi[ys, xs].astype(np.float32)
        if obj_pts is None or len(obj_pts) < 3:
            bw, bh = x2 - x1, y2 - y1
            if bw >= bh:
                center_thickness, center_len = max(
                    2, int(bh * strip_thickness_ratio)
                ), max(4, int(bw * middle_len_ratio))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                middle_mask[
                    max(y1, cy - center_thickness // 2) : min(
                        y2, cy + center_thickness // 2
                    ),
                    max(x1, cx - center_len // 2) : min(x2, cx + center_len // 2),
                ] = 1
                head_mask[y1:y2, x1 : x1 + max(2, int(bw * head_tail_ratio))] = 1
                tail_mask[y1:y2, x2 - max(2, int(bw * head_tail_ratio)) : x2] = 1
            else:
                center_thickness, center_len = max(
                    2, int(bw * strip_thickness_ratio)
                ), max(4, int(bh * middle_len_ratio))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                middle_mask[
                    max(y1, cy - center_len // 2) : min(y2, cy + center_len // 2),
                    max(x1, cx - center_thickness // 2) : min(
                        x2, cx + center_thickness // 2
                    ),
                ] = 1
                head_mask[y1 : y1 + max(2, int(bh * head_tail_ratio)), x1:x2] = 1
                tail_mask[y2 - max(2, int(bh * head_tail_ratio)) : y2, x1:x2] = 1
            union_mask[y1:y2, x1:x2] = np.clip(union_mask[y1:y2, x1:x2] + 1, 0, 1)
            continue
        major_v, minor_v = weighted_pca(obj_pts, obj_weights)
        if major_v is None:
            continue
        origin = np.mean(obj_pts, axis=0)
        proj_major = (obj_pts - origin) @ major_v
        proj_minor = (obj_pts - origin) @ minor_v
        major_min, major_max = float(np.min(proj_major)), float(np.max(proj_major))
        minor_min, minor_max = float(np.min(proj_minor)), float(np.max(proj_minor))
        major_len = max(1.0, major_max - major_min)
        minor_len = max(1.0, minor_max - minor_min)
        scores = score_grid_cells(
            obj_pts, obj_weights, major_v, minor_v, origin, grid_n=grid_n
        )
        if scores is None:
            continue
        topk = max(1, (grid_n * grid_n) // 2)
        cell_scores = []
        cell_major_pos = np.linspace(0.0, 1.0, grid_n, endpoint=False) + 0.5 / grid_n
        for r in range(grid_n):
            for c in range(grid_n):
                cell_scores.append(
                    (float(scores[r, c]), r, c, float(cell_major_pos[r]))
                )
        # 鏍规嵁涓昏酱浣嶇疆鍒掑垎澶淬€佷腑銆佸熬鍖哄煙鍊欓€夌綉鏍?
        head_candidates = sorted(
            [it for it in cell_scores if it[3] >= (1.0 - head_tail_ratio)],
            key=lambda x: x[0],
            reverse=True,
        )[:topk]
        tail_candidates = sorted(
            [it for it in cell_scores if it[3] <= head_tail_ratio],
            key=lambda x: x[0],
            reverse=True,
        )[:topk]
        middle_candidates = sorted(
            [it for it in cell_scores if abs(it[3] - 0.5) <= (middle_len_ratio * 0.5)],
            key=lambda x: x[0],
            reverse=True,
        )[: max(1, topk // 2)]
        full_head = np.zeros((h, w), dtype=np.uint8)
        full_tail = np.zeros((h, w), dtype=np.uint8)
        full_middle = np.zeros((h, w), dtype=np.uint8)

        def fill_cells(target_mask, selected_cells):
            for _, r, c, _ in selected_cells:
                r0, r1 = r / float(grid_n), (r + 1) / float(grid_n)
                c0, c1 = c / float(grid_n), (c + 1) / float(grid_n)
                cell = (
                    major_min + r0 * major_len,
                    major_min + r1 * major_len,
                    minor_min + c0 * minor_len,
                    minor_min + c1 * minor_len,
                )
                inside = (
                    (proj_major >= cell[0])
                    & (proj_major < cell[1])
                    & (proj_minor >= cell[2])
                    & (proj_minor < cell[3])
                )
                pts_sel = obj_pts[inside]
                if pts_sel.size > 0:
                    yy = np.clip((y1 + pts_sel[:, 1].astype(np.int32)), 0, h - 1)
                    xx = np.clip((x1 + pts_sel[:, 0].astype(np.int32)), 0, w - 1)
                    target_mask[yy, xx] = 1

        fill_cells(full_head, head_candidates)
        fill_cells(full_tail, tail_candidates)
        fill_cells(full_middle, middle_candidates)
        # 鎵╁紶閿佸畾鍖哄煙锛屼娇寰楀鎶楁壈鍔ㄦ洿鏈夋晥
        full_head = cv2.dilate(
            full_head, np.ones((3, 3), np.uint8), iterations=2
        )  # 鎵╁紶2娆＄‘淇濊鐩?
        full_tail = cv2.dilate(full_tail, np.ones((3, 3), np.uint8), iterations=2)
        full_middle = cv2.dilate(full_middle, np.ones((3, 3), np.uint8), iterations=2)
        head_mask = np.maximum(head_mask, full_head)
        tail_mask = np.maximum(tail_mask, full_tail)
        middle_mask = np.maximum(middle_mask, full_middle)
        union_mask[y1:y2, x1:x2] = np.clip(union_mask[y1:y2, x1:x2] + 1, 0, 1)
    return {
        "head": head_mask,
        "tail": tail_mask,
        "middle": middle_mask,
        "union": union_mask,
    }


# ================= 鍙鍖栨ā鍧椾慨澶嶄笌澧炲己 =================
def make_heatmap_preview(heatmap):
    heatmap_norm = stretch_map(heatmap)
    return cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def make_heatmap_overlay(img_bgr, heatmap, alpha=0.5):
    img = ensure_cv2_image(img_bgr).astype(np.float32)
    heat = make_heatmap_preview(heatmap).astype(np.float32)
    overlay = cv2.addWeighted(img, alpha, heat, 1.0 - alpha, 0.0)
    return np.clip(overlay, 0, 255).astype(np.uint8)


# 銆愭柊澧炪€戝垎鑹插彲瑙嗗寲锛氱孩=澶撮儴(鑸瑰ご/鑸瑰熬)锛岃摑=灏鹃儴(鑸瑰熬/鑸瑰ご)锛岀豢=涓儴(鑸板矝)
def make_structure_preview(img_bgr, region_masks):
    img = ensure_cv2_image(img_bgr).copy()
    overlay = img.astype(np.float32)
    # 绾㈣壊閫氶亾 - Head
    overlay[region_masks["head"] > 0] = [0, 0, 255]
    # 缁胯壊閫氶亾 - Middle
    overlay[region_masks["middle"] > 0] = [0, 255, 0]
    # 钃濊壊閫氶亾 - Tail
    overlay[region_masks["tail"] > 0] = [255, 0, 0]
    # 娣峰悎鏄剧ず
    vis = cv2.addWeighted(img.astype(np.float32), 0.5, overlay, 0.5, 0.0)
    return np.clip(vis, 0, 255).astype(np.uint8)


def make_mask_preview(mask):
    preview = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    preview[mask > 0] = (0, 0, 255)
    return preview


def save_visuals(out_dir, img_name, img_bgr, adv_img, weak_map, region_masks):
    base, _ = os.path.splitext(img_name)
    os.makedirs(out_dir, exist_ok=True)
    # 1. 鐑姏鍥惧彲瑙嗗寲 (淇缁村害瀵艰嚧鍏ㄩ粦鐨勯棶棰?
    cv2.imwrite(
        os.path.join(out_dir, "{}_weakmap.png".format(base)),
        make_heatmap_preview(weak_map),
    )
    cv2.imwrite(
        os.path.join(out_dir, "{}_weakmap_overlay.png".format(base)),
        make_heatmap_overlay(img_bgr, weak_map),
    )
    # 2. 缁撴瀯閿佸畾鍙鍖?(楠岃瘉鏄惁绮剧‘閿佸畾閲嶇偣鍖哄煙)
    cv2.imwrite(
        os.path.join(out_dir, "{}_structure_lock.png".format(base)),
        make_structure_preview(img_bgr, region_masks),
    )
    cv2.imwrite(
        os.path.join(out_dir, "{}_mask.png".format(base)),
        make_mask_preview(region_masks["union"]),
    )
    if adv_img is None:
        return
    diff = np.abs(adv_img.astype(np.int16) - img_bgr.astype(np.int16)).astype(np.uint8)
    heat = cv2.applyColorMap(
        np.clip(np.max(diff, axis=2) * 25, 0, 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    cv2.imwrite(os.path.join(out_dir, "{}_perturb.png".format(base)), heat)


def calc_iou_matrix(bbox1, bbox2):
    if len(bbox1) == 0 or len(bbox2) == 0:
        return np.zeros((len(bbox1), len(bbox2)), dtype=np.float32)
    b1 = np.array(bbox1, dtype=np.float32)
    b2 = np.array(bbox2, dtype=np.float32)
    xmin1, ymin1, xmax1, ymax1 = np.split(b1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2 = np.split(b2, 4, axis=-1)
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))
    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    inter = h * w
    union = area1 + np.squeeze(area2, axis=-1) - inter
    return inter / np.maximum(union, 1e-6)


def pseudo_angle_from_box(box):
    x1, y1, x2, y2 = box[:4]
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return float(np.degrees(np.arctan2(h, w)))


def box_center(box):
    x1, y1, x2, y2 = box[:4]
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


def summarize_metrics(init_boxes, adv_boxes, score_thr=0.3):
    init = [b[:4] for b in init_boxes if b[-1] > score_thr]
    adv = [b[:4] for b in adv_boxes if b[-1] > score_thr]
    init_n, adv_n = len(init), len(adv)
    disappear_rate = float(max(init_n - adv_n, 0) / max(init_n, 1))
    if init_n == 0:
        return {
            "disappear_rate": 0.0,
            "iou_drop": 0.0,
            "angle_dev": 0.0,
            "center_shift_px": 0.0,
            "center_shift_norm": 0.0,
            "init_count": 0,
            "adv_count": adv_n,
        }
    ious = calc_iou_matrix(init, adv)
    if ious.size == 0:
        mean_iou, angle_dev, center_shift_px, center_shift_norm = 0.0, 90.0, 0.0, 0.0
    else:
        max_idx = np.argmax(ious, axis=1)
        best_iou = ious[np.arange(ious.shape[0]), max_idx]
        mean_iou = float(np.mean(best_iou))
        devs, shifts_px, shifts_norm = [], [], []
        for i, j in enumerate(max_idx):
            a1, a2 = pseudo_angle_from_box(init[i]), pseudo_angle_from_box(adv[j])
            devs.append(abs(a1 - a2))
            c1x, c1y = box_center(init[i])
            c2x, c2y = box_center(adv[j])
            dist = float(np.sqrt((c2x - c1x) ** 2 + (c2y - c1y) ** 2))
            shifts_px.append(dist)
            x1, y1, x2, y2 = init[i]
            diag = float(np.sqrt(max(1e-6, (x2 - x1) ** 2 + (y2 - y1) ** 2)))
            shifts_norm.append(dist / diag)
        angle_dev = float(np.mean(devs)) if devs else 0.0
        center_shift_px = float(np.mean(shifts_px)) if shifts_px else 0.0
        center_shift_norm = float(np.mean(shifts_norm)) if shifts_norm else 0.0
    return {
        "disappear_rate": disappear_rate,
        "iou_drop": float(1.0 - mean_iou),
        "angle_dev": angle_dev,
        "center_shift_px": center_shift_px,
        "center_shift_norm": center_shift_norm,
        "init_count": init_n,
        "adv_count": adv_n,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="HRSC ship attack guided by weakly-supervised localization"
    )
    parser.add_argument(
        "--config", type=str, default="./mmdetection/configs/fcos/fcos_hrsc.py"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="./mmdetection/weight/epoch_12.pth"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/cloud/cloud-ssd1/collected_files/ljj/dataset/HRSC-coco/val",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_root", type=str, default="./results")
    parser.add_argument("--save_name", type=str, default="hrsc_weakloc_attack")
    parser.add_argument("--no_save_images", action="store_true")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--score_thr", type=float, default=0.3)
    parser.add_argument("--alpha_norm", type=parse_float_expr, default=2.0 / 255.0)
    parser.add_argument("--epsilon_norm", type=parse_float_expr, default=10.0 / 255.0)
    parser.add_argument("--weak_percentile", type=float, default=85.0)
    parser.add_argument("--min_points", type=int, default=40)
    parser.add_argument("--head_tail_ratio", type=float, default=0.22)
    parser.add_argument("--middle_len_ratio", type=float, default=0.42)
    parser.add_argument("--strip_thickness_ratio", type=float, default=0.22)
    parser.add_argument("--mask_dilate", type=int, default=0)
    parser.add_argument("--grid_n", type=int, default=4)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic
    model = mmdetection_init_detector(
        config=args.config, checkpoint=args.checkpoint, device=args.device
    )
    img_names = sorted(os.listdir(args.data_root))
    start = max(0, args.start_idx)
    end = len(img_names) if args.end_idx < 0 else min(len(img_names), args.end_idx)
    img_names = img_names[start:end]
    save_img_dir = os.path.abspath(
        os.path.join(args.save_root, args.save_name, "images")
    )
    save_vis_dir = os.path.abspath(os.path.join(args.save_root, args.save_name, "vis"))
    save_txt_dir = os.path.join("./results_txt", args.save_name)
    os.makedirs(save_txt_dir, exist_ok=True)
    if not args.no_save_images:
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_vis_dir, exist_ok=True)
    per_img, per_img_linf = [], []
    pbar = tqdm(img_names, total=len(img_names))
    for img_name in pbar:
        img_path = os.path.join(args.data_root, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_tensor = to_tensor(img_bgr, args.device)
        img_tensor.requires_grad = True
        init_boxes, init_labels, init_det = fcos_attack_init(
            img_path, model, img_tensor, img_tensor
        )
        if init_boxes.size == 0:
            continue
        # === 核心修复：弱监督定位图/梯度的维度处理 ===
        weak_noise, _, _, _, _ = fcos_attack_mt(
            img_path, model, img_tensor, img_tensor, init_det, args.threshold
        )
        
        # 避免传入的梯度是 4 维（如 (1, 3, H, W)），这会由于mean和transpose判断失败导致弱定位仅剩“一条线(Width×1×1)”
        if getattr(weak_noise, "ndim", 0) == 4:
            weak_noise = weak_noise.squeeze(0)
        if getattr(weak_noise, "ndim", 0) == 3 and weak_noise.shape[0] == 3:
            weak_noise = weak_noise.transpose(1, 2, 0)  # CHW -> HWC
            
        # 防止得到 nan 后影响可视化或计算，将其替换为0
        weak_noise = np.nan_to_num(weak_noise, nan=0.0, posinf=0.0, neginf=0.0)
        
        weak_map = normalize_map(np.mean(np.abs(weak_noise), axis=2))
        
        # 如果模型内部做过长宽放缩导致与原图尺寸不一致，强制映射回原图对应大小
        target_h, target_w = img_bgr.shape[0], img_bgr.shape[1]
        if weak_map.shape[0] != target_h or weak_map.shape[1] != target_w:
            weak_map = cv2.resize(weak_map, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
        if float(np.max(weak_map)) <= 1e-12:
            weak_map = build_fallback_weak_map(img_bgr)
            
        # 过滤低置信度的边界框以避免干扰
        valid_init_boxes = [box for box in init_boxes if box[-1] > args.score_thr]
        if not valid_init_boxes:
            continue
            
        region_masks = build_weakloc_regions(
            valid_init_boxes,
            weak_map,
            head_tail_ratio=args.head_tail_ratio,
            middle_len_ratio=args.middle_len_ratio,
            strip_thickness_ratio=args.strip_thickness_ratio,
            weak_percentile=args.weak_percentile,
            min_points=args.min_points,
            grid_n=args.grid_n,
        )

        attack_mask_2d = region_masks["union"].astype(np.uint8)
        
        # 始终初始化扰动和参数
        if args.mask_dilate > 0:
            k = np.ones((3, 3), np.uint8)
            attack_mask_2d = cv2.dilate(attack_mask_2d, k, iterations=args.mask_dilate)
            
        attack_mask = attack_mask_2d.astype(np.float32)[..., None]
        original_img = img_bgr.astype(np.float32)
        perturb = np.zeros_like(original_img, dtype=np.float32)
        epsilon_px = float(args.epsilon_norm * 255.0)
        alpha_px = float(args.alpha_norm * 255.0)
        adv_img = None
        adv_boxes = init_boxes
        
        for _ in range(args.iters):
            noise, adv_boxes, _, _, _ = fcos_attack_mt(
                img_path, model, img_tensor, img_tensor, init_det, args.threshold
            )
            # 严格确保noise处理后的维度为 (H, W, 3) 且不要受到其他误导处理
            if getattr(noise, "ndim", 0) == 4:
                noise = noise[0]
            if getattr(noise, "ndim", 0) == 3 and noise.shape[0] == 3:
                noise = noise.transpose(1, 2, 0)
            
            # 使用np.nan_to_num防止因为提取出NaN或无穷大的导数而毁掉结果图像
            noise = np.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)
            
            signed_noise = np.sign(noise).astype(np.float32)
            
            if np.all(signed_noise == 0):
                break
                
            # 不要反转通道！确保梯度流跟图像是在同一通道排布下的（全是BGR或者同为RGB）
            grad_tensor_bgr = signed_noise
            
            # 强制统一宽高尺寸以防出现错位（防范从张量中挤出后导致与原图尺寸不一致的报错或静默错误）
            if grad_tensor_bgr.shape[:2] != attack_mask.shape[:2]:
                grad_tensor_bgr = cv2.resize(
                    grad_tensor_bgr, 
                    (attack_mask.shape[1], attack_mask.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
                
            delta = grad_tensor_bgr * attack_mask
            perturb = np.clip(perturb - alpha_px * delta, -epsilon_px, epsilon_px)
            adv_img = np.clip(original_img + perturb, 0, 255).astype(np.uint8)
            img_tensor = to_tensor(adv_img, args.device)
            img_tensor.requires_grad = True
        if adv_img is None:
            adv_img = ensure_cv2_image(img_tensor)
        metrics = summarize_metrics(init_boxes, adv_boxes, score_thr=args.score_thr)
        metrics["image"] = img_name
        diff_norm = (
            np.abs(adv_img.astype(np.float32) - img_bgr.astype(np.float32)) / 255.0
        )
        per_img_linf.append(float(np.max(diff_norm)))
        per_img.append(metrics)
        pbar.set_description(
            "{} [weakloc] disp={:.3f} iou_drop={:.3f} shift={:.2f}px".format(
                img_name,
                metrics["disappear_rate"],
                metrics["iou_drop"],
                metrics["center_shift_px"],
            )
        )
        if not args.no_save_images:
            cv2.imwrite(os.path.join(save_img_dir, img_name), ensure_cv2_image(adv_img))
            save_visuals(
                save_vis_dir, img_name, img_bgr, adv_img, weak_map, region_masks
            )
    if not per_img:
        print("No valid images were processed.")
        return
    mean_disappear_rate = float(np.mean([x["disappear_rate"] for x in per_img]))
    mean_iou_drop = float(np.mean([x["iou_drop"] for x in per_img]))
    mean_angle_dev = float(np.mean([x["angle_dev"] for x in per_img]))
    mean_center_shift_px = float(np.mean([x["center_shift_px"] for x in per_img]))
    mean_center_shift_norm = float(np.mean([x["center_shift_norm"] for x in per_img]))
    mean_linf = float(np.mean(per_img_linf)) if per_img_linf else 0.0
    summary = {
        "attack_type": "weakly_supervised_ship_attack",
        "epsilon_norm": args.epsilon_norm,
        "alpha_norm": args.alpha_norm,
        "weak_percentile": args.weak_percentile,
        "min_points": args.min_points,
        "head_tail_ratio": args.head_tail_ratio,
        "middle_len_ratio": args.middle_len_ratio,
        "strip_thickness_ratio": args.strip_thickness_ratio,
        "num_images": len(per_img),
        "DisappearRate": mean_disappear_rate,
        "IoU_drop": mean_iou_drop,
        "Angle_dev": mean_angle_dev,
        "CenterShiftPx": mean_center_shift_px,
        "CenterShiftNorm": mean_center_shift_norm,
        "MeanLinf": mean_linf,
        "iters": args.iters,
        "per_image": per_img,
    }
    out_json = os.path.join(save_txt_dir, "weakloc_summary.json")
    out_txt = os.path.join(save_txt_dir, "weakloc_summary.txt")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("AttackType: weakly_supervised_ship_attack\n")
        f.write("EpsilonNorm: {:.8f}\n".format(args.epsilon_norm))
        f.write("AlphaNorm: {:.8f}\n".format(args.alpha_norm))
        f.write("WeakPercentile: {}\n".format(args.weak_percentile))
        f.write("MinPoints: {}\n".format(args.min_points))
        f.write("HeadTailRatio: {}\n".format(args.head_tail_ratio))
        f.write("MiddleLenRatio: {}\n".format(args.middle_len_ratio))
        f.write("StripThicknessRatio: {}\n".format(args.strip_thickness_ratio))
        f.write("NumImages: {}\n".format(len(per_img)))
        f.write("DisappearRate: {:.6f}\n".format(mean_disappear_rate))
        f.write("IoU_drop: {:.6f}\n".format(mean_iou_drop))
        f.write("CenterShiftPx: {:.4f}\n".format(mean_center_shift_px))
        f.write("CenterShiftNorm: {:.6f}\n".format(mean_center_shift_norm))
        f.write("Angle_dev: {:.4f}\n".format(mean_angle_dev))
        f.write("MeanLinf: {:.6f}\n".format(mean_linf))
        f.write("\nPer-image details:\n")
        for item in per_img:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Saved summary to {}".format(out_json))
    print("Saved text report to {}".format(out_txt))
    if not args.no_save_images:
        print("Attack images saved to {}".format(save_img_dir))
        print("Visuals saved to {}".format(save_vis_dir))
    print(
        "Weakloc attack done | N={} | DisappearRate={:.4f} | IoU_drop={:.4f} | epsilon=10/255".format(
            len(per_img), mean_disappear_rate, mean_iou_drop
        )
    )


if __name__ == "__main__":
    main()
