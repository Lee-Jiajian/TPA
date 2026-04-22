import argparse
import json
import os

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Preview several HRSC ground-truth samples in COCO format.')
    parser.add_argument('--ann', type=str, required=True, help='Path to COCO json annotation file')
    parser.add_argument('--img_root', type=str, required=True, help='Image root referenced by json file_name')
    parser.add_argument('--out_dir', type=str, default='./results/hrsc_gt_preview', help='Directory to save preview images')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to preview')
    parser.add_argument('--category_name', type=str, default='ship', help='Category to visualize')
    parser.add_argument('--draw_rotated_fit', action='store_true', help='Fit min-area rotated box from segmentation if available')
    parser.add_argument('--draw_pseudo_regions', action='store_true',
                        help='Draw pseudo head/tail/middle rotated regions from estimated ship axis')
    parser.add_argument('--head_tail_ratio', type=float, default=0.22,
                        help='Length ratio of head/tail region over major axis')
    parser.add_argument('--middle_len_ratio', type=float, default=0.42,
                        help='Length ratio of middle strip over major axis')
    parser.add_argument('--strip_thickness_ratio', type=float, default=0.22,
                        help='Thickness ratio over minor axis for pseudo regions')
    parser.add_argument('--draw_segment_mask', action='store_true',
                        help='Draw ship mask estimated by GrabCut inside each bbox')
    parser.add_argument('--draw_segment_regions', action='store_true',
                        help='Draw head/tail/middle regions derived from segmentation mask')
    parser.add_argument('--grabcut_iters', type=int, default=3,
                        help='GrabCut iterations for mask estimation')
    return parser.parse_args()


def load_coco(ann_path):
    with open(ann_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def estimate_axis_angle_from_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None

    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)
    ys, xs = np.where(edges > 0)
    if xs.size < 20:
        return None

    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    mean = np.mean(pts, axis=0, keepdims=True)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    v = eig_vecs[:, int(np.argmax(eig_vals))]
    angle = float(np.degrees(np.arctan2(v[1], v[0])))
    return angle


def major_axis_angle_from_min_area_rect(rect):
    (_, _), (rw, rh), ra = rect
    angle = float(ra)
    if rw < rh:
        angle += 90.0
    return angle


def draw_rotated_region(img, center, size, angle_deg, color, thickness=2):
    rect = (tuple(center), tuple(size), float(angle_deg))
    box = cv2.boxPoints(rect)
    box = np.round(box).astype(np.int32)
    cv2.polylines(img, [box], True, color, thickness)


def draw_pseudo_ship_regions(img, bbox, angle_deg, head_tail_ratio, middle_len_ratio, strip_thickness_ratio):
    x, y, w, h = bbox
    cx = float(x + w * 0.5)
    cy = float(y + h * 0.5)
    major = float(max(w, h))
    minor = float(max(2.0, min(w, h)))

    head_tail_len = max(4.0, major * float(head_tail_ratio))
    middle_len = max(6.0, major * float(middle_len_ratio))
    strip_thickness = max(2.0, minor * float(strip_thickness_ratio))

    theta = np.deg2rad(angle_deg)
    ux = float(np.cos(theta))
    uy = float(np.sin(theta))

    offset = max(0.0, major * 0.5 - head_tail_len * 0.5)
    head_center = (cx + ux * offset, cy + uy * offset)
    tail_center = (cx - ux * offset, cy - uy * offset)
    mid_center = (cx, cy)

    # magenta=head, yellow=tail, cyan=middle
    draw_rotated_region(img, head_center, (head_tail_len, strip_thickness), angle_deg, (255, 0, 255), 2)
    draw_rotated_region(img, tail_center, (head_tail_len, strip_thickness), angle_deg, (0, 255, 255), 2)
    draw_rotated_region(img, mid_center, (middle_len, strip_thickness), angle_deg, (255, 255, 0), 2)


def estimate_ship_mask_grabcut(img, bbox, grabcut_iters=3):
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    x1 = max(0, min(int(round(x)), w - 1))
    y1 = max(0, min(int(round(y)), h - 1))
    x2 = max(0, min(int(round(x + bw)), w))
    y2 = max(0, min(int(round(y + bh)), h))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None

    pad_x = max(1, int((x2 - x1) * 0.08))
    pad_y = max(1, int((y2 - y1) * 0.08))
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(w, x2 + pad_x)
    ry2 = min(h, y2 + pad_y)

    roi = img[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None

    roi_h, roi_w = roi.shape[:2]
    rect = (pad_x, pad_y, max(1, x2 - x1), max(1, y2 - y1))
    rect = (
        min(max(0, rect[0]), roi_w - 2),
        min(max(0, rect[1]), roi_h - 2),
        min(max(1, rect[2]), roi_w - 1),
        min(max(1, rect[3]), roi_h - 1),
    )

    mask = np.zeros((roi_h, roi_w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, mask, rect, bgd_model, fgd_model, int(max(1, grabcut_iters)), cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

    k = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)

    if np.sum(fg) < 10:
        return None

    full = np.zeros((h, w), np.uint8)
    full[ry1:ry2, rx1:rx2] = fg
    return full


def estimate_ship_mask_threshold_fallback(img, bbox):
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    x1 = max(0, min(int(round(x)), w - 1))
    y1 = max(0, min(int(round(y)), h - 1))
    x2 = max(0, min(int(round(x + bw)), w))
    y2 = max(0, min(int(round(y + bh)), h))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None

    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Pick the mask that is not overwhelmingly background and not too sparse.
    c1 = float(np.mean(th1 > 0))
    c2 = float(np.mean(th2 > 0))
    chosen = th1 if abs(c1 - 0.5) < abs(c2 - 0.5) else th2

    k = np.ones((3, 3), np.uint8)
    chosen = cv2.morphologyEx(chosen, cv2.MORPH_OPEN, k, iterations=1)
    chosen = cv2.morphologyEx(chosen, cv2.MORPH_CLOSE, k, iterations=2)

    mask = (chosen > 0).astype(np.uint8)
    if np.sum(mask) < 10:
        return None

    full = np.zeros((h, w), np.uint8)
    full[y1:y2, x1:x2] = mask
    return full


def keep_best_component(mask, bbox):
    if mask is None or np.sum(mask) < 10:
        return None

    x, y, w, h = bbox
    cx = float(x + w * 0.5)
    cy = float(y + h * 0.5)
    bbox_area = max(1.0, float(w * h))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(np.uint8)

    best_label = -1
    best_score = -1e18
    for lab in range(1, num_labels):
        area = float(stats[lab, cv2.CC_STAT_AREA])
        if area < 10:
            continue
        cpx, cpy = centroids[lab]
        dist = np.sqrt((cpx - cx) ** 2 + (cpy - cy) ** 2)
        area_ratio = area / bbox_area
        # Prefer components near bbox center with plausible area ratio.
        score = -dist + 60.0 * min(area_ratio, 1.0)
        if score > best_score:
            best_score = score
            best_label = lab

    if best_label < 0:
        return None

    out = (labels == best_label).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    return out


def robust_ship_mask(img, bbox, grabcut_iters=3):
    m = estimate_ship_mask_grabcut(img, bbox, grabcut_iters=grabcut_iters)
    m = keep_best_component(m, bbox) if m is not None else None
    if m is not None and np.sum(m) >= 20:
        return m

    m2 = estimate_ship_mask_threshold_fallback(img, bbox)
    m2 = keep_best_component(m2, bbox) if m2 is not None else None
    if m2 is not None and np.sum(m2) >= 20:
        return m2

    return None


def segmented_regions_from_mask(mask, head_tail_ratio=0.22, middle_len_ratio=0.42, strip_thickness_ratio=0.22):
    ys, xs = np.where(mask > 0)
    if xs.size < 20:
        return None

    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    center = np.mean(pts, axis=0)
    centered = pts - center
    cov = np.cov(centered, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    major_v = eig_vecs[:, int(np.argmax(eig_vals))]
    minor_v = np.array([-major_v[1], major_v[0]], dtype=np.float32)

    # Stabilize major-axis sign by forcing a consistent orientation.
    if major_v[0] < 0:
        major_v = -major_v
    t_major = centered @ major_v
    t_minor = centered @ minor_v
    major_min, major_max = float(np.min(t_major)), float(np.max(t_major))
    minor_min, minor_max = float(np.min(t_minor)), float(np.max(t_minor))

    major_len = max(1.0, major_max - major_min)
    minor_len = max(1.0, minor_max - minor_min)
    head_tail_len = major_len * float(head_tail_ratio)
    middle_len = major_len * float(middle_len_ratio)
    strip_half = max(1.0, 0.5 * minor_len * float(strip_thickness_ratio))

    head_sel = (t_major >= (major_max - head_tail_len)) & (np.abs(t_minor) <= strip_half)
    tail_sel = (t_major <= (major_min + head_tail_len)) & (np.abs(t_minor) <= strip_half)
    mid_sel = (np.abs(t_major - 0.5 * (major_min + major_max)) <= 0.5 * middle_len) & (np.abs(t_minor) <= strip_half)

    h, w = mask.shape
    head_mask = np.zeros((h, w), np.uint8)
    tail_mask = np.zeros((h, w), np.uint8)
    mid_mask = np.zeros((h, w), np.uint8)

    p_head = pts[head_sel]
    p_tail = pts[tail_sel]
    p_mid = pts[mid_sel]
    if p_head.size > 0:
        head_mask[p_head[:, 1].astype(np.int32), p_head[:, 0].astype(np.int32)] = 1
    if p_tail.size > 0:
        tail_mask[p_tail[:, 1].astype(np.int32), p_tail[:, 0].astype(np.int32)] = 1
    if p_mid.size > 0:
        mid_mask[p_mid[:, 1].astype(np.int32), p_mid[:, 0].astype(np.int32)] = 1

    k = np.ones((3, 3), np.uint8)
    head_mask = cv2.dilate(head_mask, k, iterations=1)
    tail_mask = cv2.dilate(tail_mask, k, iterations=1)
    mid_mask = cv2.dilate(mid_mask, k, iterations=1)

    return {
        'mask': mask,
        'head': head_mask,
        'tail': tail_mask,
        'middle': mid_mask,
    }


def draw_binary_mask_overlay(img, binary_mask, color, alpha=0.35):
    if binary_mask is None:
        return
    idx = binary_mask > 0
    if not np.any(idx):
        return
    overlay = img.astype(np.float32)
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    overlay[idx] = (1.0 - alpha) * overlay[idx] + alpha * c
    img[:, :, :] = np.clip(overlay, 0, 255).astype(np.uint8)


def main():
    args = parse_args()
    coco = load_coco(args.ann)

    categories = coco.get('categories', [])
    images = coco.get('images', [])
    annotations = coco.get('annotations', [])

    cat_map = {c['name']: c['id'] for c in categories}
    if args.category_name not in cat_map:
        raise ValueError('Category {} not found in {}'.format(args.category_name, args.ann))
    target_cat = cat_map[args.category_name]

    img_map = {im['id']: im for im in images}
    ann_by_img = {}
    for ann in annotations:
        if ann.get('category_id') != target_cat:
            continue
        ann_by_img.setdefault(ann['image_id'], []).append(ann)

    selected_img_ids = sorted(list(ann_by_img.keys()))[:max(1, args.num_images)]
    os.makedirs(args.out_dir, exist_ok=True)

    saved = 0
    seg_available = 0
    pseudo_drawn = 0
    seg_region_drawn = 0
    seg_region_failed = 0
    for img_id in selected_img_ids:
        info = img_map.get(img_id)
        if info is None:
            continue
        file_name = info.get('file_name', '')
        if not file_name:
            continue

        img_path = os.path.join(args.img_root, file_name)
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            continue
        img = raw_img.copy()

        for ann in ann_by_img.get(img_id, []):
            x, y, w, h = ann['bbox']
            x1, y1 = int(round(x)), int(round(y))
            x2, y2 = int(round(x + w)), int(round(y + h))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            segs = ann.get('segmentation', [])
            seg_major_angle = None
            if isinstance(segs, list):
                all_pts = []
                for seg in segs:
                    if not isinstance(seg, list) or len(seg) < 6:
                        continue
                    pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    all_pts.append(pts)
                    cv2.polylines(img, [np.round(pts).astype(np.int32)], True, (0, 255, 0), 2)

                if args.draw_rotated_fit and all_pts:
                    pts_cat = np.concatenate(all_pts, axis=0)
                    if pts_cat.shape[0] >= 3:
                        rect = cv2.minAreaRect(pts_cat)
                        box = cv2.boxPoints(rect)
                        cv2.polylines(img, [np.round(box).astype(np.int32)], True, (255, 0, 0), 2)
                        seg_major_angle = major_axis_angle_from_min_area_rect(rect)
                        seg_available += 1

            if args.draw_pseudo_regions:
                est_angle = seg_major_angle
                if est_angle is None:
                    est_angle = estimate_axis_angle_from_crop(raw_img, x1, y1, x2, y2)
                if est_angle is None:
                    est_angle = 0.0 if w >= h else 90.0

                draw_pseudo_ship_regions(
                    img,
                    (x, y, w, h),
                    est_angle,
                    args.head_tail_ratio,
                    args.middle_len_ratio,
                    args.strip_thickness_ratio,
                )
                pseudo_drawn += 1

            if args.draw_segment_mask or args.draw_segment_regions:
                ship_mask = robust_ship_mask(raw_img, (x, y, w, h), grabcut_iters=args.grabcut_iters)
                if ship_mask is not None:
                    if args.draw_segment_mask:
                        draw_binary_mask_overlay(img, ship_mask, (0, 180, 0), alpha=0.25)
                    if args.draw_segment_regions:
                        region_pack = segmented_regions_from_mask(
                            ship_mask,
                            head_tail_ratio=args.head_tail_ratio,
                            middle_len_ratio=args.middle_len_ratio,
                            strip_thickness_ratio=args.strip_thickness_ratio,
                        )
                        if region_pack is not None:
                            draw_binary_mask_overlay(img, region_pack['head'], (255, 0, 255), alpha=0.45)
                            draw_binary_mask_overlay(img, region_pack['tail'], (0, 255, 255), alpha=0.45)
                            draw_binary_mask_overlay(img, region_pack['middle'], (255, 255, 0), alpha=0.45)
                            seg_region_drawn += 1
                        else:
                            seg_region_failed += 1
                            cv2.putText(img, 'seg-region-fail', (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    seg_region_failed += 1
                    cv2.putText(img, 'seg-mask-fail', (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        out_name = os.path.basename(file_name)
        out_path = os.path.join(args.out_dir, out_name)
        cv2.imwrite(out_path, img)
        saved += 1

    print('Saved {} GT preview images to {}'.format(saved, os.path.abspath(args.out_dir)))
    print('Legend: red=bbox, green=segmentation, blue=rotated fit (optional), magenta=head, yellow=tail, cyan=middle')
    if args.draw_rotated_fit:
        print('Annotations with segmentation-supported rotated fit: {}'.format(seg_available))
    if args.draw_pseudo_regions:
        print('Pseudo regions drawn for annotations: {}'.format(pseudo_drawn))
    if args.draw_segment_regions:
        print('Segmentation-driven regions drawn for annotations: {}'.format(seg_region_drawn))
        print('Segmentation-driven regions failed for annotations: {}'.format(seg_region_failed))


if __name__ == '__main__':
    main()
