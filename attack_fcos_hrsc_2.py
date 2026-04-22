import argparse
import json
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from mmdetection.mmdet.apis.inference import init_detector as mmdetection_init_detector
from mmdetection.mmdet.apis.inference import inference_single_attack_single_init as fcos_attack_init
from mmdetection.mmdet.apis.inference import inference_single_attack_single_mt as fcos_attack_mt


def to_tensor(img, device='cuda:0'):
    assert isinstance(img, np.ndarray), 'img must be ndarray'
    t = torch.from_numpy(img.transpose((2, 0, 1))).float().to(device)
    return t.unsqueeze(0)


def ensure_cv2_image(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.dim() == 4:
            img = img.squeeze(0)
        if img.dim() == 3:
            img = img.permute(1, 2, 0).contiguous().numpy()
    if not isinstance(img, np.ndarray):
        raise TypeError('img must be numpy array or tensor')
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


def shrink_rect(rect, shrink_ratio=0.2):
    x1, y1, x2, y2 = rect
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    dx = int(width * shrink_ratio / 2.0)
    dy = int(height * shrink_ratio / 2.0)
    nx1 = min(x2 - 1, x1 + dx)
    ny1 = min(y2 - 1, y1 + dy)
    nx2 = max(nx1 + 1, x2 - dx)
    ny2 = max(ny1 + 1, y2 - dy)
    return nx1, ny1, nx2, ny2


def pick_best_rect_by_saliency(saliency, x1, y1, x2, y2, grid_rows=2, grid_cols=2):
    best_rect = None
    best_score = -1.0
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    for row in range(grid_rows):
        for col in range(grid_cols):
            rx1 = int(x1 + col * width / grid_cols)
            rx2 = int(x1 + (col + 1) * width / grid_cols)
            ry1 = int(y1 + row * height / grid_rows)
            ry2 = int(y1 + (row + 1) * height / grid_rows)
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            score = float(np.mean(saliency[ry1:ry2, rx1:rx2]))
            if score > best_score:
                best_score = score
                best_rect = (rx1, ry1, rx2, ry2)
    if best_rect is None:
        best_rect = (x1, y1, x2, y2)
    return shrink_rect(best_rect, shrink_ratio=0.2)


def build_ship_attack_regions(boxes, img_bgr, middle_ratio=0.18, end_ratio=0.28, grid_rows=2, grid_cols=2):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    saliency = cv2.magnitude(gx, gy)

    middle_mask = np.zeros((h, w), dtype=np.float32)
    head_mask = np.zeros((h, w), dtype=np.float32)
    tail_mask = np.zeros((h, w), dtype=np.float32)

    for box in boxes:
        cb = clamp_box(box[:4], h, w)
        if cb is None:
            continue
        x1, y1, x2, y2 = cb
        width = x2 - x1
        height = y2 - y1
        if width <= 1 or height <= 1:
            continue

        if width >= height:
            center_thickness = max(2, int(height * middle_ratio))
            center_len = max(4, int(width * 0.52))
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            mx1 = max(x1, cx - center_len // 2)
            mx2 = min(x2, cx + center_len // 2)
            my1 = max(y1, cy - center_thickness // 2)
            my2 = min(y2, cy + center_thickness // 2)
            middle_mask[my1:my2, mx1:mx2] = 1.0

            head_x2 = min(x2, x1 + max(4, int(width * end_ratio)))
            tail_x1 = max(x1, x2 - max(4, int(width * end_ratio)))
            head_rect = pick_best_rect_by_saliency(saliency, x1, y1, head_x2, y2, grid_rows=grid_rows, grid_cols=grid_cols)
            tail_rect = pick_best_rect_by_saliency(saliency, tail_x1, y1, x2, y2, grid_rows=grid_rows, grid_cols=grid_cols)
        else:
            center_thickness = max(2, int(width * middle_ratio))
            center_len = max(4, int(height * 0.52))
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            mx1 = max(x1, cx - center_thickness // 2)
            mx2 = min(x2, cx + center_thickness // 2)
            my1 = max(y1, cy - center_len // 2)
            my2 = min(y2, cy + center_len // 2)
            middle_mask[my1:my2, mx1:mx2] = 1.0

            head_y2 = min(y2, y1 + max(4, int(height * end_ratio)))
            tail_y1 = max(y1, y2 - max(4, int(height * end_ratio)))
            head_rect = pick_best_rect_by_saliency(saliency, x1, y1, x2, head_y2, grid_rows=grid_rows, grid_cols=grid_cols)
            tail_rect = pick_best_rect_by_saliency(saliency, x1, tail_y1, x2, y2, grid_rows=grid_rows, grid_cols=grid_cols)

        hx1, hy1, hx2, hy2 = head_rect
        tx1, ty1, tx2, ty2 = tail_rect
        head_mask[hy1:hy2, hx1:hx2] = 1.0
        tail_mask[ty1:ty2, tx1:tx2] = 1.0

    total_mask = np.clip(middle_mask + head_mask + tail_mask, 0.0, 1.0)
    return {
        'middle': middle_mask,
        'head': head_mask,
        'tail': tail_mask,
        'total': total_mask,
    }


def build_ship_channel_bias(region_masks, grad_bgr):
    bias = np.zeros_like(grad_bgr, dtype=np.float32)
    for region_name in ['middle', 'head', 'tail']:
        region = region_masks[region_name]
        if float(region.sum()) <= 0.0:
            continue
        region_3 = region[..., None]
        region_grad = grad_bgr * region_3
        channel_step = np.sign(np.mean(region_grad[region > 0], axis=0, keepdims=True))
        if np.all(channel_step == 0):
            channel_step = np.sign(np.mean(grad_bgr.reshape(-1, 3), axis=0, keepdims=True))
        bias += region_3 * channel_step.reshape(1, 1, 3)
    return np.clip(bias, -1.0, 1.0)


def make_mask_preview(mask):
    mask_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    preview = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 3), dtype=np.uint8)
    preview[mask_u8 > 0] = (0, 0, 255)
    return preview


def make_perturb_preview(original_img, adv_img):
    diff = np.abs(adv_img.astype(np.int16) - original_img.astype(np.int16)).astype(np.uint8)
    diff_gray = np.max(diff, axis=2)
    heat = cv2.applyColorMap(np.clip(diff_gray * 25, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heat


def save_attack_visuals(save_mask_dir, save_overlay_dir, save_pert_dir, img_name, img_bgr, adv_img, attack_mask):
    base_name, _ = os.path.splitext(img_name)
    mask_vis = make_mask_preview(attack_mask[..., 0])
    cv2.imwrite(os.path.join(save_mask_dir, '{}_mask.png'.format(base_name)), mask_vis)

    if adv_img is None:
        return

    overlay = img_bgr.copy().astype(np.float32)
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255.0
    alpha = 0.45
    overlay[attack_mask[..., 0] > 0] = (1.0 - alpha) * overlay[attack_mask[..., 0] > 0] + alpha * red[attack_mask[..., 0] > 0]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_overlay_dir, '{}_overlay.png'.format(base_name)), overlay)

    perturb_vis = make_perturb_preview(img_bgr, adv_img)
    cv2.imwrite(os.path.join(save_pert_dir, '{}_perturb.png'.format(base_name)), perturb_vis)


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

    init_n = len(init)
    adv_n = len(adv)
    disappear_rate = float(max(init_n - adv_n, 0) / max(init_n, 1))

    if init_n == 0:
        return {
            'disappear_rate': 0.0,
            'iou_drop': 0.0,
            'angle_dev': 0.0,
            'center_shift_px': 0.0,
            'center_shift_norm': 0.0,
            'init_count': 0,
            'adv_count': adv_n,
        }

    ious = calc_iou_matrix(init, adv)
    if ious.size == 0:
        mean_iou = 0.0
        angle_dev = 90.0
        center_shift_px = 0.0
        center_shift_norm = 0.0
    else:
        max_idx = np.argmax(ious, axis=1)
        best_iou = ious[np.arange(ious.shape[0]), max_idx]
        mean_iou = float(np.mean(best_iou))

        devs = []
        shifts_px = []
        shifts_norm = []
        for i, j in enumerate(max_idx):
            a1 = pseudo_angle_from_box(init[i])
            a2 = pseudo_angle_from_box(adv[j])
            devs.append(abs(a1 - a2))
            c1x, c1y = box_center(init[i])
            c2x, c2y = box_center(adv[j])
            dx = c2x - c1x
            dy = c2y - c1y
            dist = float(np.sqrt(dx * dx + dy * dy))
            shifts_px.append(dist)
            x1, y1, x2, y2 = init[i]
            diag = float(np.sqrt(max(1e-6, (x2 - x1) ** 2 + (y2 - y1) ** 2)))
            shifts_norm.append(dist / diag)
        angle_dev = float(np.mean(devs)) if devs else 0.0
        center_shift_px = float(np.mean(shifts_px)) if shifts_px else 0.0
        center_shift_norm = float(np.mean(shifts_norm)) if shifts_norm else 0.0

    return {
        'disappear_rate': disappear_rate,
        'iou_drop': float(1.0 - mean_iou),
        'angle_dev': angle_dev,
        'center_shift_px': center_shift_px,
        'center_shift_norm': center_shift_norm,
        'init_count': init_n,
        'adv_count': adv_n,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='HRSC ship attack using one deployable TPA-style method')
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--score_thr', type=float, default=0.3)
    parser.add_argument('--alpha_norm', type=float, default=2.0 / 255.0,
                        help='Per-iteration step size in normalized image space')
    parser.add_argument('--epsilon_norm', type=float, default=10.0 / 255.0,
                        help='Linf budget in normalized image space, default 10/255')
    parser.add_argument('--middle_ratio', type=float, default=0.18,
                        help='Thickness ratio for the central long-strip patch')
    parser.add_argument('--end_ratio', type=float, default=0.28,
                        help='Fraction of the ship box used to search head/tail key regions')
    parser.add_argument('--grid_rows', type=int, default=2,
                        help='TPA-style grid rows for head/tail region selection')
    parser.add_argument('--grid_cols', type=int, default=2,
                        help='TPA-style grid cols for head/tail region selection')

    parser.add_argument('--config', type=str, default='./mmdetection/configs/fcos/fcos_hrsc.py')
    parser.add_argument('--checkpoint', type=str, default='./mmdetection/weight/epoch_12.pth')
    parser.add_argument('--data_root', type=str, default='/cloud/cloud-ssd1/collected_files/ljj/dataset/HRSC-coco/val')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--save_name', type=str, default='hrsc_fcos_ship_attack')
    parser.add_argument('--save_root', type=str, default='./results',
                        help='Root directory to save adversarial images')
    parser.add_argument('--no_save_images', action='store_true',
                        help='Disable saving adversarial images')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--deterministic', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    model = mmdetection_init_detector(config=args.config, checkpoint=args.checkpoint, device=args.device)

    imgs = sorted(os.listdir(args.data_root))
    s = max(0, args.start_idx)
    e = len(imgs) if args.end_idx < 0 else min(len(imgs), args.end_idx)
    imgs = imgs[s:e]

    save_images = not args.no_save_images
    save_img_dir = os.path.abspath(os.path.join(args.save_root, args.save_name, 'images'))
    save_mask_dir = os.path.abspath(os.path.join(args.save_root, args.save_name, 'masks'))
    save_overlay_dir = os.path.abspath(os.path.join(args.save_root, args.save_name, 'overlays'))
    save_pert_dir = os.path.abspath(os.path.join(args.save_root, args.save_name, 'perturbations'))
    save_txt_dir = os.path.join('./results_txt', args.save_name)
    os.makedirs(save_txt_dir, exist_ok=True)
    if save_images:
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_mask_dir, exist_ok=True)
        os.makedirs(save_overlay_dir, exist_ok=True)
        os.makedirs(save_pert_dir, exist_ok=True)

    per_img_metrics = []
    text_lines = []
    pbar = tqdm(enumerate(imgs), total=len(imgs))

    for _, img_name in pbar:
        model.zero_grad()
        img_path = os.path.join(args.data_root, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_tensor = to_tensor(img_bgr, args.device)
        img_tensor.requires_grad = True

        init_boxes, init_labels, init_det = fcos_attack_init(img_path, model, img_tensor, img_tensor)
        if init_boxes.size == 0:
            continue

        region_masks = build_ship_attack_regions(
            init_boxes,
            img_bgr,
            middle_ratio=args.middle_ratio,
            end_ratio=args.end_ratio,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )
        attack_mask = region_masks['total'][..., None]
        patch_ratio = float(np.mean(region_masks['total'] > 0))

        h, w = img_bgr.shape[:2]

        original_img = np.array(img_bgr, dtype=np.float32)
        perturb = np.zeros_like(original_img, dtype=np.float32)
        epsilon_px = float(args.epsilon_norm * 255.0)
        alpha_px = float(args.alpha_norm * 255.0)

        adv_img = None
        adv_boxes = init_boxes

        for it in range(args.iters):
            noise, adv_boxes, adv_labels, cls_loss, iou_loss = fcos_attack_mt(
                img_path, model, img_tensor, img_tensor, init_det, args.threshold
            )
            signed_noise = np.sign(noise)
            if np.sum(np.isnan(signed_noise)) == original_img.size:
                break

            grad_bgr = signed_noise.astype(np.float32)[..., ::-1].copy()
            region_bias = build_ship_channel_bias(region_masks, grad_bgr)
            perturb = perturb - alpha_px * region_bias * attack_mask
            perturb = np.clip(perturb, -epsilon_px, epsilon_px)
            adv_img = np.clip(original_img + perturb, 0, 255).astype(np.uint8)

            img_tensor = to_tensor(adv_img, args.device)
            img_tensor.requires_grad = True

        if adv_img is None:
            adv_img = ensure_cv2_image(img_tensor)

        m = summarize_metrics(init_boxes, adv_boxes, score_thr=args.score_thr)
        m['image'] = img_name
        per_img_metrics.append(m)

        pbar.set_description(
            '{} [ship] disp={:.3f} iou_drop={:.3f} shift={:.2f}px mask={:.3f}'.format(
                img_name, m['disappear_rate'], m['iou_drop'], m['center_shift_px'], patch_ratio
            )
        )

        text_lines.append(
            '[ship] {} | init={} adv={} | disappear_rate={:.4f} | iou_drop={:.4f} | center_shift_px={:.2f} | center_shift_norm={:.4f} | angle_dev={:.2f}'.format(
                img_name,
                m['init_count'],
                m['adv_count'],
                m['disappear_rate'],
                m['iou_drop'],
                m['center_shift_px'],
                m['center_shift_norm'],
                m['angle_dev'],
            )
        )

        if save_images:
            cv2.imwrite(os.path.join(save_img_dir, img_name), ensure_cv2_image(adv_img))
            save_attack_visuals(save_mask_dir, save_overlay_dir, save_pert_dir, img_name, img_bgr, adv_img, attack_mask)

    if not per_img_metrics:
        print('No valid images were processed.')
        return

    mean_disappear_rate = float(np.mean([x['disappear_rate'] for x in per_img_metrics]))
    mean_iou_drop = float(np.mean([x['iou_drop'] for x in per_img_metrics]))
    mean_angle_dev = float(np.mean([x['angle_dev'] for x in per_img_metrics]))
    mean_center_shift_px = float(np.mean([x['center_shift_px'] for x in per_img_metrics]))
    mean_center_shift_norm = float(np.mean([x['center_shift_norm'] for x in per_img_metrics]))

    summary = {
        'attack_type': 'ship_single_method',
        'epsilon_norm': args.epsilon_norm,
        'epsilon_px': epsilon_px,
        'alpha_norm': args.alpha_norm,
        'alpha_px': alpha_px,
        'middle_ratio': args.middle_ratio,
        'end_ratio': args.end_ratio,
        'grid_rows': args.grid_rows,
        'grid_cols': args.grid_cols,
        'num_images': len(per_img_metrics),
        'DisappearRate': mean_disappear_rate,
        'IoU_drop': mean_iou_drop,
        'Angle_dev': mean_angle_dev,
        'CenterShiftPx': mean_center_shift_px,
        'CenterShiftNorm': mean_center_shift_norm,
        'score_thr': args.score_thr,
        'iters': args.iters,
        'per_image': per_img_metrics,
    }

    out_json = os.path.join(save_txt_dir, 'ship_attack_summary.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    out_txt = os.path.join(save_txt_dir, 'ship_attack_summary.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('AttackType: ship_single_method\n')
        f.write('EpsilonNorm: {:.8f}\n'.format(args.epsilon_norm))
        f.write('EpsilonPx: {:.4f}\n'.format(epsilon_px))
        f.write('AlphaNorm: {:.8f}\n'.format(args.alpha_norm))
        f.write('AlphaPx: {:.4f}\n'.format(alpha_px))
        f.write('MiddleRatio: {}\n'.format(args.middle_ratio))
        f.write('EndRatio: {}\n'.format(args.end_ratio))
        f.write('GridRows: {}\n'.format(args.grid_rows))
        f.write('GridCols: {}\n'.format(args.grid_cols))
        f.write('NumImages: {}\n'.format(len(per_img_metrics)))
        f.write('ScoreThr: {}\n'.format(args.score_thr))
        f.write('Iters: {}\n'.format(args.iters))
        f.write('DisappearRate: {:.6f}\n'.format(mean_disappear_rate))
        f.write('IoU_drop: {:.6f}\n'.format(mean_iou_drop))
        f.write('CenterShiftPx: {:.4f}\n'.format(mean_center_shift_px))
        f.write('CenterShiftNorm: {:.6f}\n'.format(mean_center_shift_norm))
        f.write('Angle_dev: {:.4f}\n'.format(mean_angle_dev))
        f.write('\nPer-image details:\n')
        for line in text_lines:
            f.write(line + '\n')

    print('Saved summary to {}'.format(out_json))
    print('Saved text report to {}'.format(out_txt))
    if save_images:
        print('Adversarial images saved to {}'.format(save_img_dir))
        print('Attack masks saved to {}'.format(save_mask_dir))
        print('Overlay images saved to {}'.format(save_overlay_dir))
        print('Perturbation maps saved to {}'.format(save_pert_dir))
    else:
        print('Adversarial image saving is disabled (--no_save_images).')
    print('ShipAttack | N={} | DisappearRate={:.4f} | IoU_drop={:.4f} | CenterShiftPx={:.2f} | Angle_dev={:.4f} | epsilon=10/255'.format(
        len(per_img_metrics), mean_disappear_rate, mean_iou_drop, mean_center_shift_px, mean_angle_dev
    ))


if __name__ == '__main__':
    main()
