import argparse
import copy
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


def put_square(mask, x, y, size, h, w):
    half = max(1, size // 2)
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half)
    y2 = min(h, y + half)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0


def strategy_axis_strip(mask, boxes, h, w, small_ratio=0.12, strip_ratio=0.18):
    for b in boxes:
        cb = clamp_box(b[:4], h, w)
        if cb is None:
            continue
        x1, y1, x2, y2 = cb
        bw, bh = x2 - x1, y2 - y1
        if bw >= bh:
            t = max(2, int(bh * strip_ratio))
            yc = (y1 + y2) // 2
            ys, ye = max(y1, yc - t // 2), min(y2, yc + t // 2)
            mask[ys:ye, x1:x2] = 1.0
            s = max(2, int(min(bw, bh) * small_ratio))
            put_square(mask, x1 + bw // 4, yc, s, h, w)
            put_square(mask, x2 - bw // 4, yc, s, h, w)
        else:
            t = max(2, int(bw * strip_ratio))
            xc = (x1 + x2) // 2
            xs, xe = max(x1, xc - t // 2), min(x2, xc + t // 2)
            mask[y1:y2, xs:xe] = 1.0
            s = max(2, int(min(bw, bh) * small_ratio))
            put_square(mask, xc, y1 + bh // 4, s, h, w)
            put_square(mask, xc, y2 - bh // 4, s, h, w)


def strategy_fod_small(mask, boxes, img_bgr, h, w, k=3, patch_ratio=0.1):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    fod = cv2.magnitude(gx, gy)

    for b in boxes:
        cb = clamp_box(b[:4], h, w)
        if cb is None:
            continue
        x1, y1, x2, y2 = cb
        roi = fod[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        flat = roi.reshape(-1)
        topk = min(k, flat.size)
        idx = np.argpartition(flat, -topk)[-topk:]
        s = max(2, int(min(x2 - x1, y2 - y1) * patch_ratio))
        rw = x2 - x1
        for i in idx:
            ry = i // rw
            rx = i % rw
            put_square(mask, x1 + int(rx), y1 + int(ry), s, h, w)


def strategy_endpoints(mask, boxes, h, w, patch_ratio=0.16):
    for b in boxes:
        cb = clamp_box(b[:4], h, w)
        if cb is None:
            continue
        x1, y1, x2, y2 = cb
        bw, bh = x2 - x1, y2 - y1
        s = max(2, int(min(bw, bh) * patch_ratio))
        if bw >= bh:
            yc = (y1 + y2) // 2
            put_square(mask, x1, yc, s, h, w)
            put_square(mask, x2, yc, s, h, w)
        else:
            xc = (x1 + x2) // 2
            put_square(mask, xc, y1, s, h, w)
            put_square(mask, xc, y2, s, h, w)


def build_attack_map(strategy, boxes, img_bgr, mask_dilate=0):
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    if strategy == 'A':
        strategy_axis_strip(mask, boxes, h, w)
    elif strategy == 'B':
        strategy_fod_small(mask, boxes, img_bgr, h, w)
    elif strategy == 'C':
        strategy_endpoints(mask, boxes, h, w)
    else:
        raise ValueError('Unknown strategy: {}'.format(strategy))

    if mask_dilate > 0:
        k = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), k, iterations=mask_dilate).astype(np.float32)

    return np.stack([mask, mask, mask], axis=-1)


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
    parser = argparse.ArgumentParser(description='HRSC Stage-1 patch deployment ablation based on TPA attack pipeline')
    parser.add_argument('--strategy', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--score_thr', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=4.0,
                        help='Per-iteration perturbation step in pixel space')
    parser.add_argument('--epsilon', type=float, default=32.0,
                        help='Linf perturbation budget in pixel space')
    parser.add_argument('--mask_dilate', type=int, default=1,
                        help='Dilate patch mask iterations to enlarge attacked area')

    parser.add_argument('--config', type=str, default='./mmdetection/configs/fcos/fcos_hrsc.py')
    parser.add_argument('--checkpoint', type=str, default='./mmdetection/weight/epoch_12.pth')
    parser.add_argument('--data_root', type=str, default='/cloud/cloud-ssd1/collected_files/ljj/dataset/HRSC-coco/val')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--save_name', type=str, default='hrsc_fcos_stage1')
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
    save_img_dir = os.path.join(args.save_root, args.save_name, 'iter_{}'.format(args.strategy))
    save_img_dir = os.path.abspath(save_img_dir)
    save_txt_dir = os.path.join('./results_txt', args.save_name)
    os.makedirs(save_txt_dir, exist_ok=True)
    if save_images:
        os.makedirs(save_img_dir, exist_ok=True)

    per_img_metrics = []
    text_lines = []
    pbar = tqdm(enumerate(imgs), total=len(imgs))

    for ind, img_name in pbar:
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

        attack_map = build_attack_map(args.strategy, init_boxes, img_bgr, mask_dilate=args.mask_dilate)
        patch_ratio = float(np.mean(attack_map[..., 0] > 0))

        original_img = np.array(img_bgr, dtype=np.float32)
        perturb = np.zeros_like(original_img, dtype=np.float32)

        adv_img = None
        adv_boxes = init_boxes

        for _ in range(args.iters):
            noise, adv_boxes, adv_labels, cls_loss, iou_loss = fcos_attack_mt(
                img_path, model, img_tensor, img_tensor, init_det, args.threshold
            )
            signed_noise = np.sign(noise)
            if np.sum(np.isnan(signed_noise)) == original_img.size:
                break

            delta = signed_noise.astype(np.float32) * attack_map
            delta = delta[..., ::-1].copy()
            perturb = perturb - args.alpha * delta
            perturb = np.clip(perturb, -args.epsilon, args.epsilon)
            adv_img = np.clip(original_img + perturb, 0, 255).astype(np.uint8)

            img_tensor = to_tensor(adv_img, args.device)
            img_tensor.requires_grad = True

        if adv_img is None:
            adv_img = ensure_cv2_image(img_tensor)

        m = summarize_metrics(init_boxes, adv_boxes, score_thr=args.score_thr)
        m['image'] = img_name
        per_img_metrics.append(m)

        pbar.set_description(
            '{} [{}] disp={:.3f} iou_drop={:.3f} shift={:.2f}px mask={:.3f}'.format(
                img_name, args.strategy, m['disappear_rate'], m['iou_drop'], m['center_shift_px'], patch_ratio
            )
        )

        text_lines.append(
            '[{}] {} | init={} adv={} | disappear_rate={:.4f} | iou_drop={:.4f} | center_shift_px={:.2f} | center_shift_norm={:.4f} | angle_dev={:.2f}'.format(
                args.strategy,
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

    if not per_img_metrics:
        print('No valid images were processed.')
        return

    mean_disappear_rate = float(np.mean([x['disappear_rate'] for x in per_img_metrics]))
    mean_iou_drop = float(np.mean([x['iou_drop'] for x in per_img_metrics]))
    mean_angle_dev = float(np.mean([x['angle_dev'] for x in per_img_metrics]))
    mean_center_shift_px = float(np.mean([x['center_shift_px'] for x in per_img_metrics]))
    mean_center_shift_norm = float(np.mean([x['center_shift_norm'] for x in per_img_metrics]))

    summary = {
        'strategy': args.strategy,
        'num_images': len(per_img_metrics),
        'DisappearRate': mean_disappear_rate,
        'IoU_drop': mean_iou_drop,
        'Angle_dev': mean_angle_dev,
        'CenterShiftPx': mean_center_shift_px,
        'CenterShiftNorm': mean_center_shift_norm,
        'score_thr': args.score_thr,
        'alpha': args.alpha,
        'epsilon': args.epsilon,
        'mask_dilate': args.mask_dilate,
        'iters': args.iters,
        'per_image': per_img_metrics,
    }

    out_json = os.path.join(save_txt_dir, 'stage1_{}_summary.json'.format(args.strategy))
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    out_txt = os.path.join(save_txt_dir, 'stage1_{}_summary.txt'.format(args.strategy))
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('Strategy {}\n'.format(args.strategy))
        f.write('NumImages: {}\n'.format(len(per_img_metrics)))
        f.write('ScoreThr: {}\n'.format(args.score_thr))
        f.write('Alpha: {}\n'.format(args.alpha))
        f.write('Epsilon: {}\n'.format(args.epsilon))
        f.write('MaskDilate: {}\n'.format(args.mask_dilate))
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
    else:
        print('Adversarial image saving is disabled (--no_save_images).')
    print('Strategy={} | N={} | DisappearRate={:.4f} | IoU_drop={:.4f} | CenterShiftPx={:.2f} | Angle_dev={:.4f}'.format(
        args.strategy, len(per_img_metrics), mean_disappear_rate, mean_iou_drop, mean_center_shift_px, mean_angle_dev
    ))


if __name__ == '__main__':
    main()
