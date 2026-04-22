import sys
import os
import argparse
import torch
import cv2
import copy
import numpy as np
from tqdm import tqdm
from unittest import result

# 添加依赖路径（确保mmdetection和yolov4 eval代码能导入）
sys.path.append('./yolov4/eval_code') 
sys.path.append('../mmdetection') 

# 导入mmdetection推理函数
from mmdetection.mmdet.apis.inference import init_detector as mmdetection_init_detector
from mmdetection.mmdet.apis.inference import inference_single_attack_single_init as fcos_attack_init
from mmdetection.mmdet.apis.inference import inference_single_attack_single_mt as fcos_attack_mt

# ===================== 新增：特征提取Hook机制与热力图 =====================
class FeatureExtractor:
    """提取模型深层特征及其梯度的Hook类，用于Grad-CAM"""
    def __init__(self, model):
        self.features = None
        self.gradients = None
        # 挂载在backbone上，提取骨干网络最深层特征 (例如ResNet的layer4)
        self.hook = model.backbone.register_forward_hook(self.hook_fn)

    def save_gradient(self, grad):
        self.gradients = grad.detach()

    def hook_fn(self, module, input, output):
        # fcos backbone (ResNet) typically outputs a tuple of features (C2, C3, C4, C5)
        # 提取深层特征 (C5 即最后一个元素)
        if isinstance(output, tuple) or isinstance(output, list):
            feat = output[-1]
        else:
            feat = output
            
        self.features = feat.detach()
        # 注册tensor的反向hook来提取关于该特征图的梯度
        if feat.requires_grad:
            feat.register_hook(self.save_gradient)

    def remove(self):
        self.hook.remove()

def generate_deep_feature_heatmap(feature_tensor, gradient_tensor, orig_img):
    """
    基于Grad-CAM计算深层特征tensor的热力图并叠加在原图上
    Args:
        feature_tensor: 深层特征张量 (1, C, H, W)
        gradient_tensor: 对应的梯度张量 (1, C, H, W)，如果为None则回退到平均池化
        orig_img: 原图尺寸 (H, W, 3) cv2 uint8
    Returns:
        overlay_heatmap: 热力图与原图叠加
        pure_heatmap: 纯热力图
    """
    if gradient_tensor is not None:
        # ===== Grad-CAM 核心逻辑 ======
        # 全局平均池化得到每个通道的权重
        weights = torch.mean(gradient_tensor, dim=(2, 3), keepdim=True)
        # 特征图与权重加权求和
        cam = torch.sum(weights * feature_tensor, dim=1).squeeze(0).cpu().numpy()
    else:
        # 退回通道平均激活
        cam = torch.mean(feature_tensor, dim=1).squeeze(0).cpu().numpy()
        
    # ReLU操作，剔除对检测呈负向响应的值
    activation = np.maximum(cam, 0)
    
    # 归一化到0-255
    if activation.max() > activation.min():
        activation = (activation - activation.min()) / (activation.max() - activation.min())
    activation_u8 = np.uint8(255 * activation)
    
    # resize回原图尺寸
    orig_h, orig_w = orig_img.shape[:2]
    heatmap = cv2.resize(activation_u8, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    # 转为彩色热力图 (JET伪彩色)
    pure_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 叠加到原图上
    overlay_heatmap = cv2.addWeighted(orig_img, 0.5, pure_heatmap, 0.5, 0)
    
    return overlay_heatmap, pure_heatmap

# ===================== 新增：坐标映射核心函数 =====================
def rescale_boxes_to_original(boxes, orig_w, orig_h, resize_w=800, resize_h=800):
    """
    将resize后图像上的检测框，精准映射回原图尺寸
    Args:
        boxes: 检测框数组，shape=(N,4+)，格式[x1,y1,x2,y2, 置信度...]
        orig_w: 原图宽度
        orig_h: 原图高度
        resize_w: 模型输入resize后的宽度
        resize_h: 模型输入resize后的高度
    Returns:
        boxes_orig: 映射回原图坐标的检测框
    """
    if boxes.size == 0:
        return boxes
    scale_x = orig_w / resize_w
    scale_y = orig_h / resize_h
    boxes_orig = boxes.copy()
    # 只映射坐标，不修改置信度等其他字段
    boxes_orig[:, 0] = boxes[:, 0] * scale_x  # x1
    boxes_orig[:, 1] = boxes[:, 1] * scale_y  # y1
    boxes_orig[:, 2] = boxes[:, 2] * scale_x  # x2
    boxes_orig[:, 3] = boxes[:, 3] * scale_y  # y2
    return boxes_orig

def toTensor(img, device='cuda:0'):
    """将numpy图像(H,W,3)转为tensor(1,3,H,W)，并移到指定设备"""
    assert isinstance(img, np.ndarray), f"输入必须是np.ndarray，当前是{type(img)}"
    img = torch.from_numpy(img.transpose((2, 0, 1)))  # HWC -> CHW
    return img.float().to(device).unsqueeze(0)  # 加batch维度

def ensure_cv2_image(img):
    """将tensor转为cv2可用的uint8图像(H,W,3)"""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.dim() == 4:
            img = img.squeeze(0)  # 去掉batch维度
        img = img.permute(1, 2, 0).contiguous().numpy()  # CHW -> HWC
    # 确保数值范围在0-255，类型为uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def make_mask_preview(mask):
    """生成掩码预览图（红底黑边）"""
    mask_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    preview = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 3), dtype=np.uint8)
    preview[mask_u8 > 0] = (0, 0, 255)  # 掩码区域标红
    return preview

def make_perturb_preview(img_init, img_adv):
    """生成扰动热力图（优化对比度）"""
    # 计算像素差值（避免uint8溢出，先转int16）
    diff = np.abs(img_adv.astype(np.int16) - img_init.astype(np.int16)).astype(np.uint8)
    diff_gray = np.max(diff, axis=2)  # 3通道转单通道（取最大差值）
    # 归一化到0-255（提升对比度，避免*25溢出）
    diff_gray_norm = diff_gray * (255 / (diff_gray.max() + 1e-8))  # +1e-8避免除0
    heat = cv2.applyColorMap(np.clip(diff_gray_norm, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heat

def make_overlay_preview(img_init, img_adv, alpha=0.6):
    """生成原图和对抗图的叠加预览"""
    img_init_u8 = ensure_cv2_image(img_init)
    img_adv_u8 = ensure_cv2_image(img_adv)
    overlay = cv2.addWeighted(img_init_u8, alpha, img_adv_u8, 1.0 - alpha, 0)
    return overlay

def make_cam_patch_mask(boxes_init, img_tensor_np, patch_type='rotated', max_area_ratio=0.5, deep_features=None, deep_gradients=None):
    """
    基于深层特征（Grad-CAM）生成补丁掩码
    Args:
        boxes_init: 初始检测框列表 [[x1,y1,x2,y2], ...]
        img_tensor_np: tensor转numpy后的数组，shape=(1,3,H,W)
        patch_type: 补丁类型 'rotated'（旋转矩形）/'general'（普通矩形）
        max_area_ratio: 补丁最大占目标框面积的比例
        deep_features: 深层特征张量，由Hook机制提取，shape=(1,C,H',W')
        deep_gradients: 深层特征对应的梯度张量，shape=(1,C,H',W')
    Returns:
        attack_map_3d: 3通道补丁掩码，shape=(H,W,3)
    """
    # 显式提取图像维度（避免硬编码shape索引）
    _, _, H, W = img_tensor_np.shape
    attack_map = np.zeros((H, W), dtype=np.float32)  # 初始化掩码（单通道）

    # 生成显著性地图（利用Hook出的深层语义激活特征图和梯度计算Grad-CAM）
    if deep_features is not None and deep_gradients is not None:
        # ====== Grad-CAM 核心逻辑 ======
        weights = torch.mean(deep_gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * deep_features, dim=1).squeeze(0).cpu().numpy()
        cam = np.maximum(cam, 0) # ReLU
        # 将特征图(如25x25)线性插值上采样到原输入分辨率(如800x800)
        saliency_map = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
    elif deep_features is not None:
        # 退回通道均值
        activation = torch.mean(deep_features, dim=1).squeeze(0).cpu().numpy()
        activation = np.maximum(activation, 0)
        saliency_map = cv2.resize(activation, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        saliency_map = np.ones((H, W), dtype=np.float32)

    for box in boxes_init:
        # 解析框坐标并做边界裁剪（避免越界）
        x1, y1, x2, y2 = map(int, box[:4])
        x1 = max(0, min(x1, W - 1))
        x2 = max(x1 + 1, min(x2, W))
        y1 = max(0, min(y1, H - 1))
        y2 = max(y1 + 1, min(y2, H))
        
        w = x2 - x1
        h = y2 - y1
        box_area = w * h
        max_patch_area = box_area * max_area_ratio  # 补丁最大面积

        # 提取框内的显著性地图
        roi_cam = saliency_map[y1:y2, x1:x2]
        if roi_cam.size == 0 or roi_cam.max() <= 1e-5:
            print(f"警告：框[{x1,y1,x2,y2}]内无有效显著性，跳过")
            continue
            
        # 归一化+二值化（找高敏感区域）
        roi_cam_norm = (roi_cam - roi_cam.min()) / (roi_cam.max() - roi_cam.min())
        threshold = np.mean(roi_cam_norm) + 0.5 * np.std(roi_cam_norm)
        binary_mask = (roi_cam_norm > threshold).astype(np.uint8) * 255
        
        # 形态学开运算（去除小噪点）
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓（找高敏感区域的连通域）
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # 无轮廓时，在框中心生成默认补丁
            cx, cy = x1 + w//2, y1 + h//2
            pw = int(w * np.sqrt(max_area_ratio))
            ph = int(h * np.sqrt(max_area_ratio))
            # 裁剪补丁边界（避免越界）
            y_start = max(0, cy - ph//2)
            y_end = min(H, cy + ph//2)
            x_start = max(0, cx - pw//2)
            x_end = min(W, cx + pw//2)
            attack_map[y_start:y_end, x_start:x_end] = 1.0
            continue
            
        # 选面积最大的连通域（最显著区域）
        cnt = max(contours, key=cv2.contourArea)
        cnt = cnt + np.array([[x1, y1]])  # 轮廓坐标偏移回整图

        # 生成补丁掩码
        patch_mask = np.zeros_like(attack_map, dtype=np.uint8)
        if patch_type == 'rotated':
            # 旋转矩形：最小外接矩形（含旋转角度）
            rect = cv2.minAreaRect(cnt)
            (center, (rect_w, rect_h), angle) = rect
            rect_area = rect_w * rect_h

            # 缩放补丁到最大面积限制
            if rect_area > max_patch_area and rect_area > 0:
                scale = np.sqrt(max_patch_area / rect_area)
                rect_w *= scale
                rect_h *= scale
                rect = (center, (rect_w, rect_h), angle)

            # 绘制旋转矩形补丁
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            cv2.drawContours(patch_mask, [box_points], 0, 1, -1)
        else:
            # 普通矩形：最大外接矩形
            bx, by, bw, bh = cv2.boundingRect(cnt)
            rect_area = bw * bh

            # 缩放补丁到最大面积限制（保持中心不变）
            if rect_area > max_patch_area and rect_area > 0:
                scale = np.sqrt(max_patch_area / rect_area)
                new_bw = int(bw * scale)
                new_bh = int(bh * scale)
                bx = bx + (bw - new_bw) // 2
                by = by + (bh - new_bh) // 2
                bw, bh = new_bw, new_bh

            # 绘制普通矩形补丁
            patch_mask[by:by+bh, bx:bx+bw] = 1

        # 合并补丁到总掩码
        attack_map = np.maximum(attack_map, patch_mask)
    
    # 单通道转3通道（匹配图像通道数）
    attack_map_3d = np.stack((attack_map, attack_map, attack_map), axis=-1)       
    return attack_map_3d

def parse_args():
    """解析命令行参数（小白重点看注释里的默认值和含义）"""
    parser = argparse.ArgumentParser(description='FCOS舰船检测器CAM引导对抗攻击')
    # 核心参数
    parser.add_argument('--patch_type', type=str, default='rotated', choices=['rotated', 'general'],
                        help='补丁类型：rotated（旋转矩形）/general（普通矩形）')
    parser.add_argument('--max_area', type=float, default=0.5,
                        help='补丁占目标框面积的最大比例（建议0.3-0.5）')
    parser.add_argument('--iters', type=int, default=10,
                        help='攻击迭代次数（建议10-20）')
    parser.add_argument('--adversarial_degree', type=int, default=16,
                        help='最大扰动幅度（对抗攻击常规8-16，255会完全失真）')
    parser.add_argument('--step_size', type=float, default=2.0,
                        help='每次迭代的扰动步长（建议小于adversarial_degree/5）')
    # 路径参数
    parser.add_argument('--save_name', type=str, default='hrsc_fcos_cam',
                        help='结果保存文件夹名')
    parser.add_argument('--config', type=str, default='./mmdetection/configs/fcos/fcos_hrsc.py',
                        help='FCOS配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='./mmdetection/weight/epoch_12.pth',
                        help='FCOS权重文件路径')
    parser.add_argument('--data_root', type=str, default='/cloud/cloud-ssd1/collected_files/ljj/dataset/HRSC-coco/val/',
                        help='HRSC数据集val目录路径')
    # 其他参数
    parser.add_argument('--threshold', type=float, default=0, help='IOU阈值')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备（cpu/cuda:0）')
    parser.add_argument('--save_iter_images', action='store_true', help='是否保存迭代过程图')
    parser.add_argument('--no_save_final_images', action='store_true', help='是否不保存最终对抗图')
    parser.add_argument('--deterministic', action='store_true', help='是否固定随机种子（复现结果）')
    parser.add_argument('--start_idx', type=int, default=0, help='起始处理图片索引')
    parser.add_argument('--end_idx', type=int, default=-1, help='结束处理图片索引（-1表示全部）')
    parser.add_argument('--image_batch', type=str, default='all', help='批次名（用于结果文件命名）')
    args = parser.parse_args()
    return args

def attack_imgs(root_path, args, imgs):
    """
    批量处理图像攻击
    Args:
        root_path: 图像根目录
        args: 命令行参数
        imgs: 待处理图像列表
    """
    # 初始化FCOS模型
    print(f"加载FCOS模型：{args.config} | 权重：{args.checkpoint}")
    fcos_model = mmdetection_init_detector(config=args.config, checkpoint=args.checkpoint, device=args.device)

    # 注册Hook提取深深层特征
    feature_extractor = FeatureExtractor(fcos_model)

    # 创建结果保存目录（确保目录存在）
    save_iter_dir = f'./results/{args.save_name}/iter'
    save_vis_dir = f'./results/{args.save_name}/vis'
    save_txt_dir = f'./results_txt/{args.save_name}'
    os.makedirs(save_iter_dir, exist_ok=True)
    os.makedirs(save_vis_dir, exist_ok=True)
    os.makedirs(save_txt_dir, exist_ok=True)
    save_perts = os.path.join(save_txt_dir, f'result_{args.image_batch}.txt')
    print(f"结果保存路径：\n- 迭代图：{save_iter_dir}\n- 可视化：{save_vis_dir}\n- 指标：{save_perts}")

    # 遍历图像处理
    for ind, img_name in enumerate(imgs):
        print(f"\n===== 处理第{ind+1}/{len(imgs)}张图：{img_name} =====")
        # 清空模型梯度（避免跨图像梯度累积）
        fcos_model.zero_grad()
        
        # 读取原图（全程锁死，不修改，专门用于det_init）
        img_path = os.path.join(root_path, img_name)
        img_init = cv2.imread(img_path)
        if img_init is None:
            print(f"警告：无法读取图像 {img_path}，跳过")
            continue
        orig_h, orig_w = img_init.shape[:2]  # 记录原图尺寸，全程不变

        # 图像预处理：转tensor+resize到800x800（仅用于模型推理，不修改原图）
        img_tensor = toTensor(img_init, args.device)
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=(800, 800), mode='bilinear', align_corners=False)
        img_tensor.requires_grad = True

        # 1. 锁死干净原图初始检测
        fcos_boxes_resize, labels, init_det = fcos_attack_init(img_path, fcos_model, img_tensor, img_tensor)
        
        # 将resize后的框映射回原图尺寸，专门用于det_init绘制
        orig_boxes_all = rescale_boxes_to_original(fcos_boxes_resize, orig_w, orig_h)
        # 过滤低分框（置信度>0.3），分别保存
        boxes_init_for_attack = []
        orig_boxes_for_draw = []
        for i in range(len(labels)):
            if fcos_boxes_resize[i][-1] > 0.3:
                boxes_init_for_attack.append(fcos_boxes_resize[i][:4].astype(int))
                orig_boxes_for_draw.append(orig_boxes_all[i][:4].astype(int))
        print(f"干净原图初始检测到{len(orig_boxes_for_draw)}个舰船框")

        if len(orig_boxes_for_draw) == 0:
            continue

        # ===================== 核心优化：采用深层语义特征与特征梯度 (Grad-CAM) 引导补丁 =====================
        # 2. 为了获得深层特征对于总损失的梯度(反向权重)，这里必须走一次 fcos_attack_mt
        # (因为 mt 里调用了 detector.forward_train 并计算 loss 触发了 backward_hook)
        _, _, _, _, _ = fcos_attack_mt(img_path, fcos_model, img_tensor, img_tensor, init_det, args.threshold)
        
        # 提取深层干净特征和其反向传播得到的损失梯度
        clean_deep_features = feature_extractor.features.clone() if feature_extractor.features is not None else None
        clean_deep_gradients = feature_extractor.gradients.clone() if feature_extractor.gradients is not None else None

        # 3. 生成基于 Grad-CAM 的补丁掩码（800x800尺寸，用于攻击）
        attack_map_800 = make_cam_patch_mask(
            boxes_init=boxes_init_for_attack,
            img_tensor_np=img_tensor.cpu().detach().numpy(),
            patch_type=args.patch_type,
            max_area_ratio=args.max_area,
            deep_features=clean_deep_features,
            deep_gradients=clean_deep_gradients
        )
        # 掩码resize回原图尺寸，用于后续可视化
        attack_map_orig = cv2.resize(attack_map_800, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 初始化对抗扰动约束（避免像素溢出）
        original_img_np = img_init.astype(np.float32)
        clip_min = np.clip(original_img_np - args.adversarial_degree, 0, 255)
        clip_max = np.clip(original_img_np + args.adversarial_degree, 0, 255)

        # 迭代攻击
        current_img = copy.deepcopy(img_init)  # 攻击用的图像，初始为原图，迭代中修改
        final_adv_boxes_resize = np.array([])  # 保存攻击后的最终检测框
        pbar = tqdm(range(args.iters), desc=f"攻击迭代 {img_name}")
        for attack_iter in pbar:
            # 清空模型梯度（避免迭代间梯度累积）
            fcos_model.zero_grad()
            
            # 图像转tensor并resize到800x800（仅用于模型推理）
            img_input = toTensor(current_img, args.device)
            img_input = torch.nn.functional.interpolate(img_input, size=(800, 800), mode='bilinear', align_corners=False)
            img_input.requires_grad = True

            # 前向+反向，获取对抗梯度和攻击后的检测结果
            attack_gradient, fcos_boxes, labels_pred, class_loss, iou_loss = fcos_attack_mt(
                img_path, fcos_model, img_input, img_input, init_det, args.threshold
            )

            # 更新此时的最新的检测结果
            final_adv_boxes_resize = fcos_boxes.copy()

            # 处理梯度NaN（避免攻击崩溃）
            noise_img = np.sign(attack_gradient)
            if np.sum(np.isnan(noise_img)) == original_img_np.size:
                print(f"警告：迭代{attack_iter}梯度全为NaN，终止攻击")
                break

            # 更新进度条
            attack_rate = attack_map_orig[attack_map_orig == 1].size / attack_map_orig.size
            pbar.set_description(f"{img_name} | 初始框：{len(orig_boxes_for_draw)} | 当前框：{len(fcos_boxes)} | 补丁占比：{attack_rate:.2f}")

            # 计算扰动并resize回原图尺寸
            current_img_tensor = current_img.astype(np.float32)
            # 1. 800x800尺寸上计算扰动
            perturb_800 = noise_img.astype(np.float32) * attack_map_800 * args.step_size
            perturb_800 = perturb_800[..., ::-1].copy()  # BGR转RGB
            # 2. 扰动resize回原图尺寸
            perturb_orig = cv2.resize(perturb_800, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            # 3. 只在掩码区域施加扰动
            perturb_orig = perturb_orig * attack_map_orig
            # 4. 施加扰动并约束像素范围
            current_img = np.clip(current_img_tensor - perturb_orig, clip_min, clip_max).astype(np.uint8)

        # 保存最终对抗图
        if not args.no_save_final_images:
            final_img_path = os.path.join(save_iter_dir, img_name)
            cv2.imwrite(final_img_path, ensure_cv2_image(current_img))
            print(f"保存最终对抗图：{final_img_path}")

        # ===================== 【核心修正2：严格分离攻击前后检测结果绘制】 =====================
        img_adv = ensure_cv2_image(current_img)
        img_base, img_ext = os.path.splitext(img_name)

        # 1. 【100%干净原图的检测结果】det_init：纯原图 + 初始检测绿框，和攻击完全无关
        img_init_u8 = img_init.astype(np.uint8).copy()  # 完全未修改的干净原图
        for box in orig_boxes_for_draw:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img_init_u8, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(save_vis_dir, f"{img_base}_det_init{img_ext}"), img_init_u8)
        print(f"已生成干净原图检测结果：{img_base}_det_init{img_ext}")

        # 2. 攻击后的检测结果 det_adv：对抗图 + 攻击后检测红框
        img_adv_u8 = img_adv.copy()
        # 攻击后的框映射回原图尺寸
        if final_adv_boxes_resize.size > 0:
            adv_boxes_orig = rescale_boxes_to_original(final_adv_boxes_resize, orig_w, orig_h)
            for box in adv_boxes_orig:
                if box[-1] > 0.3:  # 过滤低分框
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(img_adv_u8, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(save_vis_dir, f"{img_base}_det_adv{img_ext}"), img_adv_u8)

        # 3. 补丁掩码叠加图
        mask_binary = attack_map_orig[..., 0] > 0
        img_mask_overlay = img_init.astype(np.uint8).copy()
        mask_color = np.zeros_like(img_mask_overlay)
        mask_color[mask_binary] = [0, 0, 255]
        img_mask_overlay = cv2.addWeighted(img_mask_overlay, 1.0, mask_color, 0.5, 0)
        cv2.imwrite(os.path.join(save_vis_dir, f"{img_base}_mask{img_ext}"), img_mask_overlay)

        # 4. 补丁区域扰动热力图
        perturb_preview = make_perturb_preview(img_init, img_adv)
        perturb_masked = np.zeros_like(perturb_preview)
        perturb_masked[:] = cv2.applyColorMap(np.array([[0]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        perturb_masked[mask_binary] = perturb_preview[mask_binary]
        cv2.imwrite(os.path.join(save_vis_dir, f"{img_base}_perturb{img_ext}"), perturb_masked)

        # 5. 原图+对抗图叠加预览
        overlay_img = make_overlay_preview(img_init, img_adv)
        cv2.imwrite(os.path.join(save_vis_dir, f"{img_base}_overlay{img_ext}"), overlay_img)

        # 6. 深层特征(Grad-CAM)热力图记录（干净层 vs 攻击受损层）
        if clean_deep_features is not None and clean_deep_gradients is not None:
            clean_heatmap_overlay, _ = generate_deep_feature_heatmap(clean_deep_features, clean_deep_gradients, img_init.astype(np.uint8))
            cv2.imwrite(os.path.join(save_vis_dir, f"{img_base}_deep_feature_init{img_ext}"), clean_heatmap_overlay)
        if feature_extractor.features is not None and feature_extractor.gradients is not None:
            adv_heatmap_overlay, _ = generate_deep_feature_heatmap(feature_extractor.features, feature_extractor.gradients, img_init.astype(np.uint8))
            cv2.imwrite(os.path.join(save_vis_dir, f"{img_base}_deep_feature_adv{img_ext}"), adv_heatmap_overlay)

        # 计算扰动指标并保存
        pp = (img_adv.astype(np.float32) - img_init) / 255.0
        pp_L2 = np.sqrt(np.sum(pp ** 2))
        pp_Linf = np.max(np.abs(pp))
        pp_L0 = np.count_nonzero(pp) / pp.size
        with open(save_perts, 'a', encoding='utf-8') as f:
            f.write(f"{ind} {img_name} L2 {pp_L2:.4f} Linf {pp_Linf:.4f} L0 {pp_L0:.4f} iters {attack_iter+1}\n")
        print(f"扰动指标：L2={pp_L2:.4f} | Linf={pp_Linf:.4f} | L0={pp_L0:.4f}")

if __name__ == '__main__':
    # 解析参数
    args = parse_args()
    
    # 固定随机种子（复现结果）
    if args.deterministic:
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("已固定随机种子，结果可复现")
    else:
        torch.backends.cudnn.benchmark = True

    # 加载图像列表
    root_path = args.data_root
    if not os.path.exists(root_path):
        raise ValueError(f"数据目录不存在：{root_path}")
    imgs = sorted(os.listdir(root_path))
    # 筛选图像范围
    start_idx = max(args.start_idx, 0)
    end_idx = len(imgs) if args.end_idx < 0 else min(args.end_idx, len(imgs))
    imgs = imgs[start_idx:end_idx]
    print(f"待处理图像数量：{len(imgs)}（{start_idx}~{end_idx}）")

    # 开始攻击
    attack_imgs(root_path, args, imgs)
    print("\n===== 所有图像处理完成 =====")