import cv2
import os
import numpy as np
from tqdm import tqdm

def process_dataset(input_dir, output_dir, border_ratio=0.05):
    """
    读取数据集图片，将边缘一定比例的区域涂黑，保持原图分辨率不变（不影响检测框坐标）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 【修复重点】：HRSC数据集的图像格式通常是 .bmp！所以必须把 .bmp 加进后缀过滤里
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    
    if not os.path.exists(input_dir):
        print(f"找不到输入路径: {input_dir}")
        return

    img_names = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    
    if len(img_names) == 0:
        print(f"在 {input_dir} 目录下没有找到格式为 {valid_exts} 的图片！请检查路径或图片格式。")
        return

    print(f"找到 {len(img_names)} 张图片，准备开始处理...")

    for img_name in tqdm(img_names, desc="处理图像"):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"警告：无法读取图像 {img_path}")
            continue
            
        h, w = img.shape[:2]
        
        # 计算需要涂黑的边框像素厚度，比如横纵方向各自的5% 
        border_h = int(h * border_ratio)
        border_w = int(w * border_ratio)

        # 直接将四周赋值为纯黑色 [0, 0, 0]
        # 顶部和底部
        img[:border_h, :] = 0
        img[-border_h:, :] = 0
        # 左侧和右侧
        img[:, :border_w] = 0
        img[:, -border_w:] = 0

        # 保存到新目录
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, img)

if __name__ == "__main__":
    # 在这里填入你的数据集路径
    INPUT_DIR = "/cloud/cloud-ssd1/collected_files/ljj/dataset/HRSC-coco/val/"
    # 新测试集保存路径
    OUTPUT_DIR = "/cloud/cloud-ssd1/collected_files/ljj/dataset/HRSC-coco/val_black_border/"
    
    # border_ratio 控制黑边的大小，0.05代表宽和高各自上下/左右5%变黑
    process_dataset(INPUT_DIR, OUTPUT_DIR, border_ratio=0.05)
    print("黑边数据集生成完毕！")
