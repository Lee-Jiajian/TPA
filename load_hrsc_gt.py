import os
import json
import cv2
import numpy as np

def load_and_visualize_hrsc_coco_gt(img_dir, ann_file, output_dir, num_samples=10):
    """
    加载并可视化 HRSC-coco 格式的 Ground Truth 标签
    如果是标准目标检测：绘制绿色水平外接矩形框（HBB）
    如果是旋转目标检测/分割：绘制红色多边形旋转框（OBB）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在加载标注文件: {ann_file} ...")
    with open(ann_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 1. 建立 image_id 到 image_info 的映射字典
    images_dict = {img['id']: img for img in dataset['images']}
    
    # 2. 将所有的标注 (annotations) 按 image_id 进行分组
    ann_dict = {}
    for ann in dataset['annotations']:
        image_id = ann['image_id']
        if image_id not in ann_dict:
            ann_dict[image_id] = []
        ann_dict[image_id].append(ann)
        
    print(f"总计图片数量: {len(images_dict)}")
    print(f"总计标注(GT)数量: {len(dataset['annotations'])}")

    # 取前 num_samples 个图片进行可视化测试
    sample_img_ids = list(images_dict.keys())[:num_samples]

    for img_id in sample_img_ids:
        img_info = images_dict[img_id]
        file_name = img_info['file_name']
        
        img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(img_path):
            print(f"警告: 找不到图片 {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # 获取该图所有的GT标注
        anns = ann_dict.get(img_id, [])
        for ann in anns:
            # 绘制水平外接矩形 (HBB) - [x, y, width, height]
            bbox = ann.get('bbox')
            if bbox:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 绿色粗线框
            
            # 绘制多边形/旋转框 (OBB) - 存在的话
            segmentation = ann.get('segmentation')
            if segmentation and isinstance(segmentation, list):
                for seg in segmentation:
                    poly = np.array(seg, dtype=np.int32).reshape((-1, 2))
                    cv2.polylines(img, [poly], isClosed=True, color=(0, 0, 255), thickness=2)  # 红色多边形

        # 保存结果
        out_path = os.path.join(output_dir, f"gt_{file_name}")
        cv2.imwrite(out_path, img)
        print(f"已保存GT可视化结果至 --> {out_path}")

if __name__ == "__main__":
    # 根据你使用的 HRSC-coco 目录结构推测
    IMG_DIR = "/cloud/cloud-ssd1/collected_files/ljj/dataset/HRSC-coco/val/"
    
    # 【注意】：请根据你的实际情况修改下面的 JSON 标注文件路径！
    # 通常命名为 instances_val.json, val.json, 或类似名称
    ANN_FILE = "/cloud/cloud-ssd1/collected_files/ljj/dataset/HRSC-coco/annotations/instances_val.json" 
    
    # 结果保存输出的文件夹
    OUTPUT_DIR = "/cloud/cloud-ssd1/collected_files/ljj/TPA/results/HRSC-GT/"
    
    if os.path.exists(ANN_FILE):
        load_and_visualize_hrsc_coco_gt(IMG_DIR, ANN_FILE, OUTPUT_DIR, num_samples=460)
        print("\n完成！")
    else:
        print(f"错误: 找不到标注文件 {ANN_FILE} ！\n请在这个脚本中修改 ANN_FILE 变量，写上你真实的 JSON 标签文件路径！")
