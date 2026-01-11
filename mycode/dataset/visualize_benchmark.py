import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# ================= ⚙️ 配置区域 =================

# 1. 刚刚生成的 COCO 格式 GT 文件
GT_JSON_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/benchmark.json"

# 2. 原始图片根目录 (对应 JSON 中 file_name 的相对路径起点)
IMAGE_ROOT = "/opt/data/private/xjx/RailMind/database/test/基准测试_1229/基准测试数据集"

# 3. 可视化结果保存目录
OUTPUT_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_gt_benchmark"

# 4. 可视化设置
DRAW_TEXT = True        # 是否画类别文字
LINE_THICKNESS = 3      # 框的粗细
FONT_SIZE = 20          # 字体大小 (OpenCV不支持直接设置字号，这里仅作占位，OpenCV用缩放因子)

# ===============================================

def draw_coco_bbox(image, bbox, label_name, color=(0, 255, 0)):
    """
    在图片上画 COCO 格式的框 [x, y, w, h]
    """
    x, y, w, h = bbox
    # 转换为左上角和右下角坐标
    pt1 = (int(x), int(y))
    pt2 = (int(x + w), int(y + h))
    
    # 1. 画框
    cv2.rectangle(image, pt1, pt2, color, LINE_THICKNESS)
    
    # 2. 画标签文字 (带背景底色)
    if DRAW_TEXT:
        text = label_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # 获取文字大小
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 文字背景矩形
        text_bg_pt1 = (pt1[0], pt1[1] - text_height - 5)
        text_bg_pt2 = (pt1[0] + text_width, pt1[1])
        
        # 防止文字画出图片上边界
        if text_bg_pt1[1] < 0:
            text_bg_pt1 = (pt1[0], pt1[1])
            text_bg_pt2 = (pt1[0] + text_width, pt1[1] + text_height + 5)
            text_pt = (pt1[0], pt1[1] + text_height)
        else:
            text_pt = (pt1[0], pt1[1] - 5)

        cv2.rectangle(image, text_bg_pt1, text_bg_pt2, color, -1) # 实心矩形作为背景
        cv2.putText(image, text, text_pt, font, font_scale, (0, 0, 0), thickness) # 黑色文字

    return image

def main():
    # 0. 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 创建输出目录: {OUTPUT_DIR}")

    # 1. 加载 GT JSON
    print(f"📖 读取 GT 文件: {GT_JSON_PATH}")
    with open(GT_JSON_PATH, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 建立 category id -> name 的映射
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 建立 image id -> annotations 的映射
    img_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    print(f"📊 包含 {len(coco_data['images'])} 张图片，开始可视化...")

    # 2. 遍历每一张图片进行可视化
    # 为了演示效果，这里只处理前 20 张 (你可以去掉 [:20] 来跑全量)
    for img_info in tqdm(coco_data['images']):
        file_name = img_info['file_name']
        img_id = img_info['id']
        
        # 拼接完整路径
        full_image_path = os.path.join(IMAGE_ROOT, file_name)
        
        # 检查图片是否存在
        if not os.path.exists(full_image_path):
            # print(f"⚠️ 图片不存在，跳过: {full_image_path}")
            continue
            
        # 读取图片 (OpenCV 读取默认为 BGR)
        # cv2.imdecode 可以处理中文路径
        img_array = np.fromfile(full_image_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"❌ 无法读取图片: {full_image_path}")
            continue

        # 获取该图的所有标注
        annotations = img_to_anns.get(img_id, [])
        
        # 如果没有标注，也保存一张原图看看
        if not annotations:
            pass 

        # 3. 绘制所有框
        for ann in annotations:
            bbox = ann['bbox'] # COCO格式: [x, y, w, h]
            cat_id = ann['category_id']
            label_name = cat_id_to_name.get(cat_id, "unknown")
            
            # 这里简单地用绿色画框
            image = draw_coco_bbox(image, bbox, label_name, color=(0, 255, 0))

        # 4. 保存结果
        # 保持原始目录结构保存 (可选)，或者扁平化保存
        # 这里为了查看方便，将文件名中的 '/' 替换为 '_' 扁平化保存
        save_name = "vis_gt_" + file_name.replace("/", "_")
        save_path = os.path.join(OUTPUT_DIR, save_name)
        
        # 处理中文路径保存问题
        cv2.imencode('.jpg', image)[1].tofile(save_path)

    print(f"\n✅ 可视化完成！请查看目录: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    from collections import defaultdict # 补充缺失的 import
    main()