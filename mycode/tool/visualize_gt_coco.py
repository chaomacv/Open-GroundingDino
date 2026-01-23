import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ================= ⚙️ 配置区域 =================

# 基础输出目录
BASE_OUTPUT_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/batch_eval_results/0113"

# 任务配置列表
TASKS = [
    {
        "name": "vis_test_split_coco_mini", # 输出文件夹名称
        "json_path": "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/test_split_coco_mini.json",
        "image_root": "/opt/data/private/xjx/RailMind/高速铁路无人机图像/FilteredLabeled"
    },
    {
        "name": "vis_benchmark_mini",       # 输出文件夹名称
        "json_path": "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/benchmark_mini.json",
        "image_root": "/opt/data/private/xjx/RailMind/database/test/基准测试_1229/基准测试数据集"
    }
]

# 绘图颜色 (BGR 格式) - 绿色
BOX_COLOR = (0, 255, 0) 
TEXT_COLOR = (0, 0, 0)

# ===============================================

def draw_box_text(img, bbox, label_name):
    """
    在图片上绘制框和标签
    bbox: [x, y, w, h]
    """
    x, y, w, h = map(int, bbox)
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    # 1. 画矩形框
    cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)

    # 2. 准备文字
    text = f"{label_name}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    
    # 获取文字尺寸
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # 3. 画文字背景 (防止文字看不清)
    # 如果文字超出上边界，就画在框内部
    if y1 - text_height - 5 < 0:
        text_y_bg = y1 + text_height + 5
        text_y_txt = y1 + text_height
    else:
        text_y_bg = y1
        text_y_txt = y1 - 5

    cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width, y1), BOX_COLOR, -1)
    
    # 4. 画文字
    cv2.putText(img, text, (x1, y1 - 5), font, font_scale, TEXT_COLOR, thickness)

def process_task(task_cfg):
    json_path = task_cfg["json_path"]
    image_root = task_cfg["image_root"]
    output_dir = os.path.join(BASE_OUTPUT_DIR, task_cfg["name"])

    print(f"\n🚀 开始处理: {os.path.basename(json_path)}")
    print(f"   📂 图片源: {image_root}")
    print(f"   💾 输出到: {output_dir}")

    # 检查输入文件
    if not os.path.exists(json_path):
        print(f"❌ 错误: JSON文件不存在 {json_path}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 建立类别 ID 到 名称 的映射
    # categories: [{'id': 1, 'name': 'bird'}, ...]
    cat_id_to_name = {cat['id']: cat['name'] for cat in data.get('categories', [])}

    # 建立 image_id 到 annotations 的映射 (加速查找)
    img_id_to_anns = {}
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    # 遍历处理图片
    images = data.get('images', [])
    print(f"   📊 共 {len(images)} 张图片")

    for img_info in tqdm(images):
        file_name = img_info['file_name']
        img_id = img_info['id']
        
        # 拼接图片完整路径
        src_path = os.path.join(image_root, file_name)
        
        # 检查图片是否存在
        if not os.path.exists(src_path):
            # print(f"⚠️ 跳过缺失图片: {file_name}")
            continue

        # 读取图片 (处理中文路径可能的问题，虽然 Linux 下通常没事，但用 cv2.imdecode 更稳)
        # img = cv2.imread(src_path) 
        # 下面这种写法支持包含中文的路径
        img = cv2.imdecode(np.fromfile(src_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            print(f"⚠️ 无法读取图片: {src_path}")
            continue

        # 获取该图片的标注
        anns = img_id_to_anns.get(img_id, [])

        # 绘制每一个标注
        for ann in anns:
            bbox = ann['bbox'] # [x, y, w, h]
            cat_id = ann['category_id']
            label_name = cat_id_to_name.get(cat_id, str(cat_id))
            
            draw_box_text(img, bbox, label_name)

        # 保存图片
        # 保持原始文件名结构，或者将斜杠替换为下划线防止创建多级目录
        # 这里为了查看方便，直接保存文件名 (flatten)
        save_name = file_name.replace("/", "_") 
        save_path = os.path.join(output_dir, save_name)
        
        # cv2.imwrite(save_path, img)
        # 支持中文路径的保存写法
        cv2.imencode('.jpg', img)[1].tofile(save_path)

    print(f"✅ 完成！结果已保存在: {output_dir}")

if __name__ == "__main__":
    for task in TASKS:
        process_task(task)