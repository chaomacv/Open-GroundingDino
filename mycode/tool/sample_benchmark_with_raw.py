import os
import json
import random
import shutil
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================

# 1. 模型推理结果目录 (源)
SOURCE_DIRS = [
    "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/batch_eval_results/0115/benchmark.json0.35/model3_only_fullneg_GTLabels",
    # "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/batch_eval_results/0115/benchmark.json0.35/model4_only_posonly_GTLabels"
]

# 2. 采样结果保存目录 (目标)
OUTPUT_ROOT = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/batch_eval_results/0115/sampled_results"

# 3. 原始数据集根目录 (用于读取图片像素)
DATASET_ROOT = "/opt/data/private/xjx/RailMind/database/test/基准测试_1229/基准测试数据集"

# 4. Ground Truth 标注文件路径
GT_JSON_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/benchmark.json"

# 5. 每个场景抽取的数量
SAMPLE_NUM = 10

# ===============================================

def load_gt_data(json_path):
    """ 加载 COCO 格式的 GT 数据，建立索引 """
    print(f"📖 正在加载 GT 标注文件: {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    
    # 1. 建立类别 ID -> Name 映射
    id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
    
    # 2. 建立 Image ID -> File Name 映射
    img_id_to_name = {img['id']: img['file_name'] for img in coco['images']}
    
    # 3. 建立 File Name -> Annotations 列表映射
    # 结果格式: {"声屏障/1.jpg": [{'bbox': [x,y,w,h], 'category_id': 1}, ...]}
    file_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id in img_id_to_name:
            file_name = img_id_to_name[img_id]
            ann['category_name'] = id_to_name.get(ann['category_id'], 'unknown')
            file_to_anns[file_name].append(ann)
            
    print(f"✅ GT 数据加载完毕，包含 {len(file_to_anns)} 张有标注的图片信息。")
    return file_to_anns

def draw_ground_truth(img_path, annotations, save_path):
    """ 在图片上绘制 GT 框 """
    # 读取图片 (处理中文路径可能需要特殊手段，但在 Linux 下通常 cv2.imread 直接支持)
    image = cv2.imread(img_path)
    if image is None:
        return False

    # 绘制框和标签
    for ann in annotations:
        bbox = ann['bbox'] # COCO 格式: [x_min, y_min, width, height]
        x, y, w, h = [int(v) for v in bbox]
        label = ann['category_name']
        
        # 颜色: 绿色 (BGR: 0, 255, 0)
        color = (0, 255, 0) 
        thickness = 2
        
        # 画矩形
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
        # 画标签背景和文字
        font_scale = 0.6
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        cv2.rectangle(image, (x, y - text_h - 5), (x + text_w, y), color, -1) # 实心背景
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

    cv2.imwrite(save_path, image)
    return True

def sample_files():
    # 1. 预加载 GT 数据
    if not os.path.exists(GT_JSON_PATH):
        print(f"❌ 错误: 找不到 GT 文件 {GT_JSON_PATH}")
        return
    gt_lookup = load_gt_data(GT_JSON_PATH)

    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    for src_dir in SOURCE_DIRS:
        if not os.path.exists(src_dir):
            print(f"❌ 找不到源目录: {src_dir}")
            continue

        model_name = os.path.basename(src_dir)
        print(f"\n🚀 正在处理模型结果: {model_name}")
        
        # --- 按场景归类 ---
        scene_map = defaultdict(list)
        all_files = os.listdir(src_dir)
        json_files = [f for f in all_files if f.endswith(".json")]
        
        for json_file in tqdm(json_files, desc="解析场景"):
            try:
                with open(os.path.join(src_dir, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # file_name example: "声屏障/xxx.JPG"
                original_filename = data.get("file_name", "")
                if "/" in original_filename:
                    scene = original_filename.split("/")[0]
                else:
                    scene = "Uncategorized"
                
                scene_map[scene].append((json_file, original_filename))
            except:
                pass

        # --- 采样处理 ---
        print(f"   🎲 开始采样与可视化绘制...")
        
        for scene, file_list in scene_map.items():
            count = min(len(file_list), SAMPLE_NUM)
            sampled_items = random.sample(file_list, count)
            
            # 结果目录: output/模型名/场景名/
            save_dir = os.path.join(OUTPUT_ROOT, model_name, scene)
            os.makedirs(save_dir, exist_ok=True)
            
            print(f"      ├─ 场景 [{scene}]: 处理 {count} 张")
            
            for j_file, raw_rel_path in sampled_items:
                base_name = os.path.splitext(j_file)[0] # vis_xxx
                
                # A. 复制 JSON 结果
                shutil.copy2(os.path.join(src_dir, j_file), os.path.join(save_dir, j_file))
                
                # B. 复制 模型预测图 (vis_xxx.jpg)
                for ext in ['.jpg', '.JPG', '.png']:
                    vis_src = os.path.join(src_dir, base_name + ext)
                    if os.path.exists(vis_src):
                        shutil.copy2(vis_src, os.path.join(save_dir, base_name + ext))
                        break
                
                # C. 处理 原始图 & GT图
                src_raw_abs = os.path.join(DATASET_ROOT, raw_rel_path)
                
                if os.path.exists(src_raw_abs):
                    # C-1. 保存纯原图 (raw_xxx.jpg)
                    raw_save_name = f"raw_{os.path.basename(raw_rel_path)}"
                    shutil.copy2(src_raw_abs, os.path.join(save_dir, raw_save_name))
                    
                    # C-2. 绘制并保存 GT 图 (gt_xxx.jpg)
                    # 从查找表中获取标注
                    current_anns = gt_lookup.get(raw_rel_path, [])
                    gt_save_name = f"gt_{os.path.basename(raw_rel_path)}"
                    gt_save_path = os.path.join(save_dir, gt_save_name)
                    
                    # 即使没有标注(负样本)也画一张图(纯图)，方便对比
                    draw_ground_truth(src_raw_abs, current_anns, gt_save_path)
                else:
                    print(f"         ⚠️ 原图缺失: {src_raw_abs}")

    print(f"\n✅ 全部完成！结果已保存在: {OUTPUT_ROOT}")

if __name__ == "__main__":
    sample_files()