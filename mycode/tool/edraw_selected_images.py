import os
import json
import cv2
import numpy as np
import hashlib
from tqdm import tqdm

# ================= ⚙️ 路径配置 (请仔细核对) =================

# 1. 原始图片根目录 (用于读取干净的底图)
RAW_IMAGE_ROOT = "/opt/data/private/xjx/RailMind/高速铁路无人机图像/FilteredLabeled"

# --- 预测组 (Prediction) ---
# 筛选清单 (你手动挑选的 vis_*.jpg 所在的文件夹)
LIST_PRED_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/show"
# 数据源 (存放 vis_*.json 的文件夹)
DATA_PRED_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_test_results_2"
# 输出目录 (保存重新绘制的预测图)
OUTPUT_PRED_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/final_show_pred"

# --- 真值组 (Ground Truth) ---
# 筛选清单 (你手动复制的 gt_*.jpg 所在的文件夹)
LIST_GT_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/gt_show"
# 数据源 (存放 gt_*.json 的文件夹)
DATA_GT_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_gt_results"
# 输出目录 (保存重新绘制的真值图)
OUTPUT_GT_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/final_show_gt"

# ================= 🎨 绘图配置 =================
LINE_THICKNESS = 2
FONT_SCALE = 0.6
TEXT_THICKNESS = 1
# ===============================================

def get_color_for_label(label_name):
    """
    🎨 核心颜色算法：根据类别名称生成固定的颜色
    原理：对字符串做 MD5 哈希，取前 3 位转成 RGB
    效果：'nut' 永远是同一个颜色，无论在哪个脚本运行
    """
    hash_object = hashlib.md5(label_name.encode())
    hex_dig = hash_object.hexdigest()
    
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    
    return (b, g, r) # OpenCV 使用 BGR 顺序

def draw_boxes(image_path, json_path, output_path, is_gt=False):
    """
    通用绘图函数
    is_gt=True: 只画类别，不画分数
    is_gt=False: 画类别 + 分数
    """
    # 1. 读取 JSON 数据
    if not os.path.exists(json_path):
        print(f"⚠️ 缺失 JSON 文件: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 读取原始底图 (保证画面干净)
    # JSON 里通常记录了原始文件名，我们利用它去 RAW_IMAGE_ROOT 找
    raw_filename = data.get('file_name', '')
    if not raw_filename:
        print(f"⚠️ JSON 中未找到文件名: {json_path}")
        return

    full_image_path = os.path.join(RAW_IMAGE_ROOT, raw_filename)
    if not os.path.exists(full_image_path):
        # 尝试备用方案：如果 JSON 里的路径不对，直接用原始图片名在 ROOT 找
        full_image_path = os.path.join(RAW_IMAGE_ROOT, os.path.basename(raw_filename))
        if not os.path.exists(full_image_path):
            print(f"❌ 找不到原始底图: {full_image_path}")
            return

    # 使用 OpenCV 读取
    # 注意：cv2.imdecode 可以处理中文路径
    image = cv2.imdecode(np.fromfile(full_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"❌ 图片读取失败: {full_image_path}")
        return

    # 3. 开始绘制
    objects = data.get('objects', [])
    for obj in objects:
        # 获取 Label
        label = obj.get('label', 'unknown')
        
        # 获取颜色 (统一算法)
        color = get_color_for_label(label)
        
        # 获取坐标 (优先找 pixel_xyxy)
        box = obj.get('box_pixel_xyxy', obj.get('bbox_xyxy', None))
        
        # 如果是 GT 且只有 xywh，需要转换
        if box is None and 'box_coco_xywh' in obj:
            x, y, w, h = obj['box_coco_xywh']
            box = [x, y, x + w, y + h]
            
        if box is None:
            continue
            
        x1, y1, x2, y2 = map(int, box)

        # 绘制矩形
        cv2.rectangle(image, (x1, y1), (x2, y2), color, LINE_THICKNESS)

        # 准备文字
        if is_gt:
            text = f"{label}" # GT 只写名字
        else:
            score = obj.get('score', 0.0)
            text = f"{label} {score:.2f}" # Pred 写名字+分数

        # 绘制文字背景
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)
        cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        
        # 绘制白色文字
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), TEXT_THICKNESS)

    # 4. 保存结果
    cv2.imwrite(output_path, image)

def process_folder(list_dir, data_dir, output_dir, file_prefix, is_gt):
    """
    处理流程封装
    list_dir: 放着 jpg 的文件夹 (用作筛选清单)
    data_dir: 放着 json 的文件夹 (数据源)
    file_prefix: "vis_" 或 "gt_"
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 创建输出目录: {output_dir}")

    # 获取筛选清单 (只看 jpg)
    files = [f for f in os.listdir(list_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"🚀 开始处理 {len(files)} 张图片 (模式: {'GT' if is_gt else 'Pred'})...")

    for filename in tqdm(files):
        # filename 例如: vis_123.jpg 或 gt_123.jpg
        
        # 1. 推导 JSON 文件名
        # 假设图片和 JSON 同名，只是后缀不同
        name_no_ext = os.path.splitext(filename)[0] # vis_123
        json_filename = name_no_ext + ".json"       # vis_123.json
        
        json_path = os.path.join(data_dir, json_filename)
        output_path = os.path.join(output_dir, filename)
        
        # 2. 调用绘图
        draw_boxes(filename, json_path, output_path, is_gt=is_gt)

def main():
    # 1. 处理预测组 (Prediction)
    print("\n🔵 正在重绘预测结果 (包含置信度)...")
    process_folder(
        list_dir=LIST_PRED_DIR,
        data_dir=DATA_PRED_DIR,
        output_dir=OUTPUT_PRED_DIR,
        file_prefix="vis_",
        is_gt=False
    )

    # 2. 处理真值组 (Ground Truth)
    print("\n🟢 正在重绘真值结果 (仅类别)...")
    process_folder(
        list_dir=LIST_GT_DIR,
        data_dir=DATA_GT_DIR,
        output_dir=OUTPUT_GT_DIR,
        file_prefix="gt_",
        is_gt=True
    )

    print("\n✅ 所有重绘任务完成！")
    print(f"👉 预测图见: {OUTPUT_PRED_DIR}")
    print(f"👉 真值图见: {OUTPUT_GT_DIR}")

if __name__ == "__main__":
    main()