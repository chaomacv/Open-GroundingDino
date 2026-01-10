import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 1. 源数据目录 (包含子文件夹)
SOURCE_ROOT = "/opt/data/private/xjx/RailMind/database/test/基准测试_1229/基准测试数据集"

# 2. 目标输出目录
OUTPUT_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_gt_1229_results"

# 3. 标签映射表 (代码 -> 英文)
LABEL_MAP = {
    "1_1_2_1": "missing fastener",
    "1_1_2_2": "broken fastener",
    "1_4_1_1": "rusty sound barrier panel",
    "1_4_2_2": "rusty sound barrier column",
    "1_4_4_1": "aging mortar layer",
    "1_5_3_1": "missing bolt",
    "1_5_3_6": "rusty bolt coating",
    "1_5_3_8": "peeling coating",
    "1_5_4_2": "rusty bridge railing",
    "2_1_5_2": "bird nest on pole",
    "3_1_2_1": "loose antenna bolt",
    "3_1_3_1": "bird nest on tower",
    "4_1_2_1": "plastic film",
    "4_1_4_1": "rubbish pile"
}

# 4. 可视化颜色 (BGR格式，这里用绿色表示GT)
GT_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
# ===============================================

def convert_labelme_to_odvg(labelme_json_path, image_path, output_dir):
    """
    核心转换函数
    """
    try:
        # 1. 读取 LabelMe JSON
        with open(labelme_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ JSON 读取失败: {labelme_json_path} - {e}")
        return

    # 2. 读取图片 (为了画图和获取准确尺寸)
    # 优先使用 OpenCV 读取，处理中文路径需注意，这里假设路径无特殊字符或系统支持
    image = cv2.imread(image_path)
    if image is None:
        # 尝试处理中文路径
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    if image is None:
        print(f"❌ 图片读取失败: {image_path}")
        return

    img_h, img_w = image.shape[:2]
    file_name = os.path.basename(image_path)
    
    # 3. 准备目标数据结构
    gt_objects = []

    shapes = data.get('shapes', [])
    for shape in shapes:
        label_code = shape.get('label', '')
        points = shape.get('points', [])
        
        # 过滤无效数据
        if not label_code or not points:
            continue
            
        # 标签映射 (Code -> English)
        label_text = LABEL_MAP.get(label_code, label_code) # 如果没在字典里，暂且保留原Code
        
        # 提取坐标 (LabelMe 的 points 可能是 [[x1,y1], [x2,y2]])
        # 确保 x1 < x2, y1 < y2
        pts = np.array(points)
        x1 = min(pts[:, 0])
        y1 = min(pts[:, 1])
        x2 = max(pts[:, 0])
        y2 = max(pts[:, 1])
        
        # 限制在图片范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        # 计算宽高
        w_pixel = x2 - x1
        h_pixel = y2 - y1
        
        # 计算归一化坐标 cx, cy, w, h
        norm_cx = (x1 + w_pixel / 2) / img_w
        norm_cy = (y1 + h_pixel / 2) / img_h
        norm_w = w_pixel / img_w
        norm_h = h_pixel / img_h
        
        gt_objects.append({
            "label": label_text,
            "score": 1.0, # GT 置信度为 1
            "box_norm_cxcywh": [norm_cx, norm_cy, norm_w, norm_h],
            "box_pixel_xyxy": [int(x1), int(y1), int(x2), int(y2)]
        })

        # --- 可视化绘制 ---
        # 画矩形
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), GT_COLOR, 2)
        
        # 画标签文字
        text = f"{label_text}"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 文字背景
        cv2.rectangle(image, (int(x1), int(y1) - text_h - 5), (int(x1) + text_w, int(y1)), GT_COLOR, -1)
        # 文字
        cv2.putText(image, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

    # 4. 保存转换后的 JSON
    # 构造输出文件名: gt_原始文件名.json
    base_name = os.path.splitext(file_name)[0]
    out_json_name = f"gt_{base_name}.json"
    out_json_path = os.path.join(output_dir, out_json_name)
    
    target_json_data = {
        "file_name": file_name,
        "original_path": image_path, # 记录原始路径
        "height": img_h,
        "width": img_w,
        "objects": gt_objects
    }
    
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(target_json_data, f, indent=4, ensure_ascii=False)

    # 5. 保存可视化后的图片
    # 构造输出文件名: gt_原始文件名.jpg
    out_img_name = f"gt_{base_name}.jpg" # 统一转为jpg以防万一，或者保持后缀
    # 为了保险，直接用 splitext 保留原后缀比较好，但你要求对齐，这里统一加 gt_ 前缀
    out_img_name = f"gt_{file_name}" 
    out_img_path = os.path.join(output_dir, out_img_name)
    
    cv2.imwrite(out_img_path, image)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 创建输出目录: {OUTPUT_DIR}")

    print(f"🔍 开始扫描源目录: {SOURCE_ROOT}")
    
    # 收集任务列表
    tasks = []
    for root, dirs, files in os.walk(SOURCE_ROOT):
        for file in files:
            # 寻找 .json 文件 (LabelMe 标注)
            if file.endswith(".json"):
                # 排除非标注文件 (如 label_map.json 或 模型生成的 vis_)
                if file == "label_map.json" or file.startswith("vis_"):
                    continue
                
                json_path = os.path.join(root, file)
                
                # 寻找同名的图片文件
                # LabelMe JSON 通常对应同名的 jpg/png
                base_name = os.path.splitext(file)[0]
                
                # 尝试常见的图片后缀
                found_img = False
                for ext in ['.JPG', '.jpg', '.png', '.jpeg', '.BMP']:
                    img_name = base_name + ext
                    img_path = os.path.join(root, img_name)
                    if os.path.exists(img_path):
                        tasks.append((json_path, img_path))
                        found_img = True
                        break
                
                if not found_img:
                    # 尝试从 JSON 的 imagePath 字段读取 (虽然那个字段通常只有文件名)
                    pass 

    print(f"📊 找到 {len(tasks)} 对 (JSON+图片) 数据。开始转换...")

    for json_path, img_path in tqdm(tasks):
        convert_labelme_to_odvg(json_path, img_path, OUTPUT_DIR)

    print(f"\n✅ 全部完成！")
    print(f"📂 结果保存在: {OUTPUT_DIR}")
    print(f"   包含内容: gt_*.json (标准化格式) 和 gt_*.jpg (可视化图片)")

if __name__ == "__main__":
    main()