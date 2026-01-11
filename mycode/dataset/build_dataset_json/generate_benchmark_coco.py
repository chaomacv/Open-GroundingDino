import json
import os
import glob
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================

# 1. 原始数据集根目录 (LabelMe 格式)
RAW_DATA_ROOT = "/opt/data/private/xjx/RailMind/database/test/基准测试_1229/基准测试数据集"

# 2. 输出文件名
OUTPUT_JSON_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/benchmark.json"

# 3. 编码映射表 (Code -> Label Name)
CODE_TO_LABEL = {
    "1_1_2_1": "fastener missing",
    "1_1_2_2": "fastener crack",
    "1_4_1_1": "plate rust",
    "1_4_2_2": "column rust",
    "1_4_4_1": "mortar aging",
    "1_5_3_1": "nut missing",
    "1_5_3_6": "coating rust",
    "1_5_3_8": "coating peeling",
    "1_5_4_2": "guard rust",
    "2_1_5_2": "nest",
    "3_1_2_1": "antenna nut loose",
    "3_1_3_1": "nest",
    "4_1_2_1": "plastic film",
    "4_1_4_1": "rubbish"
}

# 4. 超级兼容的类别 ID 表
# 包含了：新标准名、旧下划线名、JSON中的中间态名字
LABEL_TO_ID = {
    # 0: insulator
    "insulator": 0,
    
    # 1: bird protection
    "bird_protection": 1, "bird protection": 1, 
    
    # 2: fixed pulley
    "fixed_pulley": 2, "fixed pulley": 2,
    
    # 3: nest
    "nest": 3,
    
    # 4: normal nut
    "nut_normal": 4, "nut normal": 4, "normal nut": 4,
    
    # 5: rusty nut
    "nut_rust": 5, "nut rust": 5, "rusty nut": 5,
    
    # 6: missing nut
    "nut_missing": 6, "nut missing": 6, "missing nut": 6,
    
    # 7: rust
    "rust": 7,
    
    # 8: rusty guard (guard rust)
    "guard_rust": 8, "guard rust": 8, "rusty guard": 8,
    
    # 9: rusty coating (coating rust)
    "coating_rust": 9, "coating rust": 9, "rusty coating": 9,
    
    # 10: peeling coating
    "coating_peeling": 10, "coating peeling": 10, "peeling coating": 10,
    
    # 11: fastener
    "fastener": 11,
    
    # 12: missing fastener
    "fastener_missing": 12, "fastener missing": 12, "missing fastener": 12,
    
    # 13: slab crack
    "slab_crack": 13, "slab crack": 13,
    
    # 14: broken fastener (fastener crack)
    "fastener_crack": 14, "fastener crack": 14, "broken fastener": 14,
    
    # 15: rubbish
    "rubbish": 15,
    
    # 16: plastic film
    "plastic_film": 16, "plastic film": 16,
    
    # 17: normal column
    "column_normal": 17, "column normal": 17, "normal column": 17,
    
    # 18: normal mortar
    "mortar_normal": 18, "mortar normal": 18, "normal mortar": 18,
    
    # 19: rusty column (column rust) -> 💥之前报错的高发区
    "column_rust": 19, "column rust": 19, "rusty column": 19,
    
    # 20: aging mortar (mortar aging)
    "mortar_aging": 20, "mortar aging": 20, "aging mortar": 20,
    
    # 21: single nut
    "single_nut": 21, "single nut": 21,
    
    # 22: rusty plate (plate rust)
    "plate_rust": 22, "plate rust": 22, "rusty plate": 22,
    
    # 23: normal tower nut
    "tower_nut_normal": 23, "tower nut normal": 23, "normal tower nut": 23,
    
    # 24: normal antenna nut
    "antenna_nut_normal": 24, "antenna nut normal": 24, "normal antenna nut": 24,
    
    # 25: loose antenna nut
    "antenna_nut_loose": 25, "antenna nut loose": 25, "loose antenna nut": 25,
    
    # 26: car
    "car": 26,
    
    # 27: cement room
    "cement_room": 27, "cement room": 27,
    
    # 28: asbestos tile
    "asbestos_tile": 28, "asbestos tile": 28,
    
    # 29: color steel tile
    "color_steel_tile": 29, "color steel tile": 29,
    
    # 30: railroad
    "railroad": 30,
    
    # 31: vent
    "vent": 31,
    
    # 32: top
    "top": 32,
    
    # 33: track area
    "track_area": 33, "track area": 33,
    
    # 34: external structure
    "external_structure": 34, "external structure": 34,
    
    # 35: noise barrier
    "noise_barrier": 35, "noise barrier": 35,
    
    # 36: coating blister
    "coating_blister": 36, "coating blister": 36
}

# 用于生成最终 JSON 中 categories 列表的标准名称 (ID -> Clean Name)
ID_TO_CLEAN_NAME = {
    0: "insulator", 1: "bird protection", 2: "fixed pulley", 3: "nest",
    4: "normal nut", 5: "rusty nut", 6: "missing nut", 7: "rust",
    8: "rusty guard", 9: "rusty coating", 10: "peeling coating", 11: "fastener",
    12: "missing fastener", 13: "slab crack", 14: "broken fastener", 15: "rubbish",
    16: "plastic film", 17: "normal column", 18: "normal mortar", 19: "rusty column",
    20: "aging mortar", 21: "single nut", 22: "rusty plate", 23: "normal tower nut",
    24: "normal antenna nut", 25: "loose antenna nut", 26: "car", 27: "cement room",
    28: "asbestos tile", 29: "color steel tile", 30: "railroad", 31: "vent",
    32: "top", 33: "track area", 34: "external structure", 35: "noise barrier",
    36: "coating blister"
}

# ===============================================

def generate_benchmark_json():
    # 1. 准备 COCO 结构
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 填充 categories (使用最干净的标准名)
    for cat_id in sorted(ID_TO_CLEAN_NAME.keys()):
        coco_output["categories"].append({
            "id": cat_id,
            "name": ID_TO_CLEAN_NAME[cat_id],
            "supercategory": "railway"
        })

    image_id_cnt = 0
    ann_id_cnt = 0
    
    print(f"🚀 开始扫描目录: {RAW_DATA_ROOT}")

    # 2. 递归查找所有 .json 文件
    json_files = []
    for root, dirs, files in os.walk(RAW_DATA_ROOT):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    print(f"📄 找到 {len(json_files)} 个 JSON 文件，开始转换...")

    for json_path in tqdm(json_files):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # --- 处理 Image 信息 ---
            rel_dir = os.path.relpath(os.path.dirname(json_path), RAW_DATA_ROOT)
            if rel_dir == ".": rel_dir = ""
            
            image_name = data.get("imagePath")
            if not image_name:
                image_name = os.path.splitext(os.path.basename(json_path))[0] + ".JPG"
            
            # 处理可能的反斜杠 (Windows路径兼容)
            image_name = image_name.replace("\\", "/")
            
            file_name = os.path.join(rel_dir, image_name)
            
            height = data.get("imageHeight")
            width = data.get("imageWidth")
            if height is None or width is None:
                height = 3000
                width = 4000

            coco_output["images"].append({
                "id": image_id_cnt,
                "file_name": file_name,
                "height": height,
                "width": width
            })

            current_image_id = image_id_cnt
            image_id_cnt += 1

            # --- 处理 Annotations (Shapes) ---
            for shape in data.get("shapes", []):
                raw_label = shape.get("label", "")
                points = shape.get("points", [])
                
                # 1. 映射 Label
                # 先看是否在 Code 表里
                if raw_label in CODE_TO_LABEL:
                    query_name = CODE_TO_LABEL[raw_label]
                else:
                    query_name = raw_label

                # 核心：在超级兼容表里查找
                if query_name in LABEL_TO_ID:
                    category_id = LABEL_TO_ID[query_name]
                else:
                    # 最后的挣扎：尝试去空格或替换空格为下划线再试一次
                    try_name = query_name.replace(" ", "_")
                    if try_name in LABEL_TO_ID:
                        category_id = LABEL_TO_ID[try_name]
                    else:
                        print(f"⚠️ 跳过未知标签: '{raw_label}' (在文件 {file_name})")
                        continue
                
                # 2. 转换 BBox
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x_min = min(xs)
                y_min = min(ys)
                x_max = max(xs)
                y_max = max(ys)
                w = x_max - x_min
                h = y_max - y_min

                coco_output["annotations"].append({
                    "id": ann_id_cnt,
                    "image_id": current_image_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": [list(sum(points, []))]
                })
                ann_id_cnt += 1
                
        except Exception as e:
            print(f"❌ 处理文件出错 {json_path}: {e}")

    # 3. 保存结果
    output_dir = os.path.dirname(OUTPUT_JSON_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 转换完成！")
    print(f"   - 图片数: {image_id_cnt}")
    print(f"   - 标注数: {ann_id_cnt}")
    print(f"   - 输出文件: {os.path.abspath(OUTPUT_JSON_PATH)}")

if __name__ == "__main__":
    generate_benchmark_json()