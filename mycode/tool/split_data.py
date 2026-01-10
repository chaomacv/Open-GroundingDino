import json
import random
import os

# ================= ⚙️ 配置区域 =================
# 1. 你的原始全量数据文件
INPUT_FILE = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/train_odvg.jsonl"
# 2. 你的 Label Map 文件 (用于 COCO 转换)
LABEL_MAP_FILE = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/label_map.json"

# 3. 输出文件路径
OUTPUT_TRAIN_ODVG = "train_split.jsonl"    # 切分后的训练集
OUTPUT_VAL_ODVG   = "val_split.jsonl"      # 切分后的验证集 (ODVG格式备份)
OUTPUT_VAL_COCO   = "val_split_coco.json"  # 切分后的验证集 (COCO格式，用于评估)

# 4. 划分比例 (例如 0.9 表示 90% 训练，10% 验证)
TRAIN_RATIO = 0.9 
SEED = 42  # 随机种子，保证每次运行切分结果一致
# ==============================================

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"💾 已保存 ODVG 文件: {path} ({len(data)} 条)")

def convert_to_coco(odvg_data, label_map_path, output_path):
    print("🔄 正在将验证集转换为 COCO 格式...")
    
    # 读取 Label Map
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    # 翻转 label_map 确保是 {ID: Name}
    first_val = list(label_map.values())[0]
    if isinstance(first_val, int):
        id_to_name = {v: k for k, v in label_map.items()}
        name_to_id = label_map
    else:
        id_to_name = {int(k): v for k, v in label_map.items()}
        name_to_id = {v: int(k) for k, v in label_map.items()}

    coco_output = {
        "info": {"description": "Auto Split Validation Set"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 填充 Categories
    for cat_id, cat_name in id_to_name.items():
        coco_output["categories"].append({"id": cat_id, "name": cat_name, "supercategory": "none"})

    anno_id_count = 1
    for index, data in enumerate(odvg_data):
        # Image
        image_info = {
            "id": index,  # 这里的 ID 对应验证集的顺序
            "file_name": data["filename"],
            "width": data["width"],
            "height": data["height"]
        }
        coco_output["images"].append(image_info)

        # Annotation
        if "detection" in data and "instances" in data["detection"]:
            for inst in data["detection"]["instances"]:
                x1, y1, x2, y2 = inst["bbox"]
                w, h = x2 - x1, y2 - y1
                coco_bbox = [x1, y1, w, h]
                
                raw_label = inst["label"]
                category_id = raw_label
                if isinstance(raw_label, str) and not raw_label.isdigit():
                   if raw_label in name_to_id:
                       category_id = name_to_id[raw_label]

                anno = {
                    "id": anno_id_count,
                    "image_id": index,
                    "category_id": int(category_id),
                    "bbox": coco_bbox,
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": []
                }
                coco_output["annotations"].append(anno)
                anno_id_count += 1
                
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False)
    print(f"💾 已保存 COCO 文件: {output_path}")

def main():
    random.seed(SEED)
    print(f"🚀 开始读取原始文件: {INPUT_FILE}")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total = len(lines)
    print(f"📊 原始数据共 {total} 条。打乱顺序中...")
    
    # 随机打乱
    random.shuffle(lines)
    
    # 解析 JSON
    data_list = [json.loads(line) for line in lines]
    
    # 切分
    split_idx = int(total * TRAIN_RATIO)
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]
    
    print(f"✂️  切分比例 {TRAIN_RATIO}: 训练集 {len(train_data)} 条, 验证集 {len(val_data)} 条。")
    
    # 保存训练集
    save_jsonl(train_data, OUTPUT_TRAIN_ODVG)
    
    # 保存验证集 (ODVG)
    save_jsonl(val_data, OUTPUT_VAL_ODVG)
    
    # 转换验证集为 COCO
    convert_to_coco(val_data, LABEL_MAP_FILE, OUTPUT_VAL_COCO)
    
    print("\n✅ 所有工作已完成！")
    print(f"1. 新的训练集路径: {os.path.abspath(OUTPUT_TRAIN_ODVG)}")
    print(f"2. 新的验证集路径: {os.path.abspath(OUTPUT_VAL_COCO)}")

if __name__ == "__main__":
    main()