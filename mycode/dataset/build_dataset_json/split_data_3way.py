import json
import random
import os

# ================= ⚙️ 配置区域 =================
# 1. 你的原始全量数据文件
INPUT_FILE = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/train_odvg_filtered.jsonl"
# 2. 你的 Label Map 文件
LABEL_MAP_FILE = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/label_map_only.json"

# 3. 输出文件路径
OUTPUT_TRAIN_ODVG = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/train_split_only.jsonl"      # 80% 训练用
OUTPUT_VAL_COCO   = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/val_split_coco_only.json"    # 10% 训练中验证用 (COCO格式)
OUTPUT_TEST_COCO  = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/test_split_coco_only.json"   # 10% 最终测试用 (COCO格式)

# 4. 划分比例 (验证集和测试集各占多少)
VAL_RATIO = 0.1   # 10%
TEST_RATIO = 0.1  # 10%
# 剩下的 80% 自动归为训练集

SEED = 42
# ==============================================

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"💾 已保存 ODVG 文件: {path} ({len(data)} 条)")

def convert_to_coco(odvg_data, label_map_path, output_path, desc):
    print(f"🔄 正在转换 {desc} 为 COCO 格式...")
    
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    # 翻转 Label Map
    first_val = list(label_map.values())[0]
    if isinstance(first_val, int):
        id_to_name = {v: k for k, v in label_map.items()}
        name_to_id = label_map
    else:
        id_to_name = {int(k): v for k, v in label_map.items()}
        name_to_id = {v: int(k) for k, v in label_map.items()}

    coco_output = {
        "info": {"description": desc},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    for cat_id, cat_name in id_to_name.items():
        coco_output["categories"].append({"id": cat_id, "name": cat_name, "supercategory": "none"})

    anno_id_count = 1
    for index, data in enumerate(odvg_data):
        image_info = {
            "id": index, 
            "file_name": data["filename"],
            "width": data["width"],
            "height": data["height"]
        }
        coco_output["images"].append(image_info)

        if "detection" in data and "instances" in data["detection"]:
            for inst in data["detection"]["instances"]:
                x1, y1, x2, y2 = inst["bbox"]
                w, h = x2 - x1, y2 - y1
                
                raw_label = inst["label"]
                category_id = raw_label
                if isinstance(raw_label, str) and not raw_label.isdigit():
                   if raw_label in name_to_id:
                       category_id = name_to_id[raw_label]

                anno = {
                    "id": anno_id_count,
                    "image_id": index,
                    "category_id": int(category_id),
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": []
                }
                coco_output["annotations"].append(anno)
                anno_id_count += 1
                
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False)
    print(f"💾 已保存 {desc} COCO 文件: {output_path}")

def main():
    random.seed(SEED)
    print(f"🚀 开始读取原始文件: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total = len(lines)
    random.shuffle(lines) # 打乱
    data_list = [json.loads(line) for line in lines]
    
    # 计算切分点
    val_count = int(total * VAL_RATIO)
    test_count = int(total * TEST_RATIO)
    train_count = total - val_count - test_count
    
    # 切分数据
    train_data = data_list[:train_count]
    val_data = data_list[train_count : train_count + val_count]
    test_data = data_list[train_count + val_count :]
    
    print(f"✂️  总数: {total}")
    print(f"   - 训练集 (80%): {len(train_data)}")
    print(f"   - 验证集 (10%): {len(val_data)}")
    print(f"   - 测试集 (10%): {len(test_data)}")
    
    # 1. 保存训练集 (ODVG格式，用于训练)
    save_jsonl(train_data, OUTPUT_TRAIN_ODVG)
    
    # 2. 保存验证集 (转COCO格式，用于边训练边评估)
    convert_to_coco(val_data, LABEL_MAP_FILE, OUTPUT_VAL_COCO, "Validation Set")

    # 3. 保存测试集 (转COCO格式，用于最后大考)
    convert_to_coco(test_data, LABEL_MAP_FILE, OUTPUT_TEST_COCO, "Test Set")

if __name__ == "__main__":
    main()