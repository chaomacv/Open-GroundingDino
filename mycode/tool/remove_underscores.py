import json
import os
from tqdm import tqdm

# ================= ⚙️ 文件路径配置 =================
# 1. 训练集 (JSONL 格式)
TRAIN_FILE = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/train_split_cleaned.jsonl"

# 2. 验证/测试集 (COCO 格式)
VAL_FILE = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/val_split_coco.json"
TEST_FILE = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/test_split_coco.json"

# 3. Label Map (字典格式)
LABEL_MAP_FILE = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/label_map_14cls.json"
# ===================================================

def process_jsonl(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在，跳过: {file_path}")
        return

    print(f"🔄 正在处理 JSONL (训练集): {file_path}")
    temp_file = file_path + ".tmp"
    
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f_in, \
         open(temp_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in):
            data = json.loads(line)
            instances = data.get("detection", {}).get("instances", [])
            
            for inst in instances:
                original = inst.get("category", "")
                if "_" in original:
                    # 替换核心逻辑
                    inst["category"] = original.replace("_", " ")
                    count += 1
            
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    # 覆盖原文件
    os.replace(temp_file, file_path)
    print(f"✅ JSONL 处理完成，替换了 {count} 个标签。")

def process_coco(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在，跳过: {file_path}")
        return

    print(f"🔄 正在处理 COCO (验证/测试集): {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    count = 0
    # COCO 格式只需要修改 categories 里的 name
    for cat in data.get("categories", []):
        original = cat["name"]
        if "_" in original:
            cat["name"] = original.replace("_", " ")
            count += 1
            
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ COCO 处理完成，更新了 {count} 个类别定义。")

def process_label_map(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在，跳过: {file_path}")
        return

    print(f"🔄 正在处理 Label Map: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    new_data = {}
    count = 0
    for k, v in data.items():
        if "_" in v:
            new_data[k] = v.replace("_", " ")
            count += 1
        else:
            new_data[k] = v
            
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Label Map 处理完成，更新了 {count} 个条目。")

if __name__ == "__main__":
    # 执行替换
    process_jsonl(TRAIN_FILE)
    process_coco(VAL_FILE)
    process_coco(TEST_FILE)
    process_label_map(LABEL_MAP_FILE)
    
    print("\n🎉 所有文件中的下划线已成功替换为空格！请重新启动训练。")