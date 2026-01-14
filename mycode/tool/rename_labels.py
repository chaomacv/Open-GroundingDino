import json
import os
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 1. 刚才定义的 2.1 版本新 Label Map (ID -> New Name)
NEW_LABEL_MAP = {
    "0": "insulator", "1": "birdguard", "2": "pulley", "3": "nest",
    "4": "nut", "5": "rustynut", "6": "nonut", "7": "corrosion",
    "8": "rustyfence", "9": "rustypaint", "10": "peeling", "11": "clip",
    "12": "noclip", "13": "fracture", "14": "snappedclip", "15": "debris",
    "16": "plastic", "17": "pole", "18": "cement", "19": "rustypole",
    "20": "agedcement", "21": "uninut", "22": "rustyplate", "23": "towernut",
    "24": "antennanut", "25": "loosenut", "26": "vehicle", "27": "bunker",
    "28": "shingle", "29": "metalroof", "30": "track", "31": "vent",
    "32": "rooftop", "33": "ballast", "34": "infrastructure", "35": "soundwall",
    "36": "blister"
}

# 2. 待处理文件路径
FILES_TO_PROCESS = {
    "jsonl": [
        "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/train_odvg_filtered.jsonl",
        "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/train_odvg.jsonl",],
    "coco_json": [
    ]
}
# ===============================================

def process_jsonl(file_path):
    """ 处理 ODVG 格式的 JSONL 文件 """
    if not os.path.exists(file_path):
        print(f"⚠️ 跳过不存在的文件: {file_path}")
        return

    print(f"🔄 正在处理 JSONL: {file_path}")
    output_path = file_path + ".tmp"
    
    with open(file_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in):
            data = json.loads(line)
            instances = data.get("detection", {}).get("instances", [])
            
            for inst in instances:
                # 获取该实例的 label ID (转为字符串以匹配字典)
                label_id = str(inst.get("label"))
                if label_id in NEW_LABEL_MAP:
                    # 根据 ID 强制修改 category 名称
                    inst["category"] = NEW_LABEL_MAP[label_id]
            
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    os.replace(output_path, file_path)
    print(f"✅ 完成！")

def process_coco(file_path):
    """ 处理 COCO 格式的 JSON 文件 """
    if not os.path.exists(file_path):
        print(f"⚠️ 跳过不存在的文件: {file_path}")
        return

    print(f"🔄 正在处理 COCO JSON: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 修改 categories 列表中的名称
    for cat in data.get("categories", []):
        cat_id = str(cat.get("id"))
        if cat_id in NEW_LABEL_MAP:
            cat["name"] = NEW_LABEL_MAP[cat_id]

    # 2. 检查 annotations 列表 (以防某些代码逻辑依赖 annotation 里的 category_name)
    # COCO 通常只存 category_id，所以主要改 categories 即可。

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ 完成！")

if __name__ == "__main__":
    # 处理 JSONL
    for f_path in FILES_TO_PROCESS["jsonl"]:
        process_jsonl(f_path)
    
    # 处理 COCO JSON
    for f_path in FILES_TO_PROCESS["coco_json"]:
        process_coco(f_path)

    print("\n🎉 标签更名任务全部完成！请确保同步更新你的 label_map.json 文件。")