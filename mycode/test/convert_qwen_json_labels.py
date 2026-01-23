import json
import os
import shutil
import argparse

# ================= 1. 定义映射关系 =================

# 原始标签 (Old Name -> ID)
OLD_NAME_TO_ID = {
    "insulator": 0,
    "bird_protection": 1,
    "fixed_pulley": 2,
    "nest": 3,
    "nut_normal": 4,
    "nut_rust": 5,
    "nut_missing": 6,
    "rust": 7,
    "guard_rust": 8,
    "coating_rust": 9,
    "coating_peeling": 10,
    "fastener": 11,
    "fastener_missing": 12,
    "slab_crack": 13,
    "fastener_crack": 14,
    "rubbish": 15,
    "plastic_film": 16,
    "column_normal": 17,
    "mortar_normal": 18,
    "column_rust": 19,
    "mortar_aging": 20,
    "single_nut": 21,
    "plate_rust": 22,
    "tower_nut_normal": 23,
    "antenna_nut_normal": 24,
    "antenna_nut_loose": 25,
    "car": 26,
    "cement_room": 27,
    "asbestos_tile": 28,
    "color_steel_tile": 29,
    "railroad": 30,
    "vent": 31,
    "top": 32,
    "track_area": 33,
    "external_structure": 34,
    "noise_barrier": 35,
    "coating_blister": 36
}

# 目标标签 (ID -> New Name)
ID_TO_NEW_NAME = {
    0: "insulator",
    1: "birdguard",
    2: "pulley",
    3: "nest",
    4: "nut",
    5: "rustynut",
    6: "nonut",
    7: "corrosion",
    8: "rustyfence",
    9: "rustypaint",
    10: "peeling",
    11: "clip",
    12: "noclip",
    13: "fracture",
    14: "snappedclip",
    15: "debris",
    16: "plastic",
    17: "pole",
    18: "cement",
    19: "rustypole",
    20: "agedcement",
    21: "uninut",
    22: "rustyplate",
    23: "towernut",
    24: "antennanut",
    25: "loosenut",
    26: "vehicle",
    27: "bunker",
    28: "shingle",
    29: "metalroof",
    30: "track",
    31: "vent",
    32: "rooftop",
    33: "ballast",
    34: "infrastructure",
    35: "soundwall",
    36: "blister"
}

# 自动生成直接映射字典: { "column_rust": "rustypole", ... }
LABEL_MAPPING = {}
for old_name, idx in OLD_NAME_TO_ID.items():
    if idx in ID_TO_NEW_NAME:
        LABEL_MAPPING[old_name] = ID_TO_NEW_NAME[idx]

# ================= 2. 转换逻辑 =================

def convert_labels(file_path):
    print(f"🚀 开始处理文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    # 1. 创建备份
    backup_path = file_path + ".bak"
    shutil.copy2(file_path, backup_path)
    print(f"📦 已创建备份: {backup_path}")

    # 2. 读取 JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. 遍历并修改
    details = data.get("evaluation", {}).get("details", [])
    if not details:
        print("⚠️ 警告: 未找到 evaluation.details 字段")
        return

    changed_count = 0
    total_labels_checked = 0

    for item in details:
        # 修改 pred_anomaly_class
        if "pred_anomaly_class" in item:
            new_preds = []
            for label in item["pred_anomaly_class"]:
                total_labels_checked += 1
                if label in LABEL_MAPPING:
                    new_preds.append(LABEL_MAPPING[label])
                    changed_count += 1
                else:
                    # 如果已经在新列表中（可能是已经跑过脚本了），或者不在映射表中，保留原样
                    if label in ID_TO_NEW_NAME.values(): 
                        new_preds.append(label) # 已经是新名称，无需修改
                    else:
                        print(f"❓ 未知标签 (保留原样): {label}")
                        new_preds.append(label)
            item["pred_anomaly_class"] = new_preds

        # 修改 gt_anomaly_class (如果有的话)
        if "gt_anomaly_class" in item:
            new_gts = []
            for label in item["gt_anomaly_class"]:
                if label in LABEL_MAPPING:
                    new_gts.append(LABEL_MAPPING[label])
                else:
                    if label in ID_TO_NEW_NAME.values():
                        new_gts.append(label)
                    else:
                        new_gts.append(label)
            item["gt_anomaly_class"] = new_gts

    # 4. 保存回原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 转换完成!")
    print(f"📊 检查标签总数: {total_labels_checked}")
    print(f"🔄 成功替换个数: {changed_count}")
    print(f"💾 文件已更新: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default="/opt/data/private/xjx/RailMind/Qwen3_results_testset/agent/qwen3-8b-full/batch_summary_1768142978.json", help="需要修改标签的 Qwen 结果 JSON 路径")
    args = parser.parse_args()

    convert_labels(args.json_path)