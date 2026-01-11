import json
import os
from tqdm import tqdm
from collections import Counter

# ================= ⚙️ 配置区域 =================

# 目标文件夹 (将直接修改这里面的文件)
PRED_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/0110_full_test_benchmark"

# 标准类别白名单 (Standard Labels)
VALID_LABELS = [
    "insulator", "bird_protection", "fixed_pulley", "nest", 
    "nut_normal", "nut_rust", "nut_missing", 
    "rust", "guard_rust", "coating_rust", "coating_peeling", 
    "fastener", "fastener_missing", "slab_crack", "fastener_crack", 
    "rubbish", "plastic_film", 
    "column_normal", "mortar_normal", "column_rust", "mortar_aging", 
    "single_nut", "plate_rust", 
    "tower_nut_normal", "antenna_nut_normal", "antenna_nut_loose", 
    "car", "cement_room", "asbestos_tile", "color_steel_tile", 
    "railroad", "vent", "top", "track_area", 
    "external_structure", "noise_barrier", "coating_blister"
]

# 按长度倒序排列，优先匹配长词 (防止 nut_normal 匹配成 nut)
SORTED_VALID_KEYS = sorted(VALID_LABELS, key=len, reverse=True)

# ===============================================

def get_clean_label(raw_label):
    """
    输入原始乱糟糟的标签，返回清洗后的标准标签。
    如果无法识别，返回原始标签。
    """
    if not isinstance(raw_label, str):
        return str(raw_label)
    
    # 1. 预处理：转小写，去空格，去 BERT 特殊符号 ##
    # "fastener fastener" -> "fastenerfastener"
    # "plastic _ film" -> "plastic_film"
    # "##ener" -> "ener"
    processed = raw_label.lower().replace("##", "").replace(" _ ", "_").replace(" ", "_").strip()
    
    # 2. 特殊补丁 (针对特定的 BERT 分词碎片)
    if "ener" in processed and "fast" not in processed:
        return "fastener"
    
    # 3. 白名单包含匹配
    for valid_key in SORTED_VALID_KEYS:
        # 如果处理后的字符串包含了标准词 (例如 nut_normal_nut_normal 包含 nut_normal)
        if valid_key in processed:
            return valid_key
            
    # 4. 如果没匹配上，返回原值 (保留 unknown 状态)
    return raw_label

def main():
    if not os.path.exists(PRED_DIR):
        print(f"❌ 目录不存在: {PRED_DIR}")
        return

    files = [f for f in os.listdir(PRED_DIR) if f.endswith(".json")]
    print(f"📂 准备处理 {len(files)} 个文件...")

    change_log = Counter() # 记录修改了什么
    modified_files_count = 0
    total_objects_count = 0

    for file_name in tqdm(files):
        file_path = os.path.join(PRED_DIR, file_name)
        
        try:
            # 1. 读取
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_modified = False
            objects = data.get('objects', [])
            
            # 2. 修改
            for obj in objects:
                total_objects_count += 1
                old_label = obj.get('label', '')
                
                # 获取清洗后的标签
                new_label = get_clean_label(old_label)
                
                # 如果标签发生了变化，记录并应用
                if new_label != old_label:
                    obj['label'] = new_label # 修改内存中的值
                    change_log[f"{old_label} -> {new_label}"] += 1
                    file_modified = True
            
            # 3. 写入 (仅当文件内容有变动时)
            if file_modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                modified_files_count += 1
                
        except Exception as e:
            print(f"⚠️ 处理出错 {file_name}: {e}")

    # ================= 输出报告 =================
    print("\n" + "="*80)
    print("✅ 修改完成！修改详情如下 (Top 20 变化):")
    print("="*80)
    
    for change, count in change_log.most_common(20):
        print(f"{change:<60} | {count} 次")
        
    print("-" * 80)
    print(f"📂 扫描文件数: {len(files)}")
    print(f"📝 被修改文件数: {modified_files_count}")
    print(f"🏷️ 处理目标总数: {total_objects_count}")
    print(f"🔧 修复标签总数: {sum(change_log.values())}")
    print("="*80)

if __name__ == "__main__":
    main()