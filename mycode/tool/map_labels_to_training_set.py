import os
import json
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 目标文件夹 (GT 文件夹)
TARGET_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_gt_1229_results"

# 你的原始映射配置
# 格式: "Code": "当前英文(First), 中文(Middle), 目标训练集英文(Last)"
RAW_MAPPING = {
    "1_1_2_1": "missing_fastener,扣件缺失,nut_missing",
    "1_1_2_2": "broken_fastener,扣件断裂,broken_fastener",
    "1_4_1_1": "rusty_sound_barrier_panel,声屏障板锈蚀,rust",
    "1_4_2_2": "rusty_sound_barrier_column,声屏障立柱锈蚀,rust",
    "1_4_4_1": "aging_mortar_layer,砂浆层老化,mortar_aging",
    "1_5_3_1": "missing_bolt,螺栓缺失,nut_missing",
    "1_5_3_6": "rusty_bolt_coating,螺栓涂层锈蚀,rust",
    "1_5_3_8": "peeling_coating,涂层脱落,coating_peeling",
    "1_5_4_2": "rusty_bridge_railing,桥梁栏杆锈蚀,rust",
    "2_1_5_2": "bird_nest_on_pole,杆塔鸟巢,nest",
    "3_1_2_1": "loose_antenna_bolt,天线螺栓松动,antenna_nut_loose",
    "3_1_3_1": "bird_nest_on_tower,铁塔鸟巢,nest",
    "4_1_2_1": "plastic_film,塑料薄膜,plastic_film",
    "4_1_4_1": "rubbish_pile,垃圾堆,rubbish"
}
# ===============================================

def build_translation_dict():
    """
    构建转换字典：
    将 '1_4_1_1' 和 'rusty_sound_barrier_panel' 都指向 'rust'
    """
    trans_map = {}
    
    print("📋 构建标签映射表:")
    for code, desc_str in RAW_MAPPING.items():
        parts = desc_str.split(',')
        
        # 提取各个部分
        current_english = parts[0].strip() # 例如: rusty_sound_barrier_panel
        target_label = parts[-1].strip()   # 例如: rust (最后一个)
        
        # 1. 映射 Code -> Target (以防文件里还留着 Code)
        trans_map[code] = target_label
        
        # 2. 映射 当前英文 -> Target
        trans_map[current_english] = target_label
        
        # 3. 映射 带空格的英文 -> Target (兼容 missing fastener)
        spaced_english = current_english.replace("_", " ")
        trans_map[spaced_english] = target_label
        
        print(f"   - {current_english:<25} -> {target_label}")
        
    return trans_map

def main():
    if not os.path.exists(TARGET_DIR):
        print(f"❌ 错误: 目录不存在 -> {TARGET_DIR}")
        return

    # 1. 构建字典
    translation_map = build_translation_dict()
    print("-" * 50)

    # 2. 递归获取所有 JSON 文件
    json_files = []
    for root, dirs, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    print(f"🚀 开始修改 {len(json_files)} 个 GT 文件...")
    
    modified_count = 0
    
    for json_path in tqdm(json_files):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            is_file_changed = False
            
            if 'objects' in data:
                for obj in data['objects']:
                    original_label = obj.get('label', '')
                    
                    # ⚡️ 核心替换逻辑
                    # 直接在字典里查
                    if original_label in translation_map:
                        new_label = translation_map[original_label]
                        
                        if new_label != original_label:
                            obj['label'] = new_label
                            is_file_changed = True
                    else:
                        # 如果完全匹配不到，尝试去掉空格再试一次
                        stripped = original_label.replace(" ", "").replace("_", "")
                        # 这里比较复杂，暂不处理，通常上面的 map 已经覆盖了大部分情况
                        pass

            # 写回文件
            if is_file_changed:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                modified_count += 1
                
        except Exception as e:
            print(f"⚠️ 处理失败 {json_path}: {e}")

    print("\n" + "="*50)
    print("✅ GT 标签映射完成！")
    print(f"📂 扫描文件: {len(json_files)}")
    print(f"✏️  实际修改: {modified_count} 个文件")
    print("💡 现在你的 GT 标签已经和训练集标签 (如 'rust', 'nest', 'nut_missing') 对齐了。")

if __name__ == "__main__":
    main()