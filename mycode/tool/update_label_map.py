import json
import os

# ================= ⚙️ 配置区域 =================
# 输出路径
LABEL_MAP_FILE = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/label_new_map.json"

# 新的映射关系
# Key: 你的四级标签编码 (对应 dataset 中的 category_name 或 id)
# Value: 喂给模型的英文 Prompt (语义必须清晰)
new_label_map = {
    # === 轨道 (Track) ===
    "1_1_2_1": "missing fastener",       # 扣件缺失
    "1_1_2_2": "broken fastener",        # 扣件断裂

    # === 声屏障 (Sound Barrier) ===
    "1_4_1_1": "rusty sound barrier panel",  # 声屏障单元板锈蚀
    "1_4_2_2": "rusty sound barrier column", # 声屏障立柱锈蚀
    "1_4_4_1": "aging mortar layer",         # 砂浆层老化劣化

    # === 钢架桥 (Steel Bridge) ===
    "1_5_3_1": "missing bolt",               # 桥梁螺栓缺失
    "1_5_3_6": "rusty bolt coating",         # 涂层(螺栓)锈蚀
    "1_5_3_8": "peeling coating",            # 涂层脱落
    "1_5_4_2": "rusty bridge railing",       # 桥栏杆锈蚀

    # === 接触网杆 (Catenary Pole) ===
    "2_1_5_2": "bird nest on pole",          # 接触网杆鸟巢 (加 on pole 以示区分)

    # === 铁塔 (Tower) ===
    "3_1_2_1": "loose antenna bolt",         # 天线抱箍螺栓松动
    "3_1_3_1": "bird nest on tower",         # 铁塔鸟巢 (加 on tower 以示区分)

    # === 环境 (Environment) ===
    "4_1_2_1": "plastic film",               # 塑料膜 (轻飘浮物)
    "4_1_4_1": "rubbish pile"                # 垃圾堆积
}
# ===============================================

def main():
    # 确保目录存在
    dir_name = os.path.dirname(LABEL_MAP_FILE)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # 写入 JSON
    print(f"🔄 正在更新 Label Map，共 {len(new_label_map)} 个类别...")
    with open(LABEL_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_label_map, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 更新成功！文件已保存至: {LABEL_MAP_FILE}")
    print("\n📝 生成的 Prompt 预览 (将输入给模型):")
    prompt = " . ".join(new_label_map.values()) + " ."
    print(prompt)

if __name__ == "__main__":
    main()