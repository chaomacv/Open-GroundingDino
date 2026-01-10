import os
import json
import shutil
import random
from tqdm import tqdm
from collections import defaultdict

# ================= ⚙️ 配置区域 =================
# 1. 源数据文件夹
SOURCE_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_gt_1229_results"

# 2. 目标保存文件夹
TARGET_ROOT = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/sampled_gt_check"

# 3. 各场景抽样数量
QUOTAS = {
    "声屏障": 10,
    "接触网杆": 4,
    "环境": 10,
    "轨道": 10,
    "钢架桥": 10,
    "铁塔": 10
}

# 4. 支持的图片后缀列表 (优先查找前面的)
VALID_EXTS = ['.jpg', '.JPG', '.jpeg', '.png', '.BMP']
# ===============================================

def get_scene_from_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        original_path = data.get("original_path", "")
        if not original_path:
            return "Unknown"
        dir_name = os.path.dirname(original_path)
        scene_name = os.path.basename(dir_name)
        return scene_name
    except Exception:
        return "Error"

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 源目录不存在: {SOURCE_DIR}")
        return

    # 1. 扫描并归类所有 JSON 文件
    print(f"🔍 正在扫描源目录: {SOURCE_DIR}")
    files_by_scene = defaultdict(list)
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".json")]
    
    print("📂 正在解析场景归属...")
    for json_file in tqdm(all_files):
        json_path = os.path.join(SOURCE_DIR, json_file)
        scene = get_scene_from_json(json_path)
        if scene in QUOTAS:
            files_by_scene[scene].append(json_file)

    # 2. 执行抽样与复制
    if os.path.exists(TARGET_ROOT):
        print(f"⚠️ 目标目录已存在，正在清空: {TARGET_ROOT}")
        shutil.rmtree(TARGET_ROOT)
    os.makedirs(TARGET_ROOT)

    print("\n🚀 开始抽样复制...")
    total_copied = 0
    
    for scene, target_count in QUOTAS.items():
        candidates = files_by_scene[scene]
        available_count = len(candidates)
        sample_count = min(target_count, available_count)
        
        print(f"   🔹 [{scene}]: {available_count} 张 -> 抽取 {sample_count} 张")
        
        if sample_count == 0:
            continue
            
        selected_files = random.sample(candidates, sample_count)
        
        scene_dir = os.path.join(TARGET_ROOT, scene)
        os.makedirs(scene_dir, exist_ok=True)
        
        for json_file in selected_files:
            # 1. 复制 JSON
            src_json = os.path.join(SOURCE_DIR, json_file)
            dst_json = os.path.join(scene_dir, json_file)
            shutil.copy2(src_json, dst_json)
            
            # 2. 查找并复制图片 (处理 .jpg 和 .JPG)
            base_name = os.path.splitext(json_file)[0]
            found_img = False
            
            for ext in VALID_EXTS:
                possible_name = base_name + ext
                src_img = os.path.join(SOURCE_DIR, possible_name)
                
                if os.path.exists(src_img):
                    dst_img = os.path.join(scene_dir, possible_name)
                    shutil.copy2(src_img, dst_img)
                    found_img = True
                    break # 找到了就停止尝试其他后缀
            
            if not found_img:
                print(f"      ⚠️ 图片缺失: {base_name}.[jpg/JPG/...]")
            else:
                total_copied += 1

    print("\n" + "="*50)
    print(f"✅ 完成！共复制 {total_copied} 组数据。")
    print(f"📂 结果保存在: {TARGET_ROOT}")

if __name__ == "__main__":
    main()