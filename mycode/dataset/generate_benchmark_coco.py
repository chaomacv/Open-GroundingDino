import os
import json
import glob
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================

# 1. 测试集数据的根目录
DATASET_ROOT = "/opt/data/private/xjx/RailMind/database/test/基准测试_1229/基准测试数据集"

# 2. 输出文件路径
OUTPUT_JSONL = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/test_benchmark.jsonl"

# 3. 编码 -> 英文名称 映射表 (根据您提供的测试集定义)
CODE_TO_NAME = {
    # 轨道
    "1_1_2_1": "fastener_missing",
    "1_1_2_2": "fastener_crack",
    # 声屏障
    "1_4_1_1": "plate_rust",
    "1_4_2_2": "column_rust",
    "1_4_4_1": "mortar_aging",
    # 钢架桥
    "1_5_3_1": "nut_missing",
    "1_5_3_6": "coating_rust",
    "1_5_3_8": "coating_peeling",
    "1_5_4_2": "guard_rust",
    # 接触网杆
    "2_1_5_2": "nest",
    # 铁塔
    "3_1_2_1": "antenna_nut_loose",
    "3_1_3_1": "nest",
    # 环境
    "4_1_2_1": "plastic_film",
    "4_1_4_1": "rubbish"
}

# 4. 英文名称 -> 数字ID 映射表 (与训练集保持一致)
NAME_TO_ID = {
    "insulator": 0, "bird_protection": 1, "fixed_pulley": 2, "nest": 3,
    "nut_normal": 4, "nut_rust": 5, "nut_missing": 6, "rust": 7,
    "guard_rust": 8, "coating_rust": 9, "coating_peeling": 10, "fastener": 11,
    "fastener_missing": 12, "slab_crack": 13, "fastener_crack": 14, "rubbish": 15,
    "plastic_film": 16, "column_normal": 17, "mortar_normal": 18, "column_rust": 19,
    "mortar_aging": 20, "single_nut": 21, "plate_rust": 22, "tower_nut_normal": 23,
    "antenna_nut_normal": 24, "antenna_nut_loose": 25, "car": 26, "cement_room": 27,
    "asbestos_tile": 28, "color_steel_tile": 29, "railroad": 30, "vent": 31,
    "top": 32, "track_area": 33, "external_structure": 34, "noise_barrier": 35,
    "coating_blister": 36
}

# ===============================================

def find_image_file(directory, filename):
    """
    在目录下寻找图片文件，解决 .jpg 和 .JPG 大小写不一致的问题
    """
    exact_path = os.path.join(directory, filename)
    if os.path.exists(exact_path):
        return filename # 返回相对路径（其实就是文件名）
    
    base_name = os.path.basename(filename)
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        return None
        
    for f in files:
        if f.lower() == base_name.lower():
            return f
    return None

def main():
    print(f"🚀 开始处理测试集数据...")
    print(f"📂 数据源: {DATASET_ROOT}")
    
    # 递归查找所有 json 文件
    json_files = glob.glob(os.path.join(DATASET_ROOT, "**", "*.json"), recursive=True)
    print(f"🔍 找到 {len(json_files)} 个 JSON 标注文件")

    records = []
    success_count = 0
    skip_count = 0

    for json_path in tqdm(json_files):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 1. 处理图片路径
            current_dir = os.path.dirname(json_path)
            raw_filename = os.path.basename(data['imagePath'])
            
            # 寻找真实存在的图片文件
            real_filename = find_image_file(current_dir, raw_filename)
            if not real_filename:
                # print(f"⚠️ 跳过：找不到图片 {raw_filename} (在 {current_dir})")
                skip_count += 1
                continue
            
            # 生成相对于 DATASET_ROOT 的路径，方便后续通过 root + filename 读取
            abs_image_path = os.path.join(current_dir, real_filename)
            try:
                relative_path = os.path.relpath(abs_image_path, DATASET_ROOT)
            except ValueError:
                # 如果图片不在 DATASET_ROOT 下（不太可能），则跳过
                skip_count += 1
                continue

            height = data['imageHeight']
            width = data['imageWidth']
            instances = []

            # 2. 处理标注框
            for shape in data['shapes']:
                raw_label = shape['label']
                
                # 步骤 A: 编码转换 (例如 1_5_3_6 -> coating_rust)
                category_name = CODE_TO_NAME.get(raw_label)
                
                # 如果不在编码表中，看它是否本身就是英文名
                if category_name is None:
                    if raw_label in NAME_TO_ID:
                        category_name = raw_label
                    else:
                        # 如果既不是编码，也不是已知英文名，跳过该框
                        continue
                
                # 步骤 B: 获取数字 ID
                label_id = NAME_TO_ID.get(category_name)
                if label_id is None:
                    continue

                # 步骤 C: 处理坐标 (转为 x1, y1, x2, y2)
                points = shape['points']
                shape_type = shape.get('shape_type', 'rectangle')
                
                bbox = []
                if shape_type == "rectangle":
                    (x1, y1), (x2, y2) = points[0], points[1]
                    bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y2, y2)]
                elif shape_type == "polygon":
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                else:
                    continue # 不支持的形状

                instances.append({
                    "bbox": bbox,
                    "label": label_id,
                    "category": category_name
                })

            # 3. 只有当图片包含有效标注时才保存
            if instances:
                record = {
                    "filename": relative_path,
                    "height": height,
                    "width": width,
                    "detection": {
                        "instances": instances
                    }
                }
                records.append(record)
                success_count += 1

        except Exception as e:
            print(f"❌ 处理出错 {json_path}: {e}")
            skip_count += 1

    # 4. 写入结果文件
    print(f"\n💾 正在保存到: {OUTPUT_JSONL}")
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("-" * 40)
    print(f"✅ 处理完成！")
    print(f"📊 成功转换: {success_count} 张图片")
    print(f"🚫 跳过/无效: {skip_count} 张图片")
    print(f"💡 文件可直接用于测试，无需额外映射表。")

if __name__ == "__main__":
    main()