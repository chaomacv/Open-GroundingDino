import os
import json
import pandas as pd
from collections import Counter
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 这里填你要分析的文件夹路径 (支持递归查找子文件夹)
RESULT_DIR = "/opt/data/private/xjx/RailMind/database/test/基准测试_1229/基准测试数据集" 
# (注意：如果你的结果其实在另一个文件夹，请修改这里。上面你给的代码里填的是数据集路径，通常结果会在 mycode/vis_... 那个目录)
# ===============================================

def main():
    if not os.path.exists(RESULT_DIR):
        print(f"❌ 错误: 目录不存在 -> {RESULT_DIR}")
        return

    # 1. 递归扫描获取所有 JSON 文件路径
    print(f"🔍 正在递归扫描文件夹: {RESULT_DIR}")
    json_file_paths = []
    
    for root, dirs, files in os.walk(RESULT_DIR):
        for file in files:
            if file.endswith(".json"):
                # 获取绝对路径
                full_path = os.path.join(root, file)
                json_file_paths.append(full_path)
    
    json_file_paths.sort()
    
    if len(json_file_paths) == 0:
        print("⚠️ 目录及其子目录中没有找到 .json 文件！")
        return

    print(f"📊 正在分析 {len(json_file_paths)} 个结果文件...")

    # 初始化计数器
    total_objects_count = 0        # 总共检出了多少个框
    category_object_counts = Counter() # 每个类别有多少个框 (Object Level)
    category_image_counts = Counter()  # 每个类别出现在多少张图里 (Image Level)
    empty_images_count = 0         # 没有检测到任何物体的图片数

    # 循环统计
    for path in tqdm(json_file_paths):
        # path 已经是完整的绝对路径了，直接 open 即可
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ 读取失败 {path}: {e}")
            continue

        objects = data.get('objects', [])
        
        if not objects:
            empty_images_count += 1
            continue

        total_objects_count += len(objects)
        
        # 记录当前图片里出现过的类别 (用于统计 Image Level)
        seen_labels_in_this_image = set()

        for obj in objects:
            label = obj.get('label', 'unknown')
            
            # 1. 累加目标数
            category_object_counts[label] += 1
            
            # 记录到集合里
            seen_labels_in_this_image.add(label)
        
        # 2. 累加图片数 (每张图只算一次)
        for label in seen_labels_in_this_image:
            category_image_counts[label] += 1

    # --- 生成报告 ---
    print("\n" + "="*50)
    print("📈 检测结果统计报告 (Detection Statistics)")
    print("="*50)
    print(f"📂 扫描文件数: {len(json_file_paths)}")
    print(f"📦 累计检出目标: {total_objects_count}")
    print(f"⚪ 空图片数量: {empty_images_count} (未检出任何目标)")
    print("-" * 50)

    if not category_object_counts:
        print("⚠️ 没有检测到任何有效类别。")
        return

    # 整理成 DataFrame 表格展示
    stats_data = []
    for label, obj_count in category_object_counts.items():
        img_count = category_image_counts[label]
        stats_data.append({
            "Category Name (类别)": label,
            "Object Count (目标总数)": obj_count,
            "Image Count (涉及图片数)": img_count,
            "Avg per Image (平均每图个数)": round(obj_count / img_count, 2) if img_count > 0 else 0
        })

    # 创建 DataFrame 并按数量降序排列
    df = pd.DataFrame(stats_data)
    df = df.sort_values(by="Object Count (目标总数)", ascending=False).reset_index(drop=True)

    # 打印表格
    print(df.to_string())
    print("-" * 50)
    
    # 额外提示
    print("💡 提示: 如果你看到了 'unknown' 或不属于你预期的类别，")
    print("       请检查 label_map.json 或 prompt 构造逻辑。")

if __name__ == "__main__":
    main()