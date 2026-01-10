import os
import json
import pandas as pd
from collections import Counter
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 1. 基准测试集所在的根目录 (支持递归查找)
BENCHMARK_DIR = "/opt/data/private/xjx/RailMind/database/test/基准测试_1229/基准测试数据集"

# 2. 标签映射字典 (你的业务代码 -> 英文名称)
LABEL_MAP = {
    "1_1_2_1": "missing_fastener",
    "1_1_2_2": "broken_fastener",
    "1_4_1_1": "rust",
    "1_4_2_2": "rust",
    "1_4_4_1": "mortar_aging",
    "1_5_3_1": "nut_missing",
    "1_5_3_6": "rust",
    "1_5_3_8": "coating_peeling",
    "1_5_4_2": "rust",
    "2_1_5_2": "nest",
    "3_1_2_1": "antenna_nut_loose",
    "3_1_3_1": "nest",
    "4_1_2_1": "plastic_film",
    "4_1_4_1": "rubbish"
}
# ===============================================

def main():
    if not os.path.exists(BENCHMARK_DIR):
        print(f"❌ 错误: 目录不存在 -> {BENCHMARK_DIR}")
        return

    # 1. 递归扫描获取所有 JSON 文件路径
    print(f"🔍 正在递归扫描基准测试集: {BENCHMARK_DIR}")
    json_file_paths = []
    
    for root, dirs, files in os.walk(BENCHMARK_DIR):
        for file in files:
            # 过滤掉以 vis_ 开头的模型生成文件，只保留原始标注
            # 同时也排除掉 label_map.json 等无关文件
            # 假设原始标注文件没有特定前缀，或者是纯数字/字母组合
            if file.endswith(".json") and not file.startswith("vis_") and file != "label_map.json":
                full_path = os.path.join(root, file)
                json_file_paths.append(full_path)
    
    # 排序方便查看进度
    json_file_paths.sort()
    
    if len(json_file_paths) == 0:
        print("⚠️ 目录中没有找到原始标注 JSON 文件！")
        return

    print(f"📊 正在分析 {len(json_file_paths)} 个标注文件...")

    # 初始化计数器
    total_objects_count = 0        # 总共标注了多少个框
    category_object_counts = Counter() # 每个类别有多少个框 (Object Level)
    category_image_counts = Counter()  # 每个类别出现在多少张图里 (Image Level)
    
    # 记录未定义标签
    unknown_labels = Counter()

    # 循环统计
    for path in tqdm(json_file_paths):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ 读取失败 {path}: {e}")
            continue

        # LabelMe 格式通常把框放在 'shapes' 列表里
        shapes = data.get('shapes', [])
        
        # 记录当前图片里出现过的类别 (用于统计 Image Level)
        seen_labels_in_this_image = set()

        for shape in shapes:
            # 获取原始 label (例如 "1_5_3_6")
            raw_label = shape.get('label', '')
            
            # 映射到英文名称
            label_name = LABEL_MAP.get(raw_label, "Unknown")
            
            if label_name == "Unknown":
                # 如果遇到了字典里没有的标签，记录下来
                unknown_labels[raw_label] += 1
                label_name = f"Unknown ({raw_label})" # 方便在表中展示
            
            # 1. 累加目标数
            total_objects_count += 1
            category_object_counts[label_name] += 1
            
            # 记录到集合里
            seen_labels_in_this_image.add(label_name)
        
        # 2. 累加图片数 (每张图只算一次)
        for label in seen_labels_in_this_image:
            category_image_counts[label] += 1

    # --- 生成报告 ---
    print("\n" + "="*60)
    print("📈 基准测试集标注统计报告 (Ground Truth Statistics)")
    print("="*60)
    print(f"📂 扫描文件数: {len(json_file_paths)}")
    print(f"📦 累计标注目标: {total_objects_count}")
    print("-" * 60)

    if not category_object_counts:
        print("⚠️ 没有读取到任何有效标注数据。")
        return

    # 整理成 DataFrame 表格展示
    stats_data = []
    # 合并已知和未知的统计
    all_labels = list(category_object_counts.keys())
    
    for label in all_labels:
        obj_count = category_object_counts[label]
        img_count = category_image_counts[label]
        stats_data.append({
            "Label Name (类别)": label,
            "Object Count (目标总数)": obj_count,
            "Image Count (涉及图片数)": img_count,
            "Avg per Image": round(obj_count / img_count, 2) if img_count > 0 else 0
        })

    # 创建 DataFrame 并按数量降序排列
    df = pd.DataFrame(stats_data)
    df = df.sort_values(by="Object Count (目标总数)", ascending=False).reset_index(drop=True)

    # 打印表格
    print(df.to_string())
    print("-" * 60)

    # 如果有未注册的标签，额外报警
    if unknown_labels:
        print("\n⚠️  [警告] 发现未在字典中定义的标签：")
        for k, v in unknown_labels.items():
            print(f"   - 代码: {k}, 出现次数: {v}")
        print("   请检查 LABEL_MAP 是否需要更新。")

if __name__ == "__main__":
    main()