import os
import json
from collections import Counter
from tqdm import tqdm

# ================= ⚙️ 配置路径 =================
# 目标 JSON 文件夹路径
JSON_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/0110_full_test_benchmark"
# ===============================================

def main():
    if not os.path.exists(JSON_DIR):
        print(f"❌ 错误: 目录不存在 - {JSON_DIR}")
        return

    # 获取所有 .json 文件
    files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
    print(f"📂 正在扫描目录: {JSON_DIR}")
    print(f"📄 共发现 {len(files)} 个 JSON 文件，开始分析...")

    # 用于统计标签出现次数
    label_counts = Counter()
    total_boxes = 0

    # 遍历文件
    for file_name in tqdm(files):
        file_path = os.path.join(JSON_DIR, file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 获取该文件内所有检测对象的 label
                objects = data.get('objects', [])
                for obj in objects:
                    label = obj.get('label', 'unknown')
                    label_counts[label] += 1
                    total_boxes += 1
                    
        except Exception as e:
            print(f"⚠️ 读取文件出错 {file_name}: {e}")

    # === 输出统计结果 ===
    print("\n" + "="*60)
    print(f"{'Label Name (类别名称)':<40} | {'Count (数量)':<10}")
    print("-" * 60)

    # 按照数量从多到少排序输出
    for label, count in label_counts.most_common():
        # 如果标签包含 " _ "，高亮显示，提醒注意格式问题
        if " _ " in label:
            display_label = f"⚠️ {label}" 
        else:
            display_label = label
            
        print(f"{display_label:<40} | {count:<10}")

    print("-" * 60)
    print(f"📌 统计汇总:")
    print(f"   - 包含的类别总数 (Types): {len(label_counts)}")
    print(f"   - 检测出的目标总数 (Total Boxes): {total_boxes}")
    print("="*60)

if __name__ == "__main__":
    main()