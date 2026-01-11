import json
import os
from collections import defaultdict, Counter
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
JSONL_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/train_odvg.jsonl"
# ===============================================

def analyze_data():
    if not os.path.exists(JSONL_PATH):
        print(f"❌ 错误: 找不到文件 {JSONL_PATH}")
        return

    # 初始化统计变量
    total_images = 0
    scene_stats = defaultdict(lambda: {"img_count": 0, "labels": Counter()})
    global_label_counts = Counter()

    print(f"📖 正在解析文件: {JSONL_PATH} ...")

    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                data = json.loads(line.strip())
                total_images += 1

                # 1. 提取场景 (获取 filename 的父文件夹名)
                filename = data.get("filename", "")
                scene = os.path.dirname(filename)
                if not scene:
                    scene = "Root (未分类)"
                
                scene_stats[scene]["img_count"] += 1

                # 2. 提取类别信息
                # 结构: detection -> instances -> category
                instances = data.get("detection", {}).get("instances", [])
                for inst in instances:
                    label = inst.get("category", "unknown")
                    # 更新场景内部统计
                    scene_stats[scene]["labels"][label] += 1
                    # 更新全局统计
                    global_label_counts[label] += 1

            except Exception as e:
                print(f"⚠️ 跳过错误行: {e}")

    # ================= 📊 输出报告 =================
    print("\n" + "="*100)
    print(f"{'🚀 ODVG 数据集分析报告':^100}")
    print("="*100)
    print(f"📈 总体规模:")
    print(f"   - 总图片数: {total_images}")
    print(f"   - 场景总数: {len(scene_stats)}")
    print(f"   - 涵盖类别总数: {len(global_label_counts)}")
    print("-" * 100)

    # 按图片数量排序输出场景
    print(f"{'📂 按场景统计 (Scene Stats)':<40} | {'图片数':<8} | {'类别分布 (Top 3)'}")
    print("-" * 100)
    
    sorted_scenes = sorted(scene_stats.items(), key=lambda x: x[1]['img_count'], reverse=True)
    
    for scene, info in sorted_scenes:
        img_num = info['img_count']
        # 获取该场景下数量最多的前3个类别
        top_labels = info['labels'].most_common(3)
        top_labels_str = ", ".join([f"{k}({v})" for k, v in top_labels])
        
        print(f"{scene:<40} | {img_num:<8} | {top_labels_str}")
        
        # 如果你想看该场景下的所有类别，取消下面两行的注释
        # for lb, cnt in info['labels'].items():
        #     print(f"      └─ {lb}: {cnt}")

    print("-" * 100)
    print(f"🏷️ 全局类别汇总 (Global Labels):")
    # 每行打印 3 个类别以节省空间
    all_labels = global_label_counts.most_common()
    for i in range(0, len(all_labels), 3):
        chunk = all_labels[i:i+3]
        line_str = "  ".join([f"{k:<25}: {v:<6}" for k, v in chunk])
        print(line_str)
    
    print("="*100)

if __name__ == "__main__":
    analyze_data()