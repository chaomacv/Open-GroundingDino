import os
import json
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 1. 真实标注 (GT) 文件夹
DIR_GT = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_gt_1229_results"

# 2. 模型预测 (Pred) 文件夹
DIR_PRED = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_benchmark_1229_results"
# ===============================================

def get_labels_from_folder(folder_path, name):
    print(f"🔍 正在扫描 {name}: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"❌ 错误: 目录不存在 -> {folder_path}")
        return set()

    unique_labels = set()
    file_count = 0

    # 递归扫描
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_count += 1
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if 'objects' in data:
                        for obj in data['objects']:
                            label = obj.get('label', 'unknown')
                            unique_labels.add(label)
                except Exception as e:
                    pass
    
    print(f"   📄 扫描文件: {file_count}")
    print(f"   🏷️  发现类别: {len(unique_labels)} 种")
    return unique_labels

def main():
    # 1. 获取两边的标签集合
    gt_labels = get_labels_from_folder(DIR_GT, "真值 (GT)")
    pred_labels = get_labels_from_folder(DIR_PRED, "预测 (Pred)")

    print("\n" + "="*60)
    print("📊 标签一致性对比报告")
    print("="*60)

    # 2. 打印 GT 标签列表
    print(f"✅ GT 包含的标签 ({len(gt_labels)}):")
    for l in sorted(list(gt_labels)):
        print(f"   - '{l}'")  # 使用单引号包围，方便看清有没有首尾空格

    print("-" * 60)

    # 3. 打印 Pred 标签列表
    print(f"⚡ Pred 包含的标签 ({len(pred_labels)}):")
    for l in sorted(list(pred_labels)):
        print(f"   - '{l}'")

    print("-" * 60)

    # 4. 找不同
    only_in_gt = gt_labels - pred_labels
    only_in_pred = pred_labels - gt_labels
    intersection = gt_labels & pred_labels

    if not only_in_gt and not only_in_pred:
        print("🎉 完美！两边的标签种类完全一致。")
        print(f"   共同标签数量: {len(intersection)}")
    else:
        print("⚠️ 发现不一致！请检查以下差异：")
        
        if only_in_gt:
            print(f"\n🔴 只在 GT 中存在 (Pred 没预测到，或者拼写不同):")
            for l in sorted(list(only_in_gt)):
                print(f"   '{l}'")
        
        if only_in_pred:
            print(f"\n🔵 只在 Pred 中存在 (可能是误检，或者拼写不同):")
            for l in sorted(list(only_in_pred)):
                print(f"   '{l}'")

    print("="*60)

if __name__ == "__main__":
    main()