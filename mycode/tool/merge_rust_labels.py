import os
import json
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 目标文件夹: 你的预测结果 (Prediction) 文件夹
TARGET_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_benchmark_1229_results"

# 映射规则: 将左边的旧标签 -> 修改为右边的新标签
# 这里我们将5种具体的锈蚀统一为通用的 "rust"
LABEL_MAPPING = {
    "guard_rust": "rust",
    "coating_rust": "rust",
    "nut_rust": "rust",
    "column_rust": "rust",
    "plate_rust": "rust",
    
    # 为了保险，如果你之前的代码生成了带空格的版本，也可以加上：
    "guard _ rust": "rust",
    "coating _ rust": "rust",
    "nut _ rust": "rust",
    "column _ rust": "rust",
    "plate _ rust": "rust"
}
# ===============================================

def main():
    if not os.path.exists(TARGET_DIR):
        print(f"❌ 错误: 目录不存在 -> {TARGET_DIR}")
        return

    # 递归获取所有 JSON 文件
    json_files = [os.path.join(r, f) for r, _, fs in os.walk(TARGET_DIR) for f in fs if f.endswith(".json")]

    print(f"🔄 准备扫描 {len(json_files)} 个文件，执行标签合并 (Merge to 'rust')...")
    
    modified_files_count = 0
    total_labels_changed = 0
    
    for json_path in tqdm(json_files):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_changed = False
            
            if 'objects' in data:
                for obj in data['objects']:
                    current_label = obj.get('label', '')
                    
                    # 检查当前标签是否在我们的映射列表中
                    # 1. 精确匹配
                    if current_label in LABEL_MAPPING:
                        obj['label'] = LABEL_MAPPING[current_label]
                        file_changed = True
                        total_labels_changed += 1
                    
                    # 2. 容错匹配 (防止有多余空格/下划线导致匹配失败)
                    # 例如把 "guard_rust" 和 "guard rust" 都统一处理
                    else:
                        # 归一化：去掉所有空格和下划线
                        normalized_label = current_label.replace(" ", "").replace("_", "")
                        # 比如 normalized_label 变成了 "guardrust"
                        
                        # 同时也把映射表的 key 做归一化对比
                        for k, v in LABEL_MAPPING.items():
                            if k.replace(" ", "").replace("_", "") == normalized_label:
                                # 只有当它确实是那5个锈蚀之一时才改
                                if v == "rust": 
                                    obj['label'] = "rust"
                                    file_changed = True
                                    total_labels_changed += 1
                                    break

            # 如果有修改，写回文件
            if file_changed:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                modified_files_count += 1
                
        except Exception as e:
            print(f"⚠️ 处理失败 {json_path}: {e}")

    print("\n" + "="*50)
    print("✅ 锈蚀标签合并完成！")
    print(f"📂 扫描文件: {len(json_files)}")
    print(f"📝 修改文件: {modified_files_count}")
    print(f"🏷️  合并标签数: {total_labels_changed} (变为 'rust')")
    print("="*50)
    print("⚠️ 提示: 请确保你的【真值文件 (GT)】中对应的标签也已经改为了 'rust'，")
    print("        否则对比评估时会因为名称不一致导致 Recall=0。")

if __name__ == "__main__":
    main()