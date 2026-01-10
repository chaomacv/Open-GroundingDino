import os
import json
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 您的预测结果文件夹路径
TARGET_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_benchmark_1229_results"
# ===============================================

def remove_all_spaces(label):
    """
    清洗逻辑: 暴力去除所有空格
    Example: 
      "missing _ fastener" -> "missing_fastener"
      "plastic _ film"     -> "plastic_film"
      "  broken  "         -> "broken"
    """
    if not isinstance(label, str):
        return "unknown"
    
    # 替换所有空格为空字符
    return label.replace(" ", "")

def main():
    if not os.path.exists(TARGET_DIR):
        print(f"❌ 错误: 目录不存在 -> {TARGET_DIR}")
        return

    # 递归获取所有 JSON 文件
    json_files = []
    for root, dirs, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    print(f"🧹 正在处理 {len(json_files)} 个文件，去除 Label 中的所有空格...")
    
    modified_count = 0
    
    for json_path in tqdm(json_files):
        try:
            # 1. 读取
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            is_file_changed = False
            
            if 'objects' in data:
                for obj in data['objects']:
                    original_label = obj.get('label', '')
                    
                    # ⚡️ 执行去空格操作
                    new_label = remove_all_spaces(original_label)
                    
                    if new_label != original_label:
                        obj['label'] = new_label
                        is_file_changed = True

            # 2. 如果有变化，写回文件
            if is_file_changed:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                modified_count += 1
                
        except Exception as e:
            print(f"⚠️ 处理失败 {json_path}: {e}")

    print("\n" + "="*50)
    print("✅ 修复完成！")
    print(f"📂 扫描文件: {len(json_files)}")
    print(f"✏️  修改文件: {modified_count} 个")
    print("   (例如 'missing _ fastener' 已变为 'missing_fastener')")
    print("🚀 现在 Pred 和 GT 应该都是下划线格式了，可以进行对比了！")

if __name__ == "__main__":
    main()