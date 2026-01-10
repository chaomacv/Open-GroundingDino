import json
import os

# 路径配置
label_map_path = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/label_map.json"

def main():
    print(f"正在读取: {label_map_path}")
    with open(label_map_path, 'r', encoding='utf-8') as f:
        old_map = json.load(f)

    # 检查当前格式
    first_key = list(old_map.keys())[0]
    first_val = list(old_map.values())[0]
    
    # 如果 Key 是字符串名字，Value 是数字，说明需要翻转
    if isinstance(first_key, str) and not first_key.isdigit() and isinstance(first_val, int):
        print("检测到格式为 {Name: ID}，正在翻转为 {ID: Name} ...")
        
        # 翻转字典：把 ID 转成字符串作为 Key，名字作为 Value
        new_map = {str(v): k for k, v in old_map.items()}
        
        # 保存回原文件
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump(new_map, f, indent=4, ensure_ascii=False)
            
        print("✅ 修复完成！现在可以重新运行训练了。")
        print(f"示例: '0' -> '{new_map['0']}'")
        
    # 如果 Key 已经是数字字符串，说明不需要修
    elif isinstance(first_key, str) and first_key.isdigit():
        print("⚠️ 格式似乎已经是 {ID: Name} 了，无需修改。")
        
    else:
        print("⚠️ 无法识别的格式，请检查文件内容。")

if __name__ == "__main__":
    main()