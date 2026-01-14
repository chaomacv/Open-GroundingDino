import json
import os

# 配置路径
input_file = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/train_odvg.jsonl"
output_file = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/train_odvg_filtered.jsonl"

# 14个目标保留类别 (请务必确认这里和你 label_map.json 里的名字完全一致)
keep_labels = {
    "fastener missing", "fastener crack", "plate rust", "column rust", "mortar aging",
    "nut missing", "coating rust", "coating peeling", "guard rust", "nest",
    "antenna nut loose", "plastic film", "rubbish"
}

count_total_imgs = 0
count_kept_imgs = 0
count_removed_anns = 0

print(f"🚀 开始处理训练集...")

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        count_total_imgs += 1
        data = json.loads(line)
        
        # 1. 执行类别过滤
        instances = data.get("detection", {}).get("instances", [])
        filtered_instances = [inst for inst in instances if inst.get("category") in keep_labels]
        
        count_removed_anns += (len(instances) - len(filtered_instances))
        
        # 2. 只有当过滤后的标注不为空时，才写入新文件
        if len(filtered_instances) > 0:
            data["detection"]["instances"] = filtered_instances
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            count_kept_imgs += 1

print("\n" + "="*50)
print(f"📊 处理结果汇报:")
print(f"   - 原始图片总数: {count_total_imgs}")
print(f"   - 保留(含有效目标)图片数: {count_kept_imgs}")
print(f"   - 剔除(全为背景)图片数: {count_total_imgs - count_kept_imgs}")
print(f"   - 累计剔除无效标注数: {count_removed_anns}")
print(f"💾 最终文件保存在: {output_file}")
print("="*50)