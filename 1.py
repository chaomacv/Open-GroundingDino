import json

input_file = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/train_split_filtered.jsonl"
output_file = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/train_split_cleaned.jsonl"

cleaned_count = 0
with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        data = json.loads(line)
        # 只有当 instances 不为空时，才写入新文件
        if len(data.get("detection", {}).get("instances", [])) > 0:
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
        else:
            cleaned_count += 1

print(f"🧹 已删除 {cleaned_count} 张没有任何有效标注的图片。")