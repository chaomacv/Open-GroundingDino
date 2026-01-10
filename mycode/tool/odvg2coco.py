import json
import os

# ================= 配置 =================
# 输入文件（你现在的训练数据）
odvg_path = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/train_odvg.jsonl"
# 你的 label_map
label_map_path = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/label_map.json"
# 输出文件（生成的标准 COCO 格式验证集）
output_coco_path = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/val_coco.json"
# =======================================

def main():
    print("🚀 正在将 ODVG 转换为 COCO 格式...")

    # 1. 读取 Label Map
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    # 确保 label_map 是 {ID: Name} 格式 (处理之前的翻转问题)
    # 如果 key 是字符串名字，value 是 ID，需要翻转回来
    first_val = list(label_map.values())[0]
    if isinstance(first_val, int):
        print("检测到 Label Map 为 {Name: ID}，正在翻转为 {ID: Name}...")
        id_to_name = {v: k for k, v in label_map.items()}
        name_to_id = label_map
    else:
        # 假设已经是 {ID_str: Name}
        id_to_name = {int(k): v for k, v in label_map.items()}
        name_to_id = {v: int(k) for k, v in label_map.items()}

    # 2. 构建 COCO 结构
    coco_output = {
        "info": {"description": "Converted from ODVG"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 填充 Categories
    for cat_id, cat_name in id_to_name.items():
        coco_output["categories"].append({
            "id": cat_id,
            "name": cat_name,
            "supercategory": "none"
        })

    # 3. 读取 ODVG 数据并转换
    anno_id_count = 1
    with open(odvg_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for index, line in enumerate(lines):
        data = json.loads(line)
        
        # 构建 Image 信息
        # 注意：这里的 id 必须和之前 odvg.py 修改里的 index 保持一致
        image_info = {
            "id": index, 
            "file_name": data["filename"],
            "width": data["width"],
            "height": data["height"]
        }
        coco_output["images"].append(image_info)

        # 构建 Annotation 信息
        if "detection" in data and "instances" in data["detection"]:
            for inst in data["detection"]["instances"]:
                # bbox 格式：[x1, y1, x2, y2]
                x1, y1, x2, y2 = inst["bbox"]
                w = x2 - x1
                h = y2 - y1
                
                # COCO bbox 格式：[x, y, w, h]
                coco_bbox = [x1, y1, w, h]
                
                # 获取类别 ID
                # ODVG 里的 label 可能是 ID 也可能是名字，这里做个兼容
                raw_label = inst["label"] 
                category_id = raw_label
                
                # 如果是名字，转 ID
                if isinstance(raw_label, str) and not raw_label.isdigit():
                   if raw_label in name_to_id:
                       category_id = name_to_id[raw_label]
                
                anno = {
                    "id": anno_id_count,
                    "image_id": index,
                    "category_id": int(category_id),
                    "bbox": coco_bbox,
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": [] # 暂时为空
                }
                coco_output["annotations"].append(anno)
                anno_id_count += 1

    # 4. 保存
    print(f"✅ 转换完成！包含 {len(coco_output['images'])} 张图片, {len(coco_output['annotations'])} 个标注。")
    print(f"💾 保存至: {output_coco_path}")
    with open(output_coco_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False)

if __name__ == "__main__":
    main()