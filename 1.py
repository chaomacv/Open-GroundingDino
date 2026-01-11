import json
import os

# 配置路径
input_path = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/test_benchmark.jsonl"
output_path = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/test_benchmark_coco.json"

def convert_to_coco(jsonl_path, save_path):
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    categories_map = {} # 用于存储 category_name 到 id 的映射
    image_id = 0
    ann_id = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # 1. 填充 images 板块
            coco_output["images"].append({
                "id": image_id,
                "file_name": data["filename"],
                "height": data["height"],
                "width": data["width"]
            })

            # 2. 填充 annotations 板块
            for inst in data["detection"]["instances"]:
                cat_name = inst["category"]
                cat_label = inst["label"]
                
                # 更新 categories 字典
                if cat_name not in categories_map:
                    categories_map[cat_name] = cat_label
                    coco_output["categories"].append({
                        "id": cat_label,
                        "name": cat_name,
                        "supercategory": "railway"
                    })

                # 注意：输入的 bbox 是 [x1, y1, x2, y2]，COCO 需要 [x, y, width, height]
                x1, y1, x2, y2 = inst["bbox"]
                w = x2 - x1
                h = y2 - y1

                coco_output["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_label,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id += 1
            
            image_id += 1

    # 对类别按 ID 排序（可选，为了美观）
    coco_output["categories"] = sorted(coco_output["categories"], key=lambda x: x["id"])

    # 保存结果
    with open(save_path, 'w', encoding='utf-8') as f_out:
        json.dump(coco_output, f_out, indent=4, ensure_ascii=False)
    
    print(f"成功转换！共处理 {image_id} 张图片，{ann_id} 个标注。")
    print(f"保存路径: {save_path}")

if __name__ == "__main__":
    convert_to_coco(input_path, output_path)