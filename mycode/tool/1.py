import json
import os

# ================= ⚙️ 配置区域 =================
# 输入文件（你刚才转换出的那个列表格式 JSON）
INPUT_FILE = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/test_benchmark.json"
# 输出文件（标准 COCO 格式）
OUTPUT_FILE = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/test_split_coco_fixed.json"
# ===============================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_list = json.load(f)

    if not isinstance(raw_list, list):
        print("⚠️ 输入文件似乎已经是字典格式，无需转换。")
        return

    print(f"🚀 开始转换 {len(raw_list)} 条图片数据...")

    # 初始化标准 COCO 结构
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 用于去重记录 categories
    category_set = {} 
    ann_id_counter = 1

    for img_idx, item in enumerate(raw_list):
        # 1. 处理 images 信息
        # 兼容 filename 和 file_name 两种写法
        file_path = item.get("filename") or item.get("file_name")
        
        img_info = {
            "id": img_idx,
            "file_name": file_path,
            "height": item.get("height"),
            "width": item.get("width")
        }
        coco_output["images"].append(img_info)

        # 2. 处理 annotations (instances)
        # 路径: item -> detection -> instances
        instances = item.get("detection", {}).get("instances", [])
        
        for inst in instances:
            cat_name = inst.get("category")
            cat_id = inst.get("label")

            # 收集 categories
            if cat_id not in category_set:
                category_set[cat_id] = cat_name

            # 构造 annotation
            ann = {
                "id": ann_id_counter,
                "image_id": img_idx,
                "category_id": cat_id,
                "bbox": inst.get("bbox"), # 保持原始坐标
                "area": 0, # 可选
                "iscrowd": 0
            }
            coco_output["annotations"].append(ann)
            ann_id_counter += 1

    # 3. 构造 categories 列表
    for cid, cname in category_set.items():
        coco_output["categories"].append({
            "id": cid,
            "name": cname,
            "supercategory": "railway"
        })

    # 写入标准 JSON
    print(f"💾 正在保存至: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, indent=4, ensure_ascii=False)

    print(f"✅ 转换完成！")
    print(f"   - 图片数量: {len(coco_output['images'])}")
    print(f"   - 标注数量: {len(coco_output['annotations'])}")
    print(f"   - 类别数量: {len(coco_output['categories'])}")

if __name__ == "__main__":
    main()