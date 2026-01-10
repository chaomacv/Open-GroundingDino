import os
import json
import glob
from tqdm import tqdm

# ================= 配置区域 =================
# 您的数据集根目录 (根据您的截图修改)
dataset_root = "/opt/data/private/xjx/RailMind/高速铁路无人机图像/FilteredLabeled"
# 输出文件的保存路径
output_jsonl = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/train_odvg.jsonl"
output_label_map = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/label_map.json"
# ===========================================

def get_bbox_from_points(points, shape_type):
    """
    核心逻辑：将不同形状转换为 [x1, y1, x2, y2]
    """
    if shape_type == "rectangle":
        # LabelMe的矩形通常是 [[x1, y1], [x2, y2]]
        (x1, y1), (x2, y2) = points[0], points[1]
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y2, y2)]
    
    elif shape_type == "polygon":
        # 多边形：找所有点的 min_x, min_y, max_x, max_y
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return [min(xs), min(ys), max(xs), max(ys)]
    
    return None

def main():
    # 1. 扫描所有的 JSON 文件 (递归查找)
    # 使用 glob 查找所有子文件夹下的 .json 文件
    json_files = glob.glob(os.path.join(dataset_root, "**", "*.json"), recursive=True)
    print(f"找到 {len(json_files)} 个 JSON 标注文件。")

    label_to_id = {}  # 用于存储类别映射: {"nut_rust": 0, "coating_rust": 1}
    categories_info = [] # 用于生成 label_map
    
    records = [] # 存储转换后的数据

    # 2. 开始遍历转换
    for json_path in tqdm(json_files):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = data['imagePath']
            # 这里需要注意：Open-GroundingDino 训练时通常需要图片相对于 dataset_root 的路径
            # 或者我们存储绝对路径，或者存储相对 json 文件的路径
            # 为了稳妥，我们这里尝试构建相对于 dataset_root 的路径
            
            # 获取当前 json 所在的文件夹路径
            current_dir = os.path.dirname(json_path)
            # 图片的绝对路径
            image_abs_path = os.path.join(current_dir, filename)
            
            # 获取相对于 dataset_root 的相对路径 (例如: "钢架桥-仅缺陷标注-检测框/6069...Z.JPG")
            # 这样你在配置文件里设置 root 为 dataset_root 即可
            try:
                relative_path = os.path.relpath(image_abs_path, dataset_root)
            except ValueError:
                # 如果路径不在 root 下，就用文件名兜底（需确保 config 配置正确）
                relative_path = filename

            height = data['imageHeight']
            width = data['imageWidth']

            instances = []
            
            for shape in data['shapes']:
                label_name = shape['label']
                shape_type = shape['shape_type']
                points = shape['points']

                # 自动维护类别 ID
                if label_name not in label_to_id:
                    new_id = len(label_to_id)
                    label_to_id[label_name] = new_id
                    categories_info.append(label_name)

                # 获取 BBox
                bbox = get_bbox_from_points(points, shape_type)
                
                if bbox:
                    instances.append({
                        "bbox": bbox,
                        "label": label_to_id[label_name], # 这里存 int ID
                        "category": label_name            # 这里存字符串名称 (ODVG 格式优势)
                    })

            # 如果这张图有有效标注，则加入数据集
            if instances:
                record = {
                    "filename": relative_path, 
                    "height": height,
                    "width": width,
                    "detection": {
                        "instances": instances
                    }
                }
                records.append(record)

        except Exception as e:
            print(f"处理文件 {json_path} 时出错: {e}")

    # 3. 保存 ODVG 格式 (.jsonl)
    print(f"正在保存 {len(records)} 条数据到 {output_jsonl} ...")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # 4. 保存类别映射 (Label Map)
    print(f"正在保存类别映射到 {output_label_map} ...")
    with open(output_label_map, 'w', encoding='utf-8') as f:
        json.dump(label_to_id, f, indent=4, ensure_ascii=False)

    print("转换完成！")
    print(f"共发现类别: {list(label_to_id.keys())}")

if __name__ == "__main__":
    main()