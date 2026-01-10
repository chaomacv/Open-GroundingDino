import os
import json
import glob
from tqdm import tqdm

# ================= 配置区域 =================
dataset_root = "/opt/data/private/xjx/RailMind/高速铁路无人机图像/FilteredLabeled"
output_jsonl = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/train_odvg.jsonl"
output_label_map = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/label_map.json"
# ===========================================

def get_bbox_from_points(points, shape_type):
    if shape_type == "rectangle":
        (x1, y1), (x2, y2) = points[0], points[1]
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y2, y2)]
    elif shape_type == "polygon":
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None

def find_image_file(directory, filename):
    """
    在目录下寻找文件，自动处理大小写问题 (.jpg vs .JPG)
    """
    # 1. 直接拼接尝试
    exact_path = os.path.join(directory, filename)
    if os.path.exists(exact_path):
        return exact_path
    
    # 2. 如果找不到，尝试忽略大小写匹配
    base_name = os.path.basename(filename)
    name_no_ext, ext = os.path.splitext(base_name)
    
    # 列出目录下所有文件
    try:
        files_in_dir = os.listdir(directory)
    except FileNotFoundError:
        return None

    # 暴力匹配：文件名相同（忽略大小写）且后缀匹配
    for f in files_in_dir:
        if f.lower() == base_name.lower():
            return os.path.join(directory, f)
            
    return None

def main():
    json_files = glob.glob(os.path.join(dataset_root, "**", "*.json"), recursive=True)
    print(f"找到 {len(json_files)} 个 JSON 标注文件。")

    label_to_id = {}
    records = []
    
    success_count = 0
    skip_count = 0

    for json_path in tqdm(json_files):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # LabelMe 的 imagePath 有时包含路径，我们只取文件名
            original_filename = os.path.basename(data['imagePath']) 
            current_dir = os.path.dirname(json_path)
            
            # === 关键修改：寻找真实存在的图片路径 ===
            real_image_path = find_image_file(current_dir, original_filename)
            
            if real_image_path is None:
                # print(f"[警告] 图片缺失，跳过: {original_filename} (在 {current_dir})")
                skip_count += 1
                continue

            # 计算相对于 dataset_root 的路径
            try:
                relative_path = os.path.relpath(real_image_path, dataset_root)
            except ValueError:
                print(f"[错误] 图片不在数据集根目录下: {real_image_path}")
                skip_count += 1
                continue
            
            height = data['imageHeight']
            width = data['imageWidth']
            instances = []
            
            for shape in data['shapes']:
                label_name = shape['label']
                shape_type = shape['shape_type']
                points = shape['points']

                if label_name not in label_to_id:
                    label_to_id[label_name] = len(label_to_id)

                bbox = get_bbox_from_points(points, shape_type)
                if bbox:
                    instances.append({
                        "bbox": bbox,
                        "label": label_to_id[label_name],
                        "category": label_name
                    })

            if instances:
                record = {
                    "filename": relative_path, 
                    "height": height,
                    "width": width,
                    "detection": {"instances": instances}
                }
                records.append(record)
                success_count += 1

        except Exception as e:
            print(f"处理出错 {json_path}: {e}")
            skip_count += 1

    print(f"\n处理结果: 成功 {success_count} 张, 跳过 {skip_count} 张 (图片缺失或路径错误)")
    
    print(f"正在保存数据到 {output_jsonl} ...")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"正在保存 label_map 到 {output_label_map} ...")
    with open(output_label_map, 'w', encoding='utf-8') as f:
        json.dump(label_to_id, f, indent=4, ensure_ascii=False)

    print("完成！请重新运行训练脚本。")

if __name__ == "__main__":
    main()