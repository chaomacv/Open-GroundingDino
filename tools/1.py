import json
import os
import shutil
from tqdm import tqdm

# ================= 配置区域 =================
# 1. JSON 文件路径
JSON_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/benchmark.json"

# 2. 原始图片根目录
# 注意：如果 json 写的是 "声屏障/xxx.jpg"，脚本会去 SOURCE_IMG_ROOT/声屏障/xxx.jpg 找图
SOURCE_IMG_ROOT = "/opt/data/private/xjx/RailMind/database/test/基准测试_1229/基准测试数据集" 

# 3. 目标输出根目录
TARGET_ROOT = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/pipeline/dataset_root/reference_pool/AnomalyDetection"

# 4. 新生成的 JSON 保存路径
NEW_JSON_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/pipeline/dataset_root/reference_pool/AnomalyDetection/benchmark_organized.json"
# ===========================================

def organize_dataset():
    # 1. 读取 JSON
    print(f"正在读取 JSON: {JSON_PATH}")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 2. 建立 类别ID -> 类别名称(二级目录) 的映射
    # 例如: {19: "rustypole"}
    id_to_cat_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # 3. 建立 图片ID -> 类别ID 的映射
    # 还是取该图片的第一个标注作为分类依据
    img_to_cat_id = {}
    for ann in coco_data['annotations']:
        if ann['image_id'] not in img_to_cat_id:
            img_to_cat_id[ann['image_id']] = ann['category_id']

    new_images_list = []
    
    # 确保目标根目录存在
    if not os.path.exists(TARGET_ROOT):
        os.makedirs(TARGET_ROOT)

    print(f"开始重组目录结构到: {TARGET_ROOT}")

    # 4. 遍历处理
    for img_info in tqdm(coco_data['images']):
        img_id = img_info['id']
        old_relative_path = img_info['file_name']  # 例如: "声屏障/xxx.JPG"
        
        # --- 核心修改逻辑 ---
        
        # A. 提取一级目录 (从文件名中解析 "声屏障")
        # 假设路径分隔符是 '/' (JSON标准) 或 os.sep
        # 使用 replace 统一处理，防止 Windows/Linux 差异
        normalized_path = old_relative_path.replace('\\', '/')
        if '/' in normalized_path:
            parent_dir_name = normalized_path.split('/')[0] # 拿到 "声屏障"
            file_basename = os.path.basename(normalized_path) # 拿到 "xxx.JPG"
        else:
            # 如果文件名里没有目录，只是 "xxx.JPG"
            parent_dir_name = "root" 
            file_basename = old_relative_path

        # B. 获取二级目录 (从标注获取 "rustypole")
        if img_id in img_to_cat_id and img_to_cat_id[img_id] in id_to_cat_name:
            sub_dir_name = id_to_cat_name[img_to_cat_id[img_id]] # 拿到 "rustypole"
        else:
            sub_dir_name = "uncategorized" # 未标注或未知类别

        # --------------------

        # 5. 构建物理路径
        
        # 源文件绝对路径
        src_abs_path = os.path.join(SOURCE_IMG_ROOT, old_relative_path)
        
        # 目标文件夹: Target / 声屏障 / rustypole
        target_dir = os.path.join(TARGET_ROOT, parent_dir_name, sub_dir_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            
        # 目标文件绝对路径
        dst_abs_path = os.path.join(target_dir, file_basename)

        # 6. 执行复制
        try:
            if os.path.exists(src_abs_path):
                shutil.copy2(src_abs_path, dst_abs_path)
                
                # 7. 更新 JSON 中的 file_name
                # 新路径: 声屏障/rustypole/xxx.JPG
                new_relative_path = f"{parent_dir_name}/{sub_dir_name}/{file_basename}"
                img_info['file_name'] = new_relative_path
            else:
                # 仅打印警告，不中断
                print(f"[警告] 源文件未找到: {src_abs_path}")
        except Exception as e:
            print(f"[错误] 复制异常: {e}")

        new_images_list.append(img_info)

    # 8. 保存新 JSON
    coco_data['images'] = new_images_list
    
    print(f"正在保存新的 JSON: {NEW_JSON_PATH}")
    with open(NEW_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)
    
    print("处理完成！目录已重组为: 一级目录(来自原文件名) / 二级目录(来自Category) / 图片")

if __name__ == "__main__":
    organize_dataset()