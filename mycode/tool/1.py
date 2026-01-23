import os
import json
import shutil
import cv2
from tqdm import tqdm

# ================= 配置区域 =================

# 原始数据集根目录
dataset_root = "/opt/data/private/xjx/RailMind/database/test/基准测试_1229/基准测试数据集"

# 结果保存根目录
output_root = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/inference_results"

# 抽取数量限制
LIMIT_PER_CATEGORY = 10

# 目标代码映射表
target_mapping = {
    "1_5_3_8": {"id": "10", "name": "peeling",     "group": "钢桥"}, # 涂层脱落
    "1_1_2_2": {"id": "14", "name": "snappedclip", "group": "轨道"}, # 扣件断裂
    "1_5_4_2": {"id": "8",  "name": "rustyfence",  "group": "钢桥"}, # 桥栏杆锈蚀
    "1_4_4_1": {"id": "20", "name": "agedcement",  "group": "声屏障"}, # 砂浆老化
    "1_5_3_1": {"id": "6",  "name": "nonut",       "group": "钢桥"}, # 螺栓缺失
    "4_1_2_1": {"id": "16", "name": "plastic",     "group": "环境"}, # 塑料膜
    "1_4_1_1": {"id": "22", "name": "rustyplate",  "group": "声屏障"}, # 板材锈蚀
    "4_1_4_1": {"id": "15", "name": "debris",      "group": "环境"}  # 垃圾堆积
}

valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

# ================= 主逻辑 =================

def process_dataset():
    print(f"开始扫描目录: {dataset_root}")
    print(f"目标：每个类别抽取 {LIMIT_PER_CATEGORY} 张，并绘制红色边框")
    
    # 统计计数器
    count_dict = {k: 0 for k in target_mapping.keys()}
    
    # 获取所有文件列表 (os.walk)
    for root, dirs, files in os.walk(dataset_root):
        
        # 优化：如果所有类别都找齐了，可以提前结束
        if all(c >= LIMIT_PER_CATEGORY for c in count_dict.values()):
            print("所有类别均已采集完毕，提前结束扫描。")
            break

        # 筛选 JSON 文件
        json_files = [f for f in files if f.lower().endswith('.json')]
        
        for json_file in json_files:
            json_path = os.path.join(root, json_file)
            
            try:
                # 1. 读取 JSON
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                shapes = data.get('shapes', [])
                labels_in_image = set(s.get('label', '') for s in shapes)
                
                # 2. 筛选命中的类别
                matched_codes = labels_in_image.intersection(target_mapping.keys())
                
                if not matched_codes:
                    continue 

                # 3. 检查是否还需要收集
                codes_to_process = [c for c in matched_codes if count_dict[c] < LIMIT_PER_CATEGORY]
                
                if not codes_to_process:
                    continue 

                # 4. 找到图片文件
                base_name = os.path.splitext(json_file)[0]
                image_name = None
                image_path = None
                
                for ext in valid_extensions:
                    for case_ext in [ext, ext.upper()]:
                        temp_name = base_name + case_ext
                        temp_path = os.path.join(root, temp_name)
                        if os.path.exists(temp_path):
                            image_name = temp_name
                            image_path = temp_path
                            break
                    if image_path: break
                
                if not image_path:
                    continue

                # 读取原始图片
                img = cv2.imread(image_path)
                if img is None:
                    continue

                # 5. 执行保存 (针对每个命中的类别分别处理)
                for code in codes_to_process:
                    info = target_mapping[code]
                    category_folder = f"{info['id']}_{info['name']}"
                    group = info['group']
                    
                    # --- 路径构建 ---
                    save_origin_dir = os.path.join(output_root, "原图", group, category_folder)
                    save_vis_dir = os.path.join(output_root, "可视化", group, category_folder)
                    
                    os.makedirs(save_origin_dir, exist_ok=True)
                    os.makedirs(save_vis_dir, exist_ok=True)
                    
                    # --- A. 保存原图 ---
                    dest_origin_path = os.path.join(save_origin_dir, image_name)
                    if not os.path.exists(dest_origin_path):
                        shutil.copy2(image_path, dest_origin_path)
                    
                    # --- B. 绘制并保存可视化图 ---
                    # 复制一份图片用于画图，避免污染原变量
                    vis_img = img.copy()
                    
                    # 遍历 JSON 中的所有形状，只画出当前类别(code)对应的框
                    for shape in shapes:
                        if shape.get('label') == code:
                            points = shape.get('points', [])
                            if len(points) >= 2:
                                # 转换坐标为整数
                                pt1 = (int(points[0][0]), int(points[0][1]))
                                pt2 = (int(points[1][0]), int(points[1][1]))
                                
                                # 画矩形框 (红色BGR: 0, 0, 255, 线宽: 5)
                                cv2.rectangle(vis_img, pt1, pt2, (0, 0, 255), 5)
                                
                                # (可选) 在框上方写上类别名
                                text_label = f"{info['name']} ({code})"
                                cv2.putText(vis_img, text_label, (pt1[0], pt1[1]-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    dest_vis_path = os.path.join(save_vis_dir, image_name)
                    cv2.imwrite(dest_vis_path, vis_img)
                    
                    # 更新计数
                    count_dict[code] += 1
                    print(f"[{count_dict[code]}/{LIMIT_PER_CATEGORY}] 已归档并可视化: {info['name']} -> {image_name}")

            except Exception as e:
                print(f"[Error] {json_path}: {e}")

    print("\n" + "="*30)
    print("抽取完成！统计如下：")
    for code, count in count_dict.items():
        info = target_mapping[code]
        print(f"Code: {code} ({info['name']}): {count} 张")
    print(f"结果保存在: {output_root}")

if __name__ == "__main__":
    process_dataset()