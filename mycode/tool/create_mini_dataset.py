import json
import os
import random
from collections import defaultdict

# ================= 配置区域 =================
# 输入文件路径
BENCHMARK_SRC = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/benchmark.json"
TEST_SRC = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/test_split_coco.json"

# 输出文件路径 (生成的mini文件)
BENCHMARK_DST = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/benchmark_mini.json"
TEST_DST = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/test_split_coco_mini.json"

# 每个子文件夹（场景）抽取的数量
SAMPLES_PER_GROUP = 10
# 随机种子，保证每次生成的结果一致
RANDOM_SEED = 42
# ===========================================

def sample_coco_json(input_path, output_path, sample_num):
    print(f"📖 正在读取: {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data.get('categories', [])

    # 1. 根据 file_name 的文件夹名进行分组
    # 例如: "声屏障/001.jpg" -> key="声屏障"
    grouped_images = defaultdict(list)
    for img in images:
        file_name = img['file_name']
        # 获取目录名，如果没有目录则归为 'Root'
        folder_name = os.path.dirname(file_name)
        if not folder_name:
            folder_name = "Root"
        grouped_images[folder_name].append(img)

    print(f"📊 发现 {len(grouped_images)} 个场景分组:")
    
    # 2. 进行抽样
    selected_images = []
    for folder, img_list in grouped_images.items():
        # 如果图片够多就抽样，不够就全选
        count = min(len(img_list), sample_num)
        sampled = random.sample(img_list, count)
        selected_images.extend(sampled)
        print(f"   ├─ [{folder}]: 总数 {len(img_list)} -> 抽取 {len(sampled)}")

    # 3. 构建 image_id 的快速查找集合
    selected_img_ids = set(img['id'] for img in selected_images)

    # 4. 过滤对应的 annotations
    selected_annotations = [
        ann for ann in annotations 
        if ann['image_id'] in selected_img_ids
    ]

    # 5. 构建新的 JSON 数据
    new_data = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": categories # 类别通常保留全部
    }

    # 6. 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 生成完毕: {output_path}")
    print(f"   - 图片数: {len(images)} -> {len(selected_images)}")
    print(f"   - 标注数: {len(annotations)} -> {len(selected_annotations)}")
    print("-" * 60)

def main():
    random.seed(RANDOM_SEED)
    
    # 处理 Benchmark
    sample_coco_json(BENCHMARK_SRC, BENCHMARK_DST, SAMPLES_PER_GROUP)
    
    # 处理 Test Split
    sample_coco_json(TEST_SRC, TEST_DST, SAMPLES_PER_GROUP)

if __name__ == "__main__":
    main()