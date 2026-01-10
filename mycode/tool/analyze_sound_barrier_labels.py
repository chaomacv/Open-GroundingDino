import os
import json
from collections import Counter
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 1. 预测结果文件夹路径
RESULT_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_gt_1229_results"

# 2. 筛选关键词 (根据 original_path 判断)
# 只要路径里包含这个词，就认为是该场景
SCENE_KEYWORD = "声屏障" 
# ===============================================

def main():
    if not os.path.exists(RESULT_DIR):
        print(f"❌ 错误: 目录不存在 -> {RESULT_DIR}")
        return

    # 获取所有 JSON 文件
    json_files = [f for f in os.listdir(RESULT_DIR) if f.endswith(".json")]
    print(f"📂 正在扫描 {RESULT_DIR} 下的 {len(json_files)} 个文件...")
    print(f"🔍 正在筛选属于【{SCENE_KEYWORD}】场景的文件...")

    # 统计器
    scene_file_count = 0
    label_counter = Counter()
    total_objects = 0

    # 循环处理
    for file_name in tqdm(json_files):
        path = os.path.join(RESULT_DIR, file_name)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 1. 检查是否属于目标场景
            # 读取 original_path，例如: /.../基准测试数据集/声屏障/1.jpg
            original_path = data.get("original_path", "")
            
            if SCENE_KEYWORD not in original_path:
                continue  # 如果路径里没写“声屏障”，就跳过
            
            scene_file_count += 1
            
            # 2. 统计该文件内的检测结果
            objects = data.get("objects", [])
            for obj in objects:
                label = obj.get("label", "unknown")
                score = obj.get("score", 0.0)
                
                # 记录标签
                label_counter[label] += 1
                total_objects += 1
                
        except Exception as e:
            print(f"⚠️ 读取失败 {file_name}: {e}")

    # --- 输出报告 ---
    print("\n" + "="*50)
    print(f"📊 【{SCENE_KEYWORD}】场景检测详情分析")
    print("="*50)
    print(f"🖼️  覆盖图片数: {scene_file_count}")
    print(f"📦 检出目标总数: {total_objects}")
    print("-" * 50)

    if len(label_counter) == 0:
        print(f"⚠️ 在【{SCENE_KEYWORD}】场景的图片中，没有检测到任何物体！")
        print("   可能原因：")
        print("   1. 阈值太高，都被过滤了。")
        print("   2. Prompt 不对，模型没反应。")
    else:
        print(f"{'Label Name (检测标签)':<30} | {'Count (数量)':<10}")
        print("-" * 50)
        # 按数量降序排列
        for label, count in label_counter.most_common():
            print(f"{label:<30} | {count:<10}")
    
    print("="*50)

if __name__ == "__main__":
    main()