import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================

# 1. GT 可视化图文件夹 (之前生成的 vis_gt_benchmark)
GT_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_gt_benchmark"

# 2. 预测可视化图文件夹 (0110_full_test_benchmark)
PRED_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/0110_full_test_benchmark"

# 3. 结果保存位置 (会自动创建)
OUTPUT_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/comparison_results_0110"

# 4. 每个类别抽取的数量
SAMPLES_PER_CLASS = 3

# ===============================================

def main():
    if not os.path.exists(GT_DIR) or not os.path.exists(PRED_DIR):
        print("❌ 错误：输入文件夹不存在，请检查路径。")
        return

    # 清理并创建输出目录
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    print(f"📁 创建结果目录: {OUTPUT_DIR}")

    # 1. 扫描 GT 文件夹并按类别分组
    # GT 文件名格式预期: vis_gt_{类别}_{原始文件名}
    # 例如: vis_gt_声屏障_test.jpg
    
    cat_files = defaultdict(list)
    gt_files = [f for f in os.listdir(GT_DIR) if f.startswith("vis_gt_") and f.endswith((".jpg", ".JPG", ".png"))]

    print(f"🔍 正在扫描 GT 文件... (共找到 {len(gt_files)} 张)")

    for gt_filename in tqdm(gt_files):
        # 解析文件名
        # 去掉前缀 "vis_gt_"
        clean_name = gt_filename[len("vis_gt_"):]
        
        # 分割类别和文件名
        # 假设类别是第一个下划线前的部分 (因为我们之前是用 replace("/", "_") 生成的)
        # 例如: "声屏障_image_01.jpg" -> cat="声屏障", rest="image_01.jpg"
        if "_" in clean_name:
            category, real_basename = clean_name.split("_", 1)
        else:
            # 根目录图片可能没有类别前缀，归为 Root
            category = "Root"
            real_basename = clean_name

        # 构造对应的 Pred 文件名
        # 用户规则: vis_gt_轨道_000201.jpg -> vis_000201.jpg
        # 也就是说 Pred 文件名是 "vis_" + 原始文件名的 basename
        pred_filename = "vis_" + real_basename
        
        pred_path = os.path.join(PRED_DIR, pred_filename)
        gt_path = os.path.join(GT_DIR, gt_filename)

        # 检查 Pred 文件是否存在
        if os.path.exists(pred_path):
            cat_files[category].append({
                "base": real_basename,
                "gt_path": gt_path,
                "pred_path": pred_path
            })

    # 2. 抽样并复制
    print("\n🚀 开始抽样并复制...")
    
    total_copied = 0
    for category, items in cat_files.items():
        # 随机打乱 (或者去掉这一行以保持默认排序)
        random.shuffle(items)
        
        # 选取前 N 个
        selected = items[:SAMPLES_PER_CLASS]
        
        if len(selected) == 0:
            continue

        # 为该类别创建子文件夹
        cat_out_dir = os.path.join(OUTPUT_DIR, category)
        os.makedirs(cat_out_dir, exist_ok=True)
        
        print(f"   - 类别 [{category:<10}]: 抽取 {len(selected)} 对")

        for idx, item in enumerate(selected):
            # 为了方便查看，重命名文件
            # 格式: {序号}_GT_{文件名} 和 {序号}_Pred_{文件名}
            # 这样在文件夹里它们会并排显示
            new_gt_name = f"{idx+1:02d}_GT_{item['base']}"
            new_pred_name = f"{idx+1:02d}_Pred_{item['base']}"
            
            shutil.copy2(item['gt_path'], os.path.join(cat_out_dir, new_gt_name))
            shutil.copy2(item['pred_path'], os.path.join(cat_out_dir, new_pred_name))
            total_copied += 1

    print("\n" + "="*50)
    print(f"✅ 完成！共复制了 {total_copied} 对对比图像。")
    print(f"📂 结果保存在: {os.path.abspath(OUTPUT_DIR)}")
    print("💡 提示: 进入文件夹后，建议按名称排序，这样 GT 和 Pred 会成对出现。")

if __name__ == "__main__":
    main()