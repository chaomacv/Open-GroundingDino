import os
import subprocess
import sys

# ================= ⚙️ 批量任务配置 =================

PROJECT_ROOT = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino"
LOGS_ROOT = os.path.join(PROJECT_ROOT, "logs")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "batch_eval_results")

# 待评估的模型列表
MODELS_LIST = [
    "0111_railway_4gpu_wandb_full_label",
    "0111_railway_4gpu_wandb_full_label_of_only_benchmark",
    "0111_railway_4gpu_wandb_only_label",
    "0111_railway_4gpu_wandb_only_label_of_only_benchmark",
]

# 通用脚本路径
EVAL_SCRIPT = "visualize_evaluate_argparse.py"

# [新增] Label Map 路径配置
LABEL_MAP_FULL = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/label_map.json"
LABEL_MAP_ONLY = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/label_map_only.json"

# ===================================================

def run_task(model_folder, use_gt_labels):
    checkpoint = os.path.join(LOGS_ROOT, model_folder, "checkpoint_best_regular.pth")
    
    # 构造清晰的输出文件夹名
    mode_suffix = "GTLabels" if use_gt_labels else "AllLabels"
    output_dir_name = f"{model_folder}_benchmark_{mode_suffix}"
    
    output_dir = os.path.join(OUTPUT_ROOT, output_dir_name)
    log_file = os.path.join(OUTPUT_ROOT, f"{output_dir_name}.log")

    # [核心修改] 根据模型名称智能选择 Label Map
    # 如果文件夹名包含 "only_label" (且不是 full_label)，则使用 label_map_only.json
    if "only_label" in model_folder and "full_label" not in model_folder:
        current_label_map = LABEL_MAP_ONLY
        map_type = "Only (Subset)"
    else:
        current_label_map = LABEL_MAP_FULL
        map_type = "Full (Standard)"

    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    print(f"\n{'='*80}")
    print(f"🚀 启动任务: {model_folder}")
    print(f"   - 模式: {mode_suffix} (Use GT Labels Only = {use_gt_labels})")
    print(f"   - Label Map: {map_type}")
    print(f"     -> {current_label_map}")
    print(f"   - 权重: {checkpoint}")
    print(f"   - 输出: {output_dir}")
    print(f"   - 日志: {log_file}")
    print(f"{'='*80}")

    if not os.path.exists(checkpoint):
        print(f"❌ 错误: 找不到权重文件 {checkpoint}，跳过...")
        return
    
    if not os.path.exists(current_label_map):
        print(f"❌ 错误: 找不到 Label Map 文件 {current_label_map}，跳过...")
        return

    # 构造命令
    cmd = [
        "python", EVAL_SCRIPT,
        "--checkpoint_path", checkpoint,
        "--output_dir", output_dir,
        "--label_map_file", current_label_map  # [新增] 传入动态选择的 Label Map
    ]
    
    # 如果是 GT 模式，加上开关参数
    if use_gt_labels:
        cmd.append("--use_gt_labels_only")

    # 执行命令并双向输出
    with open(log_file, "w", encoding="utf-8") as f_log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1 
        )

        for line in process.stdout:
            print(line, end="") 
            f_log.write(line)   
        
        process.wait()

    if process.returncode == 0:
        print(f"✅ 任务 {output_dir_name} 完成！")
    else:
        print(f"❌ 任务 {output_dir_name} 失败，请检查日志。")

if __name__ == "__main__":
    if not os.path.exists(EVAL_SCRIPT):
        print(f"⚠️ 找不到 {EVAL_SCRIPT}，请先创建该文件！")
        sys.exit(1)

    # 双层循环
    for model in MODELS_LIST:
        run_task(model, use_gt_labels=False)
        run_task(model, use_gt_labels=True)