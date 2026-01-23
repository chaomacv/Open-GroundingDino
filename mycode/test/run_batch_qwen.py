import os
import subprocess
import sys
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import argparse
# ================= ⚙️ 基础路径配置 =================
# [新增] 解析命令行参数获取 Qwen JSON 路径
parser = argparse.ArgumentParser(description="Batch Runner")
parser.add_argument("--qwen_json", type=str, required=True, help="Qwen3 结果文件的绝对路径")
# 使用 parse_known_args 防止与后续可能的参数冲突
args_pre, _ = parser.parse_known_args()

QWEN_RESULT_JSON = args_pre.qwen_json  # <--- 这里变成了动态变量

print(f"🔗 [Batch Runner] 接收到的 Qwen JSON 路径: {QWEN_RESULT_JSON}")

PROJECT_ROOT = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino"
LOGS_ROOT = os.path.join(PROJECT_ROOT, "logs", "0113")
OUTPUT_ROOT_BASE = os.path.join(PROJECT_ROOT, "batch_eval_results", "qwen")

# 评估脚本路径
EVAL_SCRIPT = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/test/visualize_evaluate_argparse_qwen.py"

# Label Map 文件路径
LABEL_MAP_FULL = os.path.join(PROJECT_ROOT, "label_map.json")
LABEL_MAP_ONLY = os.path.join(PROJECT_ROOT, "label_map_only.json")

# ================= 🎛️ 任务开关 =================

RUN_BENCHMARK  = True 
RUN_TEST_SPLIT = False 

# ================= 💻 显卡配置 (新增) =================
AVAILABLE_GPUS = [0, 1, 2, 3]  # 你的4张卡 ID
GPU_QUEUE = Queue()
for gpu in AVAILABLE_GPUS:
    GPU_QUEUE.put(gpu)

# 线程锁，防止多线程打印日志时乱序
PRINT_LOCK = threading.Lock()

# ================= 📦 1. 待评估的模型列表 =================
MODELS_LIST = [
    # "model1_std_fullneg",
    # "model2_std_posonly",
    "model3_only_fullneg",
    "model4_only_posonly",
]

# ================= 📂 2. 数据集详细配置 =================
BENCHMARK_JSON_PATH = os.path.join(PROJECT_ROOT, "benchmark.json")
TEST_JSON_PATH = os.path.join(PROJECT_ROOT, "test_split_coco.json")

DATASET_CONFIGS = [
    {
        "name": os.path.basename(BENCHMARK_JSON_PATH), 
        "run_flag": RUN_BENCHMARK,
        "json_path": BENCHMARK_JSON_PATH,
        "image_root": "/opt/data/private/xjx/RailMind/database/test/基准测试_1229/基准测试数据集",
        "label_map": LABEL_MAP_ONLY 
    },
    {
        "name": os.path.basename(TEST_JSON_PATH),
        "run_flag": RUN_TEST_SPLIT,
        "json_path": TEST_JSON_PATH,
        "image_root": "/opt/data/private/xjx/RailMind/高速铁路无人机图像/FilteredLabeled",
        "label_map": LABEL_MAP_FULL
    }
]

# =========================================================

def safe_print(message):
    """线程安全的打印"""
    with PRINT_LOCK:
        print(message)

def run_task(task_args):
    """
    运行单个评估任务 (被线程池调用)
    """
    # 解包参数
    model_folder, dataset_cfg, use_gt_labels = task_args
    
    # --- 1. 申请 GPU ---
    gpu_id = GPU_QUEUE.get() # 如果队列空了，这里会阻塞等待
    try:
        # 1. 寻找权重文件
        checkpoint = os.path.join(LOGS_ROOT, model_folder, "checkpoint_best_regular.pth")
        if not os.path.exists(checkpoint):
            checkpoint_alt = os.path.join(LOGS_ROOT, model_folder, "checkpoint.pth")
            if os.path.exists(checkpoint_alt):
                safe_print(f"[GPU {gpu_id}] ⚠️ 提示: {model_folder} 没找到 best_regular，使用 checkpoint.pth 代替")
                checkpoint = checkpoint_alt
            else:
                safe_print(f"[GPU {gpu_id}] ❌ 错误: {model_folder} 下找不到任何权重文件，跳过...")
                return

        # 2. 构造输出路径
        dataset_name = dataset_cfg["name"]
        mode_suffix = "GTLabels" if use_gt_labels else "AllLabels"
        
        task_output_dir = os.path.join(OUTPUT_ROOT_BASE, dataset_name, f"{model_folder}_{mode_suffix}")
        log_file = os.path.join(OUTPUT_ROOT_BASE, dataset_name, f"{model_folder}_{mode_suffix}.log")

        if not os.path.exists(task_output_dir):
            os.makedirs(task_output_dir, exist_ok=True)
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        current_label_map = dataset_cfg["label_map"]

        # 打印启动信息 (简化版，防止刷屏)
        safe_print(f"🚀 [GPU {gpu_id}] 启动: {dataset_name} | {model_folder} | {mode_suffix}")
        
        if not os.path.exists(current_label_map):
            safe_print(f"[GPU {gpu_id}] ❌ 错误: Label Map 不存在")
            return
        if not os.path.exists(dataset_cfg['json_path']):
            safe_print(f"[GPU {gpu_id}] ❌ 错误: JSON 不存在")
            return

        # 3. 构造命令
        cmd = [
            "python", EVAL_SCRIPT,
            "--checkpoint_path", checkpoint,
            "--output_dir", task_output_dir,
            "--label_map_file", current_label_map,
            "--test_json_path", dataset_cfg['json_path'],
            "--image_root", dataset_cfg['image_root']
        ]
        
        if use_gt_labels:
            cmd.append("--use_gt_labels_only")
            
            # [修改核心逻辑]：只有在 Benchmark 数据集且开启 GTLabels (原意为Oracle模式) 时，
            # 注入 Qwen3 的结果作为 Prompt 来源
            if "benchmark" in dataset_cfg["name"].lower():
                cmd.append("--external_prompt_json")
                cmd.append(QWEN_RESULT_JSON)
                safe_print(f"[GPU {gpu_id}] ℹ️ 使用 Qwen3 结果替换 GT Prompt: {dataset_cfg['name']}")

        # 4. 设置环境变量 (核心并行逻辑)
        env = os.environ.copy()
        env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        # 指定该进程只能看到申请到的这张卡
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # 5. 执行命令
        with open(log_file, "w", encoding="utf-8") as f_log:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, 
                text=True,
                bufsize=1,
                env=env
            )

            for line in process.stdout:
                # 并行时，不建议把所有输出都 print 到屏幕，会非常乱
                # 这里只 print 关键进度条或报错，或者干脆只写文件
                # 为了保持清爽，我们只把非进度条信息写入日志文件
                
                # 过滤逻辑
                is_progress_bar = ("%|" in line) and (("it/s" in line) or ("s/it" in line) or ("it]" in line))
                if not is_progress_bar:
                    f_log.write(line)
            
            process.wait()

        if process.returncode == 0:
            safe_print(f"✅ [GPU {gpu_id}] 完成: {dataset_name} | {model_folder}_{mode_suffix}")
        else:
            safe_print(f"❌ [GPU {gpu_id}] 失败: {model_folder}_{mode_suffix} (查看日志)")

    finally:
        # --- 任务结束，归还 GPU ---
        GPU_QUEUE.put(gpu_id)

if __name__ == "__main__":
    if not os.path.exists(EVAL_SCRIPT):
        print(f"⚠️ 找不到 {EVAL_SCRIPT}，请检查路径！")
        sys.exit(1)

    print(f"📝 任务计划 (4卡并行模式):")
    print(f"   - Benchmark: {'✅ 开启' if RUN_BENCHMARK else '⬜ 关闭'}")
    print(f"   - TestSplit: {'✅ 开启' if RUN_TEST_SPLIT else '⬜ 关闭'}")

    # 1. 收集所有任务
    all_tasks = []
    for model in MODELS_LIST:
        for dataset in DATASET_CONFIGS:
            if not dataset["run_flag"]:
                continue
            # 添加任务参数到列表
            # all_tasks.append((model, dataset, False)) # AllLabels
            all_tasks.append((model, dataset, True))  # GTLabels

    print(f"📊 总共生成 {len(all_tasks)} 个任务，准备并行执行...")
    
    # 2. 使用线程池并发执行
    # max_workers=len(AVAILABLE_GPUS) 确保最多只有4个任务同时跑
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=len(AVAILABLE_GPUS)) as executor:
        executor.map(run_task, all_tasks)

    end_time = time.time()
    print(f"\n🎉 所有任务全部完成！耗时: {end_time - start_time:.2f} 秒")