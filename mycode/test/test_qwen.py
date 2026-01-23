import os
import sys
import argparse
import subprocess
import time

# ================= 配置路径 =================
BASE_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/test"

# 1. 标签转换脚本路径
CONVERT_SCRIPT = os.path.join(BASE_DIR, "convert_qwen_json_labels.py")

# 2. 批量评估脚本路径 (run_batch_qwen.py)
RUN_BATCH_SCRIPT = os.path.join(BASE_DIR, "run_batch_qwen.py")

def main():
    parser = argparse.ArgumentParser(description="一键运行：标签转换 -> GroundingDino批量评估")
    parser.add_argument("json_path", type=str, help="需要处理的 Qwen 结果 JSON 文件的绝对路径")
    args = parser.parse_args()

    qwen_json_path = args.json_path

    # --- 检查输入文件 ---
    if not os.path.exists(qwen_json_path):
        print(f"❌ 错误: 输入的 JSON 文件不存在: {qwen_json_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"🚀 [Step 1/2] 正在执行标签标准化转换...")
    print(f"📄 目标文件: {qwen_json_path}")
    print("=" * 60)

    # --- Step 1: 执行标签转换 ---
    # 构造命令: python convert_qwen_json_labels.py --json_path <path>
    cmd_convert = ["python", CONVERT_SCRIPT, "--json_path", qwen_json_path]
    
    try:
        # check=True 会在脚本返回非0状态码时抛出异常
        subprocess.run(cmd_convert, check=True)
        print("\n✅ [Step 1] 标签转换完成！\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ [Step 1] 转换脚本执行失败，退出代码: {e.returncode}")
        sys.exit(1)

    print("=" * 60)
    print(f"🚀 [Step 2/2] 正在启动 GroundingDINO 批量评估...")
    print(f"🔗 注入 External Prompt JSON: {qwen_json_path}")
    print("=" * 60)

    # --- Step 2: 执行批量评估 ---
    # 构造命令: python run_batch_qwen.py --qwen_json <path>
    # 注意：我们刚才修改 run_batch_qwen.py 增加了 --qwen_json 参数
    cmd_batch = ["python", RUN_BATCH_SCRIPT, "--qwen_json", qwen_json_path]

    try:
        start_time = time.time()
        # 这里使用 subprocess.call 或 run，将输出直接打印到控制台
        subprocess.run(cmd_batch, check=True)
        end_time = time.time()
        print(f"\n✅ [Step 2] 批量评估全部完成！总耗时: {end_time - start_time:.2f} 秒")
    except subprocess.CalledProcessError as e:
        print(f"❌ [Step 2] 评估脚本执行失败，退出代码: {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()