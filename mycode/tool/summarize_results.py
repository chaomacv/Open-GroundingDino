import os
import re
import pandas as pd
import glob

# ================= ⚙️ 配置区域 =================
# 你的结果根目录
# 根据你之前的截图，所有 json0.2, json0.25 都在 batch_eval_results_0113 下面
RESULTS_ROOT = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/batch_eval_results/0115"

# 输出文件名
OUTPUT_CSV = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/batch_eval_results/0115/all_results_summary_0115.csv"
# ===========================================

def parse_log_file(file_path):
    """从日志文件中提取 Overall 统计信息 (包含 FN)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 匹配日志中的最后一行统计信息
        # 目标格式: 🏆 总体概览 (Overall): Precision: 0.2131 | Recall: 0.5417 | GT: 96 | TP: 52 | FP: 192 | FN: 44
        pattern = r"Overall.*Precision:\s*([\d\.]+).*Recall:\s*([\d\.]+).*GT:\s*(\d+).*TP:\s*(\d+).*FP:\s*(\d+).*FN:\s*(\d+)"
        match = re.search(pattern, content)
        
        if match:
            return {
                "Precision": float(match.group(1)),
                "Recall": float(match.group(2)),
                "GT": int(match.group(3)),
                "TP": int(match.group(4)),
                "FP": int(match.group(5)),
                "FN": int(match.group(6)) # 新增 FN
            }
    except Exception as e:
        print(f"❌ 读取错误 {file_path}: {e}")
    return None

def extract_metadata(folder_name, filename):
    """
    从路径提取元数据
    """
    # 1. 提取阈值 (Threshold)
    # 匹配 .json 后面的数字 (例如 benchmark_mini.json0.2 -> 0.2)
    threshold = 0.0
    thresh_match = re.search(r"json(\d+\.?\d*)", folder_name)
    if thresh_match:
        threshold = float(thresh_match.group(1))
    
    # 2. 提取模式 (Mode) 和 模型名 (Model)
    clean_name = filename.replace(".log", "")
    mode = "Unknown"
    model = "Unknown"
    
    if clean_name.endswith("_AllLabels"):
        mode = "AllLabels"
        model = clean_name.replace("_AllLabels", "")
    elif clean_name.endswith("_GTLabels"):
        mode = "GTLabels"
        model = clean_name.replace("_GTLabels", "")
    else:
        model = clean_name
    
    return threshold, model, mode

def main():
    print(f"🚀 开始全量扫描: {RESULTS_ROOT}")
    
    data_list = []
    
    # 递归查找所有 .log 文件
    log_files = glob.glob(os.path.join(RESULTS_ROOT, "**", "*.log"), recursive=True)
    
    print(f"📊 发现 {len(log_files)} 个日志文件，正在处理...")
    
    for log_path in log_files:
        folder_path = os.path.dirname(log_path)
        folder_name = os.path.basename(folder_path) # e.g., benchmark_mini.json0.2
        file_name = os.path.basename(log_path)      # e.g., model1_...log
        
        # 1. 解析文件名信息
        threshold, model, mode = extract_metadata(folder_name, file_name)
        
        # 2. 解析文件内容
        stats = parse_log_file(log_path)
        
        if stats:
            # 整合所有信息
            entry = {
                "Threshold": threshold,
                "Model": model,
                "Mode": mode,
                **stats # 展开 Precision, Recall, GT, TP, FP, FN
            }
            
            # 额外计算 F1 (方便分析)
            p = stats["Precision"]
            r = stats["Recall"]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            entry["F1-Score"] = round(f1, 4)
            
            data_list.append(entry)

    # 3. 生成表格
    df = pd.DataFrame(data_list)
    
    if df.empty:
        print("❌ 未提取到数据，请检查 RESULTS_ROOT 路径是否正确。")
        return

    # 4. 排序：阈值 -> 模型 -> 模式
    df = df.sort_values(by=["Threshold", "Model", "Mode"])
    
    # 5. 设置列顺序 (符合直觉的阅读顺序)
    cols = ["Threshold", "Model", "Mode", "F1-Score", "Precision", "Recall", "GT", "TP", "FP", "FN"]
    df = df[cols]

    # 6. 保存
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print("✅ 全量汇总完成！")
    print(f"💾 表格已保存为: {os.path.abspath(OUTPUT_CSV)}")
    print(f"📊 共提取了 {len(df)} 行数据")
    print("="*80)
    
    # 打印前几行预览
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()