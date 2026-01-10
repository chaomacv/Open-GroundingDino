import os
from collections import Counter
from pathlib import Path

# ================= 配置区域 =================
dataset_root = "/opt/data/private/xjx/RailMind/高速铁路无人机图像/FilteredLabeled"
# ===========================================

def count_files(directory):
    extension_counts = Counter()
    total_files = 0

    print(f"正在扫描目录: {directory} ...")
    
    # 递归遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            total_files += 1
            # 获取文件后缀 (转为小写以避免 .JPG 和 .jpg 分开统计)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext == '':
                file_ext = '无后缀'
            extension_counts[file_ext] += 1

    print("-" * 30)
    print(f"统计结果 (总文件数: {total_files}):")
    print("-" * 30)
    
    # 按数量降序打印
    for ext, count in extension_counts.most_common():
        print(f"{ext:<10} : {count} 个")
    print("-" * 30)

if __name__ == "__main__":
    if os.path.exists(dataset_root):
        count_files(dataset_root)
    else:
        print(f"错误: 路径不存在 - {dataset_root}")