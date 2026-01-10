import os
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import pandas as pd

# ================= ⚙️ 配置区域 =================
# 1. 原始 GroundingDINO 生成的 JSON 文件夹 (基准)
DIR_ORIGINAL = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_test_results_2"

# 2. 新版 GroundedDINO-VL 生成的 JSON 文件夹 (待测)
DIR_VL = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_test_results_new/labels"

# 3. 匹配阈值 (IoU > 0.5 且类别相同 视为同一个目标)
IOU_THRESHOLD = 0.5
# ===============================================

def compute_iou(box1, box2):
    """计算两个 [x1, y1, x2, y2] 矩形的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def load_detections(json_path):
    """
    读取 JSON 并标准化格式。
    兼容两个版本的 JSON key 命名差异。
    返回列表: [{'label': str, 'score': float, 'box': [x1,y1,x2,y2]}, ...]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    normalized_objs = []
    if 'objects' not in data:
        return []

    for obj in data['objects']:
        # 兼容性处理：不同脚本可能使用不同的键名
        # 1. 获取 Box (像素坐标)
        if 'box_pixel_xyxy' in obj:
            box = obj['box_pixel_xyxy']
        elif 'bbox_xyxy' in obj:
            box = obj['bbox_xyxy']
        else:
            continue # 找不到坐标跳过
            
        # 2. 获取 Score
        score = obj.get('score', 0.0)
        
        # 3. 获取 Label
        label = obj.get('label', 'unknown')
        
        normalized_objs.append({
            'label': label,
            'score': float(score),
            'box': box
        })
    
    return normalized_objs

def compare_single_image(objs_gt, objs_pred):
    """对比单张图片的检测结果"""
    stats = {
        'iou_sum': 0.0,
        'score_diff_sum': 0.0,
        'matched_count': 0,
        'missed_count': 0,  # 原版有，VL版没有
        'extra_count': 0    # VL版有，原版没有
    }
    
    # 简单的贪婪匹配：为每个 GT 找最佳 Pred
    matched_indices = set()
    
    for gt in objs_gt:
        best_iou = -1
        best_idx = -1
        
        for idx, pred in enumerate(objs_pred):
            if idx in matched_indices:
                continue
            
            # 只有类别相同才进行 IoU 匹配
            if gt['label'] != pred['label']:
                continue
                
            iou = compute_iou(gt['box'], pred['box'])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        
        # 判定是否匹配成功
        if best_iou >= IOU_THRESHOLD:
            matched_indices.add(best_idx)
            stats['matched_count'] += 1
            stats['iou_sum'] += best_iou
            stats['score_diff_sum'] += abs(gt['score'] - objs_pred[best_idx]['score'])
        else:
            stats['missed_count'] += 1
            
    stats['extra_count'] = len(objs_pred) - len(matched_indices)
    return stats

def main():
    print("🔍 开始对比 GroundingDINO (基准) vs GroundedDINO-VL (新版)...")
    
    # 获取文件列表
    files_orig = sorted([os.path.basename(x) for x in glob(os.path.join(DIR_ORIGINAL, "*.json"))])
    files_vl = sorted([os.path.basename(x) for x in glob(os.path.join(DIR_VL, "*.json"))])
    
    # 找交集
    common_files = set(files_orig) & set(files_vl)
    print(f"📂 原始文件数: {len(files_orig)}")
    print(f"📂 新版文件数: {len(files_vl)}")
    print(f"🔗 共同文件数: {len(common_files)}")
    
    if len(common_files) == 0:
        print("❌ 没有找到同名文件，请检查文件夹路径和文件名格式！")
        return

    # 全局统计
    total_stats = {
        'files_processed': 0,
        'total_objects_orig': 0,
        'total_objects_vl': 0,
        'matched': 0,
        'missed': 0,
        'extra': 0,
        'iou_accum': 0.0,
        'score_diff_accum': 0.0
    }
    
    detailed_diffs = []

    for filename in tqdm(common_files):
        path_orig = os.path.join(DIR_ORIGINAL, filename)
        path_vl = os.path.join(DIR_VL, filename)
        
        objs_orig = load_detections(path_orig)
        objs_vl = load_detections(path_vl)
        
        res = compare_single_image(objs_orig, objs_vl)
        
        # 更新全局统计
        total_stats['files_processed'] += 1
        total_stats['total_objects_orig'] += len(objs_orig)
        total_stats['total_objects_vl'] += len(objs_vl)
        total_stats['matched'] += res['matched_count']
        total_stats['missed'] += res['missed_count']
        total_stats['extra'] += res['extra_count']
        total_stats['iou_accum'] += res['iou_sum']
        total_stats['score_diff_accum'] += res['score_diff_sum']
        
        # 记录显著差异 (用于后续分析)
        if res['missed_count'] > 0 or res['extra_count'] > 0:
            detailed_diffs.append({
                'file': filename,
                'orig_count': len(objs_orig),
                'vl_count': len(objs_vl),
                'matched': res['matched_count']
            })

    # --- 计算最终指标 ---
    avg_iou = total_stats['iou_accum'] / total_stats['matched'] if total_stats['matched'] > 0 else 0
    avg_score_diff = total_stats['score_diff_accum'] / total_stats['matched'] if total_stats['matched'] > 0 else 0
    
    print("\n" + "="*40)
    print("📊 对比结果汇总报告")
    print("="*40)
    print(f"✅ 处理图片数量: {total_stats['files_processed']}")
    print(f"📦 原始检测框总数: {total_stats['total_objects_orig']}")
    print(f"📦 新版检测框总数: {total_stats['total_objects_vl']}")
    print("-" * 20)
    print(f"🤝 成功匹配数 (Matched): {total_stats['matched']}")
    print(f"📉 原始有但新版丢失 (Missed): {total_stats['missed']}")
    print(f"📈 新版有但原始没有 (Extra): {total_stats['extra']}")
    print("-" * 20)
    print(f"🎯 平均 IoU (位置一致性): {avg_iou:.4f} (越接近1.0越好)")
    print(f"🔢 平均置信度差异 (Score Diff): {avg_score_diff:.4f} (越小越好)")
    
    if detailed_diffs:
        print("\n⚠️ 发现差异较大的文件 (前10个):")
        df = pd.DataFrame(detailed_diffs)
        print(df.head(10).to_string(index=False))
    else:
        print("\n🎉 完美！所有文件检测数量完全一致。")

if __name__ == "__main__":
    main()