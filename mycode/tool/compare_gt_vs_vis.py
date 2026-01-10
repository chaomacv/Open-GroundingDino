import os
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import pandas as pd

# ================= ⚙️ 配置区域 =================
# 1. 真实标注 (Ground Truth) 文件夹 (存放 gt_*.json)
DIR_GT = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_gt_results"

# 2. 模型预测 (Prediction) 文件夹 (存放 vis_*.json)
DIR_PRED = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_test_results_oracle_prompt"

# 3. 匹配阈值 (IoU > 0.5 且类别相同 视为匹配成功)
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

def load_detections(json_path, is_gt=False):
    """
    读取 JSON 并标准化格式。
    is_gt: 标记是否为真实标注 (GT没有score，需要默认设为1.0)
    """
    if not os.path.exists(json_path):
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    normalized_objs = []
    if 'objects' not in data:
        return []

    for obj in data['objects']:
        # 1. 获取 Box (统一找 box_pixel_xyxy)
        # 兼容之前脚本生成的 key
        if 'box_pixel_xyxy' in obj:
            box = obj['box_pixel_xyxy']
        elif 'bbox' in obj: # COCO 原始格式有时候是 xywh，要注意
             # 这里假设我们之前的脚本都生成了 pixel_xyxy，如果没有则跳过
             continue
        else:
            continue
            
        # 2. 获取 Score
        # GT 默认为 1.0，预测结果读取真实 score
        if is_gt:
            score = 1.0
        else:
            score = obj.get('score', 0.0)
        
        # 3. 获取 Label
        label = obj.get('label', 'unknown')
        
        normalized_objs.append({
            'label': label,
            'score': float(score),
            'box': box
        })
    
    return normalized_objs

def compare_single_pair(objs_gt, objs_pred):
    """对比一对图片的检测结果"""
    stats = {
        'matched_count': 0,
        'missed_count': 0,  # GT有，Pred没有 (漏检)
        'extra_count': 0,   # GT没有，Pred有 (误检)
        'iou_sum': 0.0
    }
    
    # 简单的贪婪匹配：为每个 GT 找最佳 Pred
    matched_pred_indices = set()
    
    for gt in objs_gt:
        best_iou = -1
        best_idx = -1
        
        for idx, pred in enumerate(objs_pred):
            if idx in matched_pred_indices:
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
            matched_pred_indices.add(best_idx)
            stats['matched_count'] += 1
            stats['iou_sum'] += best_iou
        else:
            stats['missed_count'] += 1
            
    stats['extra_count'] = len(objs_pred) - len(matched_pred_indices)
    return stats

def match_files(dir_gt, dir_pred):
    """
    关键修改：匹配 gt_XXX.json 和 vis_XXX.json
    返回 list of tuples: [(path_gt, path_pred, core_name), ...]
    """
    pairs = []
    
    # 获取所有 gt 文件名
    gt_files = [f for f in os.listdir(dir_gt) if f.startswith("gt_") and f.endswith(".json")]
    # 获取所有 vis 文件名
    pred_files = [f for f in os.listdir(dir_pred) if f.startswith("vis_") and f.endswith(".json")]

    # 建立映射表: core_name -> full_filename
    # 例如: "123.json" -> "gt_123.json"
    gt_map = {f[3:]: f for f in gt_files}   # 去掉 "gt_" (前3个字符)
    pred_map = {f[4:]: f for f in pred_files} # 去掉 "vis_" (前4个字符)

    # 找交集
    common_cores = set(gt_map.keys()) & set(pred_map.keys())
    
    for core in sorted(list(common_cores)):
        path_gt = os.path.join(dir_gt, gt_map[core])
        path_pred = os.path.join(dir_pred, pred_map[core])
        pairs.append((path_gt, path_pred, core))
        
    return pairs

def main():
    print("🔍 开始对比 Ground Truth (GT) vs Prediction (Vis)...")
    print(f"📂 GT 目录: {DIR_GT}")
    print(f"📂 Pred 目录: {DIR_PRED}")

    # 1. 匹配文件对
    file_pairs = match_files(DIR_GT, DIR_PRED)
    
    if len(file_pairs) == 0:
        print("❌ 未找到匹配的文件对！请检查：")
        print("   1. 文件夹路径是否正确")
        print("   2. GT文件是否以 'gt_' 开头")
        print("   3. Vis文件是否以 'vis_' 开头")
        return

    print(f"🔗 成功匹配 {len(file_pairs)} 对文件。开始逐一分析...")

    # 2. 全局统计变量
    total_stats = {
        'gt_objects': 0,
        'pred_objects': 0,
        'matched': 0,
        'missed': 0,
        'extra': 0,
        'iou_accum': 0.0
    }
    
    detailed_diffs = []

    # 3. 循环对比
    for path_gt, path_pred, core_name in tqdm(file_pairs):
        # 加载数据
        objs_gt = load_detections(path_gt, is_gt=True)
        objs_pred = load_detections(path_pred, is_gt=False)
        
        # 对比
        res = compare_single_pair(objs_gt, objs_pred)
        
        # 累加统计
        total_stats['gt_objects'] += len(objs_gt)
        total_stats['pred_objects'] += len(objs_pred)
        total_stats['matched'] += res['matched_count']
        total_stats['missed'] += res['missed_count']
        total_stats['extra'] += res['extra_count']
        total_stats['iou_accum'] += res['iou_sum']
        
        # 记录有差异的文件 (漏检或误检 > 0)
        if res['missed_count'] > 0 or res['extra_count'] > 0:
            detailed_diffs.append({
                'file': core_name,  # 去掉前缀的原始文件名
                'gt_count': len(objs_gt),
                'pred_count': len(objs_pred),
                'missed': res['missed_count'],
                'extra': res['extra_count']
            })

    # 4. 计算指标
    avg_iou = total_stats['iou_accum'] / total_stats['matched'] if total_stats['matched'] > 0 else 0
    
    # 计算召回率 (Recall) = Matched / GT_Total
    recall = total_stats['matched'] / total_stats['gt_objects'] if total_stats['gt_objects'] > 0 else 0
    
    # 计算精确率 (Precision) = Matched / Pred_Total
    precision = total_stats['matched'] / total_stats['pred_objects'] if total_stats['pred_objects'] > 0 else 0

    print("\n" + "="*50)
    print("📊 最终评测报告 (Evaluation Report)")
    print("="*50)
    print(f"✅ 统计图片: {len(file_pairs)} 张")
    print(f"📦 真实目标总数 (GT): {total_stats['gt_objects']}")
    print(f"📦 模型预测总数 (Pred): {total_stats['pred_objects']}")
    print("-" * 30)
    print(f"🤝 正确检测 (TP): {total_stats['matched']}")
    print(f"📉 漏检 (FN): {total_stats['missed']}  <-- 重点关注")
    print(f"📈 误检 (FP): {total_stats['extra']}")
    print("-" * 30)
    print(f"🎯 平均 IoU: {avg_iou:.4f}")
    print(f"🔵 召回率 (Recall): {recall:.2%}")
    print(f"🔴 精确率 (Precision): {precision:.2%}")
    
    if detailed_diffs:
        print("\n⚠️ 差异最大的前 10 个文件:")
        df = pd.DataFrame(detailed_diffs)
        # 按漏检数排序
        df = df.sort_values(by='missed', ascending=False)
        print(df.head(10).to_string(index=False))
    else:
        print("\n🎉 完美匹配！模型预测与真实标注完全一致。")

if __name__ == "__main__":
    main()