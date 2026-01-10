import os
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import pandas as pd
from collections import defaultdict

# ================= ⚙️ Configuration Area =================
# 1. Ground Truth (GT) Folder
DIR_GT = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_gt_1229_results"

# 2. Prediction (Pred) Folder
DIR_PRED = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_benchmark_1229_results"

# 3. Matching Threshold
IOU_THRESHOLD = 0.5
# ========================================================

def compute_iou(box1, box2):
    """Calculate IoU for two [x1, y1, x2, y2] rectangles"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def load_data_and_meta(json_path, is_gt=False):
    """
    Read JSON, return (detection box list, metadata dictionary)
    """
    if not os.path.exists(json_path):
        return [], {}

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract metadata (for scene identification)
    meta = {
        "original_path": data.get("original_path", ""),
        "file_name": data.get("file_name", "")
    }

    normalized_objs = []
    if 'objects' in data:
        for obj in data['objects']:
            # 1. Get Box
            if 'box_pixel_xyxy' in obj:
                box = obj['box_pixel_xyxy']
            elif 'bbox' in obj: 
                continue 
            else:
                continue
            
            # 2. Get Score
            score = 1.0 if is_gt else obj.get('score', 0.0)
            
            # 3. Get Label
            label = obj.get('label', 'unknown')
            
            normalized_objs.append({
                'label': label,
                'score': float(score),
                'box': box
            })
    
    return normalized_objs, meta

def get_scene_name(original_path):
    """Extract scene name from original path"""
    if not original_path:
        return "Unknown"
    try:
        dir_name = os.path.dirname(original_path)
        scene_name = os.path.basename(dir_name)
        return scene_name
    except:
        return "Unknown"

def compare_single_pair(objs_gt, objs_pred):
    """Compare a single image, return statistics"""
    stats = {
        'matched': 0,
        'missed': 0,
        'extra': 0,
        'iou_accum': 0.0
    }
    
    matched_pred_indices = set()
    
    for gt in objs_gt:
        best_iou = -1
        best_idx = -1
        
        for idx, pred in enumerate(objs_pred):
            if idx in matched_pred_indices:
                continue
            
            if gt['label'] != pred['label']:
                continue
                
            iou = compute_iou(gt['box'], pred['box'])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        
        if best_iou >= IOU_THRESHOLD:
            matched_pred_indices.add(best_idx)
            stats['matched'] += 1
            stats['iou_accum'] += best_iou
        else:
            stats['missed'] += 1
            
    stats['extra'] = len(objs_pred) - len(matched_pred_indices)
    return stats

def match_files(dir_gt, dir_pred):
    pairs = []
    gt_files = [f for f in os.listdir(dir_gt) if f.startswith("gt_") and f.endswith(".json")]
    pred_files = [f for f in os.listdir(dir_pred) if f.startswith("vis_") and f.endswith(".json")]

    gt_map = {f[3:]: f for f in gt_files}
    pred_map = {f[4:]: f for f in pred_files}

    common_cores = set(gt_map.keys()) & set(pred_map.keys())
    
    for core in sorted(list(common_cores)):
        path_gt = os.path.join(dir_gt, gt_map[core])
        path_pred = os.path.join(dir_pred, pred_map[core])
        pairs.append((path_gt, path_pred, core))
        
    return pairs

def main():
    print("🔍 Starting Scene-based Evaluation...")
    print(f"📂 GT Dir: {DIR_GT}")
    print(f"📂 Pred Dir: {DIR_PRED}")

    file_pairs = match_files(DIR_GT, DIR_PRED)
    
    if len(file_pairs) == 0:
        print("❌ No matched files found!")
        return

    print(f"🔗 Successfully matched {len(file_pairs)} pairs. Analyzing...")

    # Initialize scene stats
    scene_stats = defaultdict(lambda: {
        'files': 0, 
        'gt_objects': 0, 
        'pred_objects': 0, 
        'matched': 0, 
        'missed': 0, 
        'extra': 0, 
        'iou_accum': 0.0
    })

    # Global stats
    total_stats = {
        'gt_objects': 0,
        'pred_objects': 0,
        'matched': 0
    }

    for path_gt, path_pred, core_name in tqdm(file_pairs):
        # 1. Load Data
        objs_gt, meta_gt = load_data_and_meta(path_gt, is_gt=True)
        objs_pred, _ = load_data_and_meta(path_pred, is_gt=False)
        
        # 2. Identify Scene
        scene = get_scene_name(meta_gt.get('original_path'))
        
        # 3. Compare
        res = compare_single_pair(objs_gt, objs_pred)
        
        # 4. Update Scene Stats
        s = scene_stats[scene]
        s['files'] += 1
        s['gt_objects'] += len(objs_gt)
        s['pred_objects'] += len(objs_pred)
        s['matched'] += res['matched']
        s['missed'] += res['missed']
        s['extra'] += res['extra']
        s['iou_accum'] += res['iou_accum']

        # 5. Update Global Stats (Direct Accumulation, Safer)
        total_stats['gt_objects'] += len(objs_gt)
        total_stats['pred_objects'] += len(objs_pred)
        total_stats['matched'] += res['matched']

    # --- Generate Report ---
    report_data = []
    
    for scene, stats in scene_stats.items():
        # Metrics
        recall = stats['matched'] / stats['gt_objects'] if stats['gt_objects'] > 0 else 0
        precision = stats['matched'] / stats['pred_objects'] if stats['pred_objects'] > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        avg_iou = stats['iou_accum'] / stats['matched'] if stats['matched'] > 0 else 0
        
        report_data.append({
            "Scene": scene,
            "Images": stats['files'],
            "GT Objects": stats['gt_objects'],
            "Pred Objects": stats['pred_objects'],
            "Recall": recall,
            "Precision": precision,
            "F1-Score": f1,
            "mIoU": avg_iou,
            "Missed": stats['missed'],
            "Extra": stats['extra']
        })

    # Create DataFrame
    df = pd.DataFrame(report_data)
    
    # Format percentages for display
    # (We keep raw numbers for sorting, create display columns)
    df_display = df.copy()
    df_display['Recall'] = df['Recall'].apply(lambda x: f"{x:.2%}")
    df_display['Precision'] = df['Precision'].apply(lambda x: f"{x:.2%}")
    df_display['F1-Score'] = df['F1-Score'].apply(lambda x: f"{x:.4f}")
    df_display['mIoU'] = df['mIoU'].apply(lambda x: f"{x:.4f}")

    df_display = df_display.sort_values(by="Scene")

    print("\n" + "="*100)
    print("📊 Evaluation by Scene")
    print("="*100)
    print(df_display.to_string(index=False))
    print("-" * 100)
    
    # Calculate Global Metrics
    g_rec = total_stats['matched'] / total_stats['gt_objects'] if total_stats['gt_objects'] > 0 else 0
    g_prec = total_stats['matched'] / total_stats['pred_objects'] if total_stats['pred_objects'] > 0 else 0
    g_f1 = 2 * (g_prec * g_rec) / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0
    
    print(f"\n🏆 Global Metrics:")
    print(f"   Recall:    {g_rec:.2%}  ({total_stats['matched']} / {total_stats['gt_objects']})")
    print(f"   Precision: {g_prec:.2%}  ({total_stats['matched']} / {total_stats['pred_objects']})")
    print(f"   F1-Score:  {g_f1:.4f}")

if __name__ == "__main__":
    main()