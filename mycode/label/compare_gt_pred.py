import json
import os
import numpy as np
import torch
import torchvision.ops.boxes as box_ops
from collections import defaultdict
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================

# 1. 真实标注文件 (Ground Truth)
GT_JSON_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/benchmark.json"

# 2. 生成的预测结果文件夹
PRED_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/0110_full_test_benchmark"

# 3. 评估参数
IOU_THRESHOLD = 0.4

# ===============================================

# 4. 标准类别白名单 (用于统计 GT 数量，预测出的脏标签不在其中)
VALID_LABELS_MAP = {
    "insulator": "insulator",
    "bird_protection": "bird_protection",
    "fixed_pulley": "fixed_pulley",
    "nest": "nest",
    "nut_normal": "nut_normal",
    "nut_rust": "nut_rust",
    "nut_missing": "nut_missing",
    "rust": "rust",
    "guard_rust": "guard_rust",
    "coating_rust": "coating_rust",
    "coating_peeling": "coating_peeling",
    "fastener": "fastener",
    "fastener_missing": "fastener_missing",
    "slab_crack": "slab_crack",
    "fastener_crack": "fastener_crack",
    "rubbish": "rubbish",
    "plastic_film": "plastic_film",
    "column_normal": "column_normal",
    "mortar_normal": "mortar_normal",
    "column_rust": "column_rust",
    "mortar_aging": "mortar_aging",
    "single_nut": "single_nut",
    "plate_rust": "plate_rust",
    "tower_nut_normal": "tower_nut_normal",
    "antenna_nut_normal": "antenna_nut_normal",
    "antenna_nut_loose": "antenna_nut_loose",
    "car": "car",
    "cement_room": "cement_room",
    "asbestos_tile": "asbestos_tile",
    "color_steel_tile": "color_steel_tile",
    "railroad": "railroad",
    "vent": "vent",
    "top": "top",
    "track_area": "track_area",
    "external_structure": "external_structure",
    "noise_barrier": "noise_barrier",
    "coating_blister": "coating_blister"
}

def normalize_label(label):
    """
    [修改] 禁用清洗功能，保持原始预测结果。
    仅进行 str 转换以防止报错，不做任何字符替换或映射。
    """
    if label is None:
        return "unknown"
    return str(label)

# ===============================================

class Evaluator:
    def __init__(self):
        self.stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0})

    def update(self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        if len(pred_boxes) > 0:
            p_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
            p_scores = torch.tensor(pred_scores, dtype=torch.float32)
        else:
            p_boxes = torch.empty((0, 4))
            p_scores = torch.empty((0,))

        if len(gt_boxes) > 0:
            g_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
        else:
            g_boxes = torch.empty((0, 4))

        # 统计 GT 数量 (只统计白名单内的有效GT)
        for label in gt_labels:
            if label in VALID_LABELS_MAP:
                self.stats[label]['gt_count'] += 1

        # 获取本图中所有出现的类别（包括预测出的脏标签）
        all_labels = set(pred_labels) | set(gt_labels)

        for label in all_labels:
            # 如果标签既不在白名单，也不是预测结果（即它是无关的脏数据），跳过
            # 但这里我们要保留预测出的脏标签，所以只要它在 pred_labels 里，就会被处理
            if label not in VALID_LABELS_MAP and label not in pred_labels:
                continue

            p_idx = [i for i, x in enumerate(pred_labels) if x == label]
            g_idx = [i for i, x in enumerate(gt_labels) if x == label]

            curr_p_boxes = p_boxes[p_idx] if len(p_idx) > 0 else torch.empty((0, 4))
            curr_p_scores = p_scores[p_idx] if len(p_idx) > 0 else torch.empty((0,))
            curr_g_boxes = g_boxes[g_idx] if len(g_idx) > 0 else torch.empty((0, 4))

            # 只有预测，没有GT -> FP (比如预测了 "fastenerener")
            if len(curr_g_boxes) == 0:
                self.stats[label]['fp'] += len(curr_p_boxes)
                continue
            
            # 只有GT，没有预测 -> FN (比如 GT是 "fastener"，但没预测出来)
            if len(curr_p_boxes) == 0:
                self.stats[label]['fn'] += len(curr_g_boxes)
                continue

            # 计算 IoU 并匹配
            ious = box_ops.box_iou(curr_p_boxes, curr_g_boxes)
            gt_matched = torch.zeros(len(curr_g_boxes), dtype=torch.bool)
            sorted_indices = torch.argsort(curr_p_scores, descending=True)

            for idx in sorted_indices:
                max_iou, max_gt_idx = torch.max(ious[idx], dim=0)
                if max_iou >= IOU_THRESHOLD and not gt_matched[max_gt_idx]:
                    self.stats[label]['tp'] += 1
                    gt_matched[max_gt_idx] = True
                else:
                    self.stats[label]['fp'] += 1
            
            # 剩余未匹配的 GT 计为 FN
            num_tp = torch.sum(gt_matched).item()
            num_fn = len(curr_g_boxes) - num_tp
            self.stats[label]['fn'] += num_fn

    def print_report(self):
        print("\n" + "="*110)
        print(f"{'📊 原始对比报告 (No Cleaning / Raw Labels)':^110}")
        print("="*110)
        print(f"{'Class Name':<40} | {'Precision':<10} | {'Recall':<10} | {'GT':<6} | {'TP':<6} | {'FP':<6} | {'FN':<6}")
        print("-" * 110)

        total_tp, total_fp, total_fn, total_gt = 0, 0, 0, 0

        # 获取所有统计到的标签并排序
        all_stat_labels = sorted(self.stats.keys())

        for label in all_stat_labels:
            s = self.stats[label]
            tp, fp, fn, gt = s['tp'], s['fp'], s['fn'], s['gt_count']
            
            # 过滤掉完全为空的行
            if gt == 0 and tp == 0 and fp == 0:
                continue

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_gt += gt

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (gt + 1e-6)
            
            # 标记脏标签：如果不在白名单里，前面加个 '*'
            display_name = label if label in VALID_LABELS_MAP else f"* {label}"

            print(f"{display_name:<40} | {precision:.4f}     | {recall:.4f}     | {gt:<6} | {tp:<6} | {fp:<6} | {fn:<6}")

        print("-" * 110)
        # 计算 Micro Average (全局累计)
        all_prec = total_tp / (total_tp + total_fp + 1e-6)
        all_rec = total_tp / (total_gt + 1e-6)
        
        print(f"{'🏆 Overall (Micro Average)':<40} | {all_prec:.4f}     | {all_rec:.4f}     | {total_gt:<6} | {total_tp:<6} | {total_fp:<6} | {total_fn:<6}")
        print("="*110)
        print("注：带 '*' 的类别为预测出的原始脏标签 (不在标准白名单中)")

def coco_box_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def main():
    print(f"📖 正在加载 GT 文件: {GT_JSON_PATH} ...")
    with open(GT_JSON_PATH, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    # GT 标签不做任何清洗，假设 GT 文件本身是干净的
    cat_id_to_name = {cat['id']: str(cat['name']) for cat in gt_data['categories']}
    
    gt_anns_map = defaultdict(list)
    for ann in gt_data['annotations']:
        gt_anns_map[ann['image_id']].append(ann)

    print(f"✅ GT 加载完成，共 {len(gt_data['images'])} 张图片。")
    
    evaluator = Evaluator()

    print("🚀 开始对比评估 (Raw Mode)...")
    for img_info in tqdm(gt_data['images']):
        file_name = img_info['file_name']
        img_id = img_info['id']
        
        base_name_no_ext = os.path.splitext(os.path.basename(file_name))[0]
        pred_json_name = f"vis_{base_name_no_ext}.json"
        pred_json_path = os.path.join(PRED_DIR, pred_json_name)

        gt_boxes = []
        gt_labels = []
        for ann in gt_anns_map.get(img_id, []):
            gt_boxes.append(coco_box_to_xyxy(ann['bbox']))
            gt_labels.append(cat_id_to_name.get(ann['category_id'], "unknown"))

        pred_boxes = []
        pred_scores = []
        pred_labels = []

        if os.path.exists(pred_json_path):
            try:
                with open(pred_json_path, 'r', encoding='utf-8') as f:
                    pred_data = json.load(f)
                
                for obj in pred_data.get('objects', []):
                    pred_boxes.append(obj['box_pixel_xyxy'])
                    pred_scores.append(obj['score'])
                    # [核心] 直接使用原始预测标签，不清洗
                    raw_label = str(obj['label'])
                    pred_labels.append(raw_label)
            except Exception:
                pass

        evaluator.update(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)

    evaluator.print_report()

if __name__ == "__main__":
    main()