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
IOU_THRESHOLD = 0.5

# ===============================================

# 4. 核心关注白名单 (只评估列表中的 13 类缺陷/目标，忽略其他所有类别)
TARGET_LABELS_SET = {
    "fastener_missing",   # 扣件缺失
    "fastener_crack",     # 扣件断裂
    "plate_rust",         # 单元板锈蚀
    "column_rust",        # 立柱锈蚀
    "mortar_aging",       # 砂浆层老化
    "nut_missing",        # 螺栓缺失
    "coating_rust",       # 涂层锈蚀
    "coating_peeling",    # 涂层脱落
    "guard_rust",         # 桥栏杆锈蚀
    "nest",               # 鸟巢
    "antenna_nut_loose",  # 天线螺栓松动
    "plastic_film",       # 塑料膜
    "rubbish"             # 垃圾
}

# ===============================================

class Evaluator:
    def __init__(self):
        # 仅统计白名单内的类别
        self.stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0})

    def update(self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        """
        核心更新逻辑：先过滤，再评估。
        """
        
        # --- 1. 预处理：将数据转换为 Tensor ---
        if len(pred_boxes) > 0:
            p_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
            p_scores = torch.tensor(pred_scores, dtype=torch.float32)
            p_labels = np.array(pred_labels)
        else:
            p_boxes = torch.empty((0, 4))
            p_scores = torch.empty((0,))
            p_labels = np.array([])

        if len(gt_boxes) > 0:
            g_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
            g_labels = np.array(gt_labels)
        else:
            g_boxes = torch.empty((0, 4))
            g_labels = np.array([])

        # --- 2. 核心过滤：只保留白名单内的 GT 和 Pred ---
        # 过滤 GT
        valid_gt_indices = [i for i, label in enumerate(g_labels) if label in TARGET_LABELS_SET]
        filtered_gt_boxes = g_boxes[valid_gt_indices] if valid_gt_indices else torch.empty((0, 4))
        filtered_gt_labels = g_labels[valid_gt_indices] if valid_gt_indices else np.array([])

        # 过滤 Pred (无关的预测直接丢弃，不算 FP)
        valid_pred_indices = [i for i, label in enumerate(p_labels) if label in TARGET_LABELS_SET]
        filtered_pred_boxes = p_boxes[valid_pred_indices] if valid_pred_indices else torch.empty((0, 4))
        filtered_pred_scores = p_scores[valid_pred_indices] if valid_pred_indices else torch.empty((0,))
        filtered_pred_labels = p_labels[valid_pred_indices] if valid_pred_indices else np.array([])

        # --- 3. 统计 GT 数量 ---
        for label in filtered_gt_labels:
            self.stats[label]['gt_count'] += 1

        # --- 4. 逐类别匹配 ---
        # 此时参与循环的只有白名单内的类别
        unique_labels = set(filtered_gt_labels) | set(filtered_pred_labels)

        for label in unique_labels:
            # 获取该类别在过滤后数据中的索引
            p_idx = [i for i, x in enumerate(filtered_pred_labels) if x == label]
            g_idx = [i for i, x in enumerate(filtered_gt_labels) if x == label]

            curr_p_boxes = filtered_pred_boxes[p_idx] if len(p_idx) > 0 else torch.empty((0, 4))
            curr_p_scores = filtered_pred_scores[p_idx] if len(p_idx) > 0 else torch.empty((0,))
            curr_g_boxes = filtered_gt_boxes[g_idx] if len(g_idx) > 0 else torch.empty((0, 4))

            # Case A: 只有预测，没有GT -> FP
            if len(curr_g_boxes) == 0:
                self.stats[label]['fp'] += len(curr_p_boxes)
                continue
            
            # Case B: 只有GT，没有预测 -> FN
            if len(curr_p_boxes) == 0:
                self.stats[label]['fn'] += len(curr_g_boxes)
                continue

            # Case C: 计算 IoU 并匹配
            ious = box_ops.box_iou(curr_p_boxes, curr_g_boxes)
            gt_matched = torch.zeros(len(curr_g_boxes), dtype=torch.bool)
            
            # 按分数从高到低匹配
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
        print(f"{'📊 专项缺陷评估报告 (Specific Defects Only)':^110}")
        print("="*110)
        print(f"{'Target Class Name':<30} | {'Precision':<10} | {'Recall':<10} | {'GT':<6} | {'TP':<6} | {'FP':<6} | {'FN':<6}")
        print("-" * 110)

        total_tp, total_fp, total_fn, total_gt = 0, 0, 0, 0

        # 按字母顺序输出
        for label in sorted(list(TARGET_LABELS_SET)):
            s = self.stats[label]
            tp, fp, fn, gt = s['tp'], s['fp'], s['fn'], s['gt_count']
            
            # 即使全为0也要显示，因为这是我们关注的目标
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_gt += gt

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (gt + 1e-6)
            
            print(f"{label:<30} | {precision:.4f}     | {recall:.4f}     | {gt:<6} | {tp:<6} | {fp:<6} | {fn:<6}")

        print("-" * 110)
        # 计算 Micro Average (全局累计)
        all_prec = total_tp / (total_tp + total_fp + 1e-6)
        all_rec = total_tp / (total_gt + 1e-6)
        
        print(f"{'🏆 Overall (Target Only)':<30} | {all_prec:.4f}     | {all_rec:.4f}     | {total_gt:<6} | {total_tp:<6} | {total_fp:<6} | {total_fn:<6}")
        print("="*110)

def coco_box_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def main():
    print(f"📖 正在加载 GT 文件: {GT_JSON_PATH} ...")
    with open(GT_JSON_PATH, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    # 建立 ID -> Name 映射
    cat_id_to_name = {cat['id']: str(cat['name']) for cat in gt_data['categories']}
    
    gt_anns_map = defaultdict(list)
    for ann in gt_data['annotations']:
        gt_anns_map[ann['image_id']].append(ann)

    print(f"✅ GT 加载完成，共 {len(gt_data['images'])} 张图片。")
    print(f"🎯 仅评估以下 {len(TARGET_LABELS_SET)} 类目标: {TARGET_LABELS_SET}")
    
    evaluator = Evaluator()

    print("🚀 开始专项评估...")
    for img_info in tqdm(gt_data['images']):
        file_name = img_info['file_name']
        img_id = img_info['id']
        
        base_name_no_ext = os.path.splitext(os.path.basename(file_name))[0]
        pred_json_name = f"vis_{base_name_no_ext}.json"
        pred_json_path = os.path.join(PRED_DIR, pred_json_name)

        # 1. 解析 GT
        gt_boxes = []
        gt_labels = []
        for ann in gt_anns_map.get(img_id, []):
            gt_boxes.append(coco_box_to_xyxy(ann['bbox']))
            gt_labels.append(cat_id_to_name.get(ann['category_id'], "unknown"))

        # 2. 解析 Pred
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
                    # 保持原始标签
                    pred_labels.append(str(obj['label']))
            except Exception:
                pass

        evaluator.update(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)

    evaluator.print_report()

if __name__ == "__main__":
    main()