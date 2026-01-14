import json
import os
import torch
import numpy as np
import torchvision.ops.boxes as box_ops
from collections import defaultdict
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================

# 1. 真实标注文件 (Ground Truth - 标准 COCO 格式)
GT_JSON_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/test_split_coco.json"

# 2. 生成的预测结果文件夹 (包含 vis_xxx.json 的目录)
PRED_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/0110_full_test_benchmark"

# 3. 评估阈值 (IoU > 0.5 算匹配)
IOU_THRESHOLD = 0.5

# ===============================================

def coco_box_to_xyxy(box):
    """将 [x, y, w, h] 转换为 [x1, y1, x2, y2]"""
    x, y, w, h = box
    return [x, y, x + w, y + h]

class RawEvaluator:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        # 数据结构: {scene_name: {class_name: {'tp': [], 'fp': [], 'scores': [], 'num_gt': 0}}}
        self.stats = defaultdict(lambda: defaultdict(lambda: {'tp': [], 'fp': [], 'scores': [], 'num_gt': 0}))

    def update(self, scene, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
        """
        核心逻辑：这里的 label 都是原始字符串，不做任何清洗
        """
        # 获取所有出现过的标签（并集）
        unique_labels = set(pred_labels) | set(gt_labels)
        
        for label in unique_labels:
            # 1. 筛选出当前标签的 预测框 和 GT框
            p_indices = [i for i, x in enumerate(pred_labels) if x == label]
            g_indices = [i for i, x in enumerate(gt_labels) if x == label]
            
            p_boxes_cls = pred_boxes[p_indices] if len(p_indices) > 0 else torch.empty((0, 4))
            p_scores_cls = pred_scores[p_indices] if len(p_indices) > 0 else torch.empty((0,))
            g_boxes_cls = gt_boxes[g_indices] if len(g_indices) > 0 else torch.empty((0, 4))
            
            # 记录该场景下，该原始标签的 GT 数量
            self.stats[scene][label]['num_gt'] += len(g_boxes_cls)
            
            if len(p_boxes_cls) == 0:
                continue

            # 如果全是预测，没有 GT -> 全是 FP (误检)
            if len(g_boxes_cls) == 0:
                self.stats[scene][label]['fp'].extend([1] * len(p_boxes_cls))
                self.stats[scene][label]['tp'].extend([0] * len(p_boxes_cls))
                self.stats[scene][label]['scores'].extend(p_scores_cls.tolist())
                continue

            # 计算 IoU
            ious = box_ops.box_iou(p_boxes_cls, g_boxes_cls)
            gt_detected = torch.zeros(len(g_boxes_cls), dtype=torch.bool)
            
            # 按分数从高到低匹配
            sorted_indices = torch.argsort(p_scores_cls, descending=True)
            
            for idx in sorted_indices:
                max_iou, max_gt_idx = torch.max(ious[idx], dim=0)
                is_tp = False
                # 只有 IoU 达标 且 标签字符串完全一致(上面筛选过了) 且 GT未被占用 才算 TP
                if max_iou >= self.iou_threshold:
                    if not gt_detected[max_gt_idx]:
                        gt_detected[max_gt_idx] = True
                        is_tp = True
                
                self.stats[scene][label]['tp'].append(1 if is_tp else 0)
                self.stats[scene][label]['fp'].append(0 if is_tp else 1)
                self.stats[scene][label]['scores'].append(p_scores_cls[idx].item())

    def calculate_ap(self, tp, fp, n_pos):
        if n_pos == 0: return 0.0
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp / n_pos
        prec = tp / (tp + fp + 1e-6)
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0: p = 0
            else: p = np.max(prec[rec >= t])
            ap += p / 11.0
        return ap

    def print_results(self):
        print("\n" + "="*140)
        print(f"📊 原始标签评估报告 (Raw Label Evaluation) | IoU Threshold = {self.iou_threshold}")
        print("⚠️ 注意：此模式下 'nut_rust' 和 'rusty nut' 会被视为两个完全不同的类别！")
        print("="*140)
        
        # 格式化表头
        header = "{:<40} | {:<10} | {:<10} | {:<10} | {:<8} | {:<8} | {:<8} | {:<8}".format(
            "Scene / Raw Label", "Precision", "Recall", "AP@50", "GT Count", "Pred", "TP", "FP")
        print(header)
        print("-" * 140)

        total_tp, total_fp, total_gt = 0, 0, 0
        
        # 按场景排序输出
        for scene in sorted(self.stats.keys()):
            class_data = self.stats[scene]
            print(f"📂 场景: {scene}")
            scene_tp, scene_fp, scene_gt = 0, 0, 0
            
            # 按标签名排序输出
            for label in sorted(class_data.keys()):
                data = class_data[label]
                tp = np.array(data['tp'])
                fp = np.array(data['fp'])
                num_gt = data['num_gt']
                
                sum_tp = np.sum(tp) if len(tp) > 0 else 0
                sum_fp = np.sum(fp) if len(fp) > 0 else 0
                
                scene_tp += sum_tp
                scene_fp += sum_fp
                scene_gt += num_gt
                
                # 过滤掉没有任何数据的类别，避免刷屏
                if num_gt == 0 and sum_tp == 0 and sum_fp == 0:
                    continue

                precision = sum_tp / (sum_tp + sum_fp + 1e-6)
                recall = sum_tp / (num_gt + 1e-6)
                ap = self.calculate_ap(tp, fp, num_gt)
                
                print("{:<40} | {:.4f}     | {:.4f}     | {:.4f}     | {:<8} | {:<8} | {:<8} | {:<8}".format(
                    f"  ├─ {label}", precision, recall, ap, num_gt, int(sum_tp+sum_fp), int(sum_tp), int(sum_fp)))
            
            # 场景小结
            s_prec = scene_tp / (scene_tp + scene_fp + 1e-6)
            s_rec = scene_tp / (scene_gt + 1e-6)
            print("{:<40} | {:.4f}     | {:.4f}     | -          | {:<8} | {:<8} | {:<8} | {:<8}".format(
                f"  └─ [Scene Total]", s_prec, s_rec, scene_gt, int(scene_tp + scene_fp), int(scene_tp), int(scene_fp)))
            print("-" * 140)
            
            total_tp += scene_tp
            total_fp += scene_fp
            total_gt += scene_gt

        all_prec = total_tp / (total_tp + total_fp + 1e-6)
        all_rec = total_tp / (total_gt + 1e-6)
        
        print("="*140)
        print(f"🏆 总体概览 (Micro Average):")
        print(f"   Precision: {all_prec:.4f}")
        print(f"   Recall:    {all_rec:.4f}")
        print(f"   GT Total:  {total_gt}")
        print(f"   TP Total:  {int(total_tp)} (正确匹配)")
        print(f"   FP Total:  {int(total_fp)} (类别不符或位置不准)")
        print(f"   FN Total:  {int(total_gt - total_tp)} (漏检)")
        print("="*140)

def main():
    if not os.path.exists(PRED_DIR):
        print(f"❌ 预测文件夹不存在: {PRED_DIR}")
        return

    print(f"📖 读取 GT 文件: {GT_JSON_PATH}")
    with open(GT_JSON_PATH, 'r') as f:
        gt_data = json.load(f)
    
    # 1. 建立 GT 索引 (ID -> 原始 Label Name)
    # 不做任何 lower() 或 replace() 操作，保持原汁原味
    cat_id_to_raw_name = {}
    for cat in gt_data.get('categories', []):
        cat_id_to_raw_name[cat['id']] = cat['name']

    # image id -> annotations
    gt_anns_map = defaultdict(list)
    for ann in gt_data.get('annotations', []):
        gt_anns_map[ann['image_id']].append(ann)

    evaluator = RawEvaluator(iou_threshold=IOU_THRESHOLD)
    
    missing_pred_count = 0
    
    print(f"🚀 开始对比评估 {len(gt_data['images'])} 张图片...")
    
    for img_info in tqdm(gt_data['images']):
        file_name = img_info['file_name'] 
        img_id = img_info['id']
        
        # 提取场景名称
        scene_name = os.path.dirname(file_name)
        if not scene_name: scene_name = "Root"

        # 构造预测文件路径
        base_name_no_ext = os.path.splitext(os.path.basename(file_name))[0]
        pred_json_name = f"vis_{base_name_no_ext}.json"
        pred_json_path = os.path.join(PRED_DIR, pred_json_name)

        # 准备 GT 数据
        gt_boxes = []
        gt_labels = []
        for ann in gt_anns_map.get(img_id, []):
            gt_boxes.append(coco_box_to_xyxy(ann['bbox']))
            # 使用原始 GT 标签
            gt_labels.append(cat_id_to_raw_name.get(ann['category_id'], "unknown"))

        # 准备 Pred 数据
        pred_boxes = []
        pred_scores = []
        pred_labels = []

        if os.path.exists(pred_json_path):
            try:
                with open(pred_json_path, 'r', encoding='utf-8') as f:
                    pred_data = json.load(f)
                
                # 兼容 {"objects": []} 格式
                objects = []
                if isinstance(pred_data, dict):
                    objects = pred_data.get("objects", [])
                elif isinstance(pred_data, list): # 兼容你的旧列表格式
                    pass # 需要根据具体结构解析，这里假设是标准格式
                
                for obj in objects:
                    pred_boxes.append(obj['box_pixel_xyxy'])
                    pred_scores.append(obj['score'])
                    # 使用原始 Pred 标签 (不做任何清洗)
                    pred_labels.append(obj['label'])
                    
            except Exception as e:
                pass
        else:
            missing_pred_count += 1

        # 转换为 Tensor 并更新评估器
        if len(pred_boxes) > 0:
            t_pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
            t_pred_scores = torch.tensor(pred_scores, dtype=torch.float32)
        else:
            t_pred_boxes = torch.empty((0, 4))
            t_pred_scores = torch.empty((0,))
            
        if len(gt_boxes) > 0:
            t_gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
        else:
            t_gt_boxes = torch.empty((0, 4))

        evaluator.update(scene_name, t_pred_boxes, t_pred_scores, pred_labels, t_gt_boxes, gt_labels)

    if missing_pred_count > 0:
        print(f"\n⚠️ 警告: 有 {missing_pred_count} 张 GT 图片未找到对应的预测 JSON 文件。")
        
    evaluator.print_results()

if __name__ == "__main__":
    main()