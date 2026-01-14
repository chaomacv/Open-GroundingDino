import argparse
import os
import torch
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import cv2
from collections import defaultdict
import torchvision.ops.boxes as box_ops

# 引入 GroundingDINO 模块
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict, annotate

def get_args():
    parser = argparse.ArgumentParser(description="GroundingDINO Batch Evaluation")
    # 动态参数
    parser.add_argument("--checkpoint_path", required=True, help="模型权重路径")
    parser.add_argument("--output_dir", required=True, help="结果输出目录")
    
    # [新增] 模式开关：加上这个参数即为 True，不加即为 False
    parser.add_argument("--use_gt_labels_only", action="store_true", help="是否仅使用 GT 标签 (Oracle Mode)")
    
    # 固定参数
    parser.add_argument("--config_path", default="/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/config/cfg_odvg.py")
    parser.add_argument("--test_json_path", default="/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/benchmark.json")
    parser.add_argument("--image_root", default="/opt/data/private/xjx/RailMind/database/test/基准测试_1229/基准测试数据集")
    parser.add_argument("--label_map_file", default="/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/label_map.json")
    parser.add_argument("--bert_path", default="/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/GroundingDINO/weights/bert-base-uncased")
    
    # 阈值
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    
    return parser.parse_args()

class SceneEvaluator:
    def __init__(self, iou_threshold=0.5, use_gt_labels_only=False):
        self.iou_threshold = iou_threshold
        self.use_gt_labels_only = use_gt_labels_only # 记录模式用于打印
        self.stats = defaultdict(lambda: defaultdict(lambda: {'tp': [], 'fp': [], 'scores': [], 'num_gt': 0}))
        
    def update(self, scene, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
        unique_labels = set(pred_labels) | set(gt_labels)
        for label in unique_labels:
            p_indices = [i for i, x in enumerate(pred_labels) if x == label]
            g_indices = [i for i, x in enumerate(gt_labels) if x == label]
            
            p_boxes_cls = pred_boxes[p_indices] if len(p_indices) > 0 else torch.empty((0, 4))
            p_scores_cls = pred_scores[p_indices] if len(p_indices) > 0 else torch.empty((0,))
            g_boxes_cls = gt_boxes[g_indices] if len(g_indices) > 0 else torch.empty((0, 4))
            
            self.stats[scene][label]['num_gt'] += len(g_boxes_cls)
            
            if len(p_boxes_cls) == 0: continue
            if len(g_boxes_cls) == 0:
                self.stats[scene][label]['fp'].extend([1] * len(p_boxes_cls))
                self.stats[scene][label]['tp'].extend([0] * len(p_boxes_cls))
                self.stats[scene][label]['scores'].extend(p_scores_cls.tolist())
                continue

            ious = box_ops.box_iou(p_boxes_cls, g_boxes_cls) 
            gt_detected = torch.zeros(len(g_boxes_cls), dtype=torch.bool)
            sorted_indices = torch.argsort(p_scores_cls, descending=True)
            
            for idx in sorted_indices:
                max_iou, max_gt_idx = torch.max(ious[idx], dim=0)
                is_tp = False
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
        print("\n" + "="*100)
        mode_str = 'GT Labels Only (Oracle)' if self.use_gt_labels_only else 'All Labels (Standard)'
        print(f"📊 评估结果 | Mode: {mode_str} | IoU: {self.iou_threshold}")
        print("="*100)
        header = "{:<30} | {:<15} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format("Scene / Class", "Precision", "Recall", "AP@50", "GT Count", "Pred", "TP", "FP")
        print(header)
        print("-" * 120)

        total_tp, total_fp, total_gt = 0, 0, 0
        for scene, class_data in self.stats.items():
            print(f"📂 场景: {scene}")
            scene_tp, scene_fp, scene_gt = 0, 0, 0
            for label, data in class_data.items():
                tp, fp = np.array(data['tp']), np.array(data['fp'])
                num_gt = data['num_gt']
                sum_tp, sum_fp = (np.sum(tp) if len(tp)>0 else 0), (np.sum(fp) if len(fp)>0 else 0)
                scene_tp += sum_tp; scene_fp += sum_fp; scene_gt += num_gt
                
                precision = sum_tp / (sum_tp + sum_fp + 1e-6)
                recall = sum_tp / (num_gt + 1e-6)
                ap = self.calculate_ap(tp, fp, num_gt)
                print("{:<30} | {:.4f}          | {:.4f}   | {:.4f}   | {:<8} | {:<8} | {:<8} | {:<8}".format(
                    f"  ├─ {label}", precision, recall, ap, num_gt, int(sum_tp+sum_fp), int(sum_tp), int(sum_fp)))
            
            s_prec = scene_tp / (scene_tp + scene_fp + 1e-6)
            s_rec = scene_tp / (scene_gt + 1e-6)
            print("{:<30} | {:.4f}          | {:.4f}   | -        | {:<8} | {:<8} | {:<8} | {:<8}".format(
                f"  └─ [Scene Total]", s_prec, s_rec, scene_gt, int(scene_tp + scene_fp), int(scene_tp), int(scene_fp)))
            print("-" * 120)
            total_tp += scene_tp; total_fp += scene_fp; total_gt += scene_gt

        all_prec = total_tp / (total_tp + total_fp + 1e-6)
        all_rec = total_tp / (total_gt + 1e-6)
        print("="*100)
        print(f"🏆 总体概览 (Overall): Precision: {all_prec:.4f} | Recall: {all_rec:.4f} | GT: {total_gt} | TP: {int(total_tp)} | FP: {int(total_fp)} | FN: {int(total_gt - total_tp)}")
        print("="*100)

def load_model(model_config_path, model_checkpoint_path, bert_path, device="cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.text_encoder_type = bert_path
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"✅ 模型加载完成: {model_checkpoint_path}")
    model.eval()
    return model.to(device)

def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image

def coco_box_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"📖 读取 Label Map: {args.label_map_file}")
    with open(args.label_map_file, 'r') as f:
        label_map = json.load(f)
    
    id_to_name = {int(k): v for k, v in label_map.items()}
    all_class_names = [str(v) for v in label_map.values()]
    FULL_PROMPT = " . ".join(all_class_names) + " ."
    
    # [核心修改] 根据参数决定 Prompt 策略
    USE_GT_LABELS_ONLY = args.use_gt_labels_only

    if USE_GT_LABELS_ONLY:
        print(f"⚠️ 模式: [GT Labels Only] - 仅使用图片中真实存在的标签进行检测")
    else:
        print(f"⚠️ 模式: [All Labels] - 使用全量 Prompt: {FULL_PROMPT[:50]}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.enabled = False
    
    model = load_model(args.config_path, args.checkpoint_path, args.bert_path, device)

    with open(args.test_json_path, 'r') as f:
        coco_data = json.load(f)
    
    gt_dict = defaultdict(list)
    for ann in coco_data['annotations']:
        gt_dict[ann['image_id']].append(ann)
    
    images_info = coco_data['images']
    print(f"📊 准备处理 {len(images_info)} 张图片...")

    evaluator = SceneEvaluator(iou_threshold=args.iou_threshold, use_gt_labels_only=USE_GT_LABELS_ONLY)
    torch.cuda.empty_cache()

    with torch.no_grad():
        for img_info in tqdm(images_info):
            file_name = img_info['file_name']
            img_id = img_info['id']
            full_image_path = os.path.join(args.image_root, file_name)
            scene_name = os.path.dirname(file_name) 
            if scene_name == "": scene_name = "Root"

            if not os.path.exists(full_image_path): continue

            # [核心修改] 动态构造 Prompt
            current_gt_anns = gt_dict.get(img_id, [])
            current_prompt = ""
            
            if USE_GT_LABELS_ONLY:
                if len(current_gt_anns) == 0:
                    current_prompt = "object ." # 负样本给一个占位符
                else:
                    unique_cat_ids = set([ann['category_id'] for ann in current_gt_anns])
                    unique_names = [id_to_name.get(cid, str(cid)) for cid in unique_cat_ids]
                    current_prompt = " . ".join(unique_names) + " ."
            else:
                current_prompt = FULL_PROMPT

            image_source, image = load_image(full_image_path)
            img_w, img_h = image_source.size

            boxes, logits, phrases = predict(
                model=model, image=image, caption=current_prompt,
                box_threshold=args.box_threshold, text_threshold=args.text_threshold, device=device
            )

            pred_boxes_xyxy = []
            pred_scores = []
            pred_labels = []
            json_results = []

            for box, score, label in zip(boxes, logits, phrases):
                cx, cy, w, h = box.tolist()
                x1 = (cx - w/2) * img_w
                y1 = (cy - h/2) * img_h
                x2 = (cx + w/2) * img_w
                y2 = (cy + h/2) * img_h
                pred_boxes_xyxy.append([x1, y1, x2, y2])
                pred_scores.append(score.item())
                pred_labels.append(label)
                json_results.append({
                    "label": label, "score": round(score.item(), 4),
                    "box_pixel_xyxy": [int(x1), int(y1), int(x2), int(y2)]
                })

            gt_boxes_xyxy = []
            gt_labels = []
            for ann in gt_dict.get(img_id, []):
                gt_boxes_xyxy.append(coco_box_to_xyxy(ann['bbox']))
                gt_labels.append(id_to_name.get(ann['category_id'], str(ann['category_id'])))

            # 评估更新
            t_pred_boxes = torch.tensor(pred_boxes_xyxy) if pred_boxes_xyxy else torch.empty((0, 4))
            t_pred_scores = torch.tensor(pred_scores) if pred_scores else torch.empty((0,))
            t_gt_boxes = torch.tensor(gt_boxes_xyxy) if gt_boxes_xyxy else torch.empty((0, 4))
            evaluator.update(scene_name, t_pred_boxes, t_pred_scores, pred_labels, t_gt_boxes, gt_labels)

            # 保存 JSON 和 图片
            base_name = os.path.basename(file_name)
            json_save_path = os.path.join(args.output_dir, "vis_" + os.path.splitext(base_name)[0] + ".json")
            with open(json_save_path, "w", encoding='utf-8') as f_json:
                json.dump({"file_name": file_name, "height": img_h, "width": img_w, "objects": json_results}, f_json, indent=4, ensure_ascii=False)
            
            annotated_frame = annotate(image_source=np.asarray(image_source), boxes=boxes, logits=logits, phrases=phrases)
            cv2.imwrite(os.path.join(args.output_dir, "vis_" + base_name), annotated_frame)
            
            torch.cuda.empty_cache()

    evaluator.print_results()
    print(f"\n✅ 全部完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()