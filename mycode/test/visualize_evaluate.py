import argparse
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
from tqdm import tqdm
import cv2
from collections import defaultdict
import torchvision.ops.boxes as box_ops

# 引入 GroundingDINO 的必要模块
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
# 引入官方推理工具
from groundingdino.util.inference import predict, annotate

# ================= ⚙️ 配置区域 =================
# 1. 模型配置
CONFIG_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/config/cfg_odvg.py"
CHECKPOINT_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/logs/railway_4gpu_wandb_full_label/checkpoint_best_regular.pth"
TEST_JSON_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/test_split_coco_fixed.json"
IMAGE_ROOT = "/opt/data/private/xjx/RailMind/database/test/基准测试_1229/基准测试数据集"
LABEL_MAP_FILE = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/label_map.json"
OUTPUT_DIR = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/0110_full_test_benchmark" # 建议修改输出目录名以区分
BERT_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/GroundingDINO/weights/bert-base-uncased"



# 2. 阈值设置
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

# 3. [新增] Prompt 模式选择
# True: 仅使用该图片真实包含的标签 (Oracle Mode)
# False: 使用 Label Map 中所有标签 (Standard Mode)
USE_GT_LABELS_ONLY = False
# =================================================

# ... [SceneEvaluator 类保持不变] ...
class SceneEvaluator:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        # 数据结构: {scene_name: {class_name: {'tp': [], 'fp': [], 'scores': [], 'num_gt': 0}}}
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
            
            if len(p_boxes_cls) == 0:
                continue

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
        print(f"📊 评估结果 (IoU Threshold = {self.iou_threshold}) | Mode: {'GT Labels Only' if USE_GT_LABELS_ONLY else 'All Labels'}")
        print("="*100)
        header = "{:<30} | {:<15} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format(
            "Scene / Class", "Precision", "Recall", "AP@50", "GT Count", "Pred Count", "TP", "FP")
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
                    f"  ├─ {label}", precision, recall, ap, num_gt, len(tp), int(sum_tp), int(sum_fp)))
            
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

# ... [工具函数 load_model, load_image, coco_box_to_xyxy 保持不变] ...
def load_model(model_config_path, model_checkpoint_path, device="cuda"):
    args = SLConfig.fromfile(model_config_path)
    print(f"🔄 强制使用本地 BERT 路径: {BERT_PATH}")
    args.text_encoder_type = BERT_PATH
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"✅ 模型加载完成")
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
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 准备全局 Label Map 
    print(f"📖 读取 Label Map: {LABEL_MAP_FILE}")
    with open(LABEL_MAP_FILE, 'r') as f:
        label_map = json.load(f)
    
    id_to_name = {int(k): v for k, v in label_map.items()}
    all_class_names = list(label_map.values())
    all_class_names = [str(name) for name in all_class_names if isinstance(name, (str, int))]
    
    # 构造 全量 Prompt
    FULL_PROMPT = " . ".join(all_class_names) + " ."
    
    if USE_GT_LABELS_ONLY:
        print(f"⚠️ 模式: [GT Labels Only] - 仅使用图片中真实存在的标签进行检测")
    else:
        print(f"⚠️ 模式: [All Labels] - 使用所有标签进行检测")
        print(f"📝 全量 Prompt: {FULL_PROMPT[:50]}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device)

    # 2. 读取数据
    with open(TEST_JSON_PATH, 'r') as f:
        coco_data = json.load(f)
    
    gt_dict = defaultdict(list)
    for ann in coco_data['annotations']:
        gt_dict[ann['image_id']].append(ann)
    
    images_info = coco_data['images']
    print(f"📊 准备处理 {len(images_info)} 张图片...")

    evaluator = SceneEvaluator(iou_threshold=IOU_THRESHOLD)
    
    # 显式清空显存
    torch.cuda.empty_cache()

    # 3. 推理循环
    with torch.no_grad():
        for img_info in tqdm(images_info):
            file_name = img_info['file_name']
            img_id = img_info['id']
            full_image_path = os.path.join(IMAGE_ROOT, file_name)
            scene_name = os.path.dirname(file_name) 
            if scene_name == "": scene_name = "Root"

            if not os.path.exists(full_image_path):
                continue

            # =========================================================
            # [核心修改] 动态构造 Prompt
            # =========================================================
            
            # 1. 获取该图所有的 GT Category ID
            current_gt_anns = gt_dict.get(img_id, [])
            
            # 2. 确定当前使用的 Prompt
            current_prompt = ""
            
            if USE_GT_LABELS_ONLY:
                if len(current_gt_anns) == 0:
                    # 如果该图没有任何标注 (负样本)，我们如何处理？
                    # 策略A: 跳过检测 (因为没有目标可以检测) -> 这样算出来全是 TN
                    # 策略B: 仍然给全量标签测试误检 -> 这样更有意义
                    # 这里采用策略B，或者你可以给一个空 prompt，但 GroundingDINO 可能报错
                    # 为了避免报错，如果是空标注图片，我们给一个 "random" 或者沿用 Full Prompt
                    # 建议：如果是空标注，跳过本次循环的 inference，或者给一个必定不存在的类
                    if len(current_gt_anns) == 0:
                        # 这是一个只有背景的图，为了测试误检，我们可以随便给一个 Prompt
                        # 或者跳过。这里选择跳过 Inference，直接记录 GT=0
                        # 也可以选择给一个 'object .' 看它会不会乱检
                        current_prompt = "object ." 
                    else:
                        # 提取该图包含的唯一类别名称
                        unique_cat_ids = set([ann['category_id'] for ann in current_gt_anns])
                        unique_names = [id_to_name.get(cid, str(cid)) for cid in unique_cat_ids]
                        current_prompt = " . ".join(unique_names) + " ."
                else:
                    # 有标注，使用 GT 类别
                    unique_cat_ids = set([ann['category_id'] for ann in current_gt_anns])
                    unique_names = [id_to_name.get(cid, str(cid)) for cid in unique_cat_ids]
                    current_prompt = " . ".join(unique_names) + " ."
            else:
                # 使用全量标签
                current_prompt = FULL_PROMPT

            # 3. 加载图片与推理
            image_source, image = load_image(full_image_path)
            img_w, img_h = image_source.size

            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=current_prompt, # 使用动态构建的 prompt
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=device
            )

            # =========================================================
            # 数据准备 (与之前相同)
            # =========================================================
            pred_boxes_xyxy = []
            pred_scores = []
            pred_labels = []
            json_results = []

            for box, score, label in zip(boxes, logits, phrases):
                box_norm = box.tolist() 
                cx, cy, w, h = box_norm
                x1 = (cx - w/2) * img_w
                y1 = (cy - h/2) * img_h
                x2 = (cx + w/2) * img_w
                y2 = (cy + h/2) * img_h
                
                pred_boxes_xyxy.append([x1, y1, x2, y2])
                pred_scores.append(score.item())
                pred_labels.append(label)

                json_results.append({
                    "label": label,
                    "score": round(score.item(), 4),
                    "box_pixel_xyxy": [int(x1), int(y1), int(x2), int(y2)]
                })

            # 处理 GT
            gt_boxes_xyxy = []
            gt_labels = []
            for ann in current_gt_anns:
                xyxy = coco_box_to_xyxy(ann['bbox'])
                gt_boxes_xyxy.append(xyxy)
                cat_name = id_to_name.get(ann['category_id'], str(ann['category_id'])) 
                gt_labels.append(cat_name)

            # 更新评估器
            if len(pred_boxes_xyxy) > 0:
                t_pred_boxes = torch.tensor(pred_boxes_xyxy)
                t_pred_scores = torch.tensor(pred_scores)
            else:
                t_pred_boxes = torch.empty((0, 4))
                t_pred_scores = torch.empty((0,))
                
            if len(gt_boxes_xyxy) > 0:
                t_gt_boxes = torch.tensor(gt_boxes_xyxy)
            else:
                t_gt_boxes = torch.empty((0, 4))
                
            evaluator.update(scene_name, t_pred_boxes, t_pred_scores, pred_labels, t_gt_boxes, gt_labels)

            # 保存结果
            base_name = os.path.basename(file_name)
            json_save_name = "vis_" + os.path.splitext(base_name)[0] + ".json"
            json_save_path = os.path.join(OUTPUT_DIR, json_save_name)
            with open(json_save_path, "w", encoding='utf-8') as f_json:
                json.dump({"file_name": file_name, "height": img_h, "width": img_w, "objects": json_results}, f_json, indent=4, ensure_ascii=False)

            annotated_frame = annotate(image_source=np.asarray(image_source), boxes=boxes, logits=logits, phrases=phrases)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "vis_" + os.path.basename(file_name)), annotated_frame)

    evaluator.print_results()
    print(f"\n✅ 全部完成！结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()