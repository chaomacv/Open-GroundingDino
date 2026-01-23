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
from groundingdino.util.inference import predict

def get_args():
    parser = argparse.ArgumentParser(description="GroundingDINO Batch Evaluation")
    parser.add_argument("--checkpoint_path", required=True, help="模型权重路径")
    parser.add_argument("--output_dir", required=True, help="结果输出目录")
    parser.add_argument("--use_gt_labels_only", action="store_true", help="是否仅使用 GT 标签 (Oracle Mode)")
    parser.add_argument("--config_path", default="/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/config/cfg_odvg.py")
    parser.add_argument("--test_json_path", default="")
    parser.add_argument("--image_root", default="")
    parser.add_argument("--label_map_file", default="")
    parser.add_argument("--bert_path", default="/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/GroundingDINO/weights/bert-base-uncased")
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.35)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--external_prompt_json", default=None, help="使用外部JSON(如Qwen3结果)的内容作为Prompt")
    return parser.parse_args()

def custom_annotate(image_pil, boxes, logits, phrases):
    """
    使用 OpenCV 原生绘制，避免 supervision 版本冲突
    boxes: Tensor (N, 4) cx, cy, w, h normalized
    """
    # PIL (RGB) -> OpenCV (BGR)
    image_cv = np.array(image_pil)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    h_img, w_img, _ = image_cv.shape
    
    if boxes is None or boxes.numel() == 0:
        return image_cv

    # 1. 反归一化: (cx, cy, w, h) 0~1 -> 像素坐标
    boxes = boxes * torch.Tensor([w_img, h_img, w_img, h_img])
    
    # 2. [核心修复] 手动转换 cxcywh -> xyxy，不依赖外部库函数
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    
    # 堆叠并转为 numpy int
    xyxy = torch.stack((x1, y1, x2, y2), dim=-1).numpy().astype(int)
    logits = logits.numpy()

    for box, score, label in zip(xyxy, logits, phrases):
        x1, y1, x2, y2 = box
        
        # 3. 绘制矩形 (绿色, 线宽2)
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 4. 绘制标签
        text = f"{label} {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 背景框
        cv2.rectangle(image_cv, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)
        # 文字
        cv2.putText(image_cv, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)

    return image_cv

class SceneEvaluator:
    def __init__(self, iou_threshold=0.5, use_gt_labels_only=False):
        self.iou_threshold = iou_threshold
        self.use_gt_labels_only = use_gt_labels_only 
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
        sorted_scenes = sorted(self.stats.keys())
        
        for scene in sorted_scenes:
            class_data = self.stats[scene]
            print(f"📂 场景: {scene}")
            scene_tp, scene_fp, scene_gt = 0, 0, 0
            
            for label in sorted(class_data.keys()):
                data = class_data[label]
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
    
    VALID_LABELS_SET = set(all_class_names)
    print(f"🎯 [Filter Mode] 仅计算以下 {len(VALID_LABELS_SET)} 类: {list(VALID_LABELS_SET)[:5]}...")

    FULL_PROMPT = " . ".join(all_class_names) + " ."
    USE_GT_LABELS_ONLY = args.use_gt_labels_only

    # ================= [新增] 加载外部 Prompt 数据 (Qwen3 Results) =================
    external_prompts_map = {}
    if args.external_prompt_json and os.path.exists(args.external_prompt_json):
        print(f"📖 [External Prompt] 正在加载外部 Prompt 源: {args.external_prompt_json}")
        try:
            with open(args.external_prompt_json, 'r') as f:
                qwen_data = json.load(f)
            
            # 建立映射: relative_path -> pred_anomaly_class
            # Qwen结果里的路径可能有 .enc 后缀，我们统一去掉 .enc 进行匹配
            count_loaded = 0
            for item in qwen_data.get("evaluation", {}).get("details", []):
                rel_path = item["model_output"]["relative_path"]
                # 去除 .enc 后缀以确保匹配 (例如: a.jpg.enc -> a.jpg)
                clean_key = rel_path.replace(".enc", "")
                
                # 获取预测类别 list
                pred_classes = item.get("pred_anomaly_class", [])
                external_prompts_map[clean_key] = pred_classes
                count_loaded += 1
                
            print(f"✅ [External Prompt] 已成功索引 {count_loaded} 条外部数据")
        except Exception as e:
            print(f"❌ [Error] 加载外部 JSON 失败: {e}")
            return
    # ==============================================================================

    if args.external_prompt_json:
        print(f"⚠️ 模式: [External Prompt] - 优先使用 {os.path.basename(args.external_prompt_json)} 提供的类别")
    elif USE_GT_LABELS_ONLY:
        print(f"⚠️ 模式: [GT Labels Only] - 仅使用图片中真实存在的标签进行检测")
    else:
        print(f"⚠️ 模式: [All Labels] - 使用全量 Prompt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.enabled = False
    
    model = load_model(args.config_path, args.checkpoint_path, args.bert_path, device)

    print(f"📖 读取测试集 Annotation: {args.test_json_path}")
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

            if not os.path.exists(full_image_path): 
                # print(f"Skipping missing image: {full_image_path}")
                continue

            # [Filter] 1. 过滤 GT Annotations (仅保留在 LabelMap 中的)
            current_gt_anns = []
            for ann in gt_dict.get(img_id, []):
                lb = id_to_name.get(ann['category_id'], str(ann['category_id']))
                if lb in VALID_LABELS_SET:
                    current_gt_anns.append(ann)

            # ================= [修改] Prompt 构建逻辑 =================
            current_prompt = ""
            
            # 优先级 1: 外部 JSON (Qwen3)
            # 查找键值需去掉 .enc
            lookup_key = file_name.replace(".enc", "")
            
            if args.external_prompt_json:
                if lookup_key in external_prompts_map:
                    # 获取 Qwen3 预测的类别列表
                    pred_classes = external_prompts_map[lookup_key]
                    
                    if len(pred_classes) == 0:
                        # Qwen3 认为无异常 -> Prompt: "clean ." (GD 应该检测不出任何东西)
                        current_prompt = "clean ."
                    else:
                        # 过滤: 仅保留 GD Label Map 中存在的类别，防止非法 Prompt
                        valid_preds = [cls for cls in pred_classes if cls in VALID_LABELS_SET]
                        
                        if len(valid_preds) > 0:
                            current_prompt = " . ".join(valid_preds) + " ."
                        else:
                            # 预测了类别但都不在 LabelMap 里 (例如 Qwen 幻觉了新词)
                            current_prompt = "object ."
                else:
                    # 图片不在 Qwen 结果中，兜底策略
                    current_prompt = "object ."
            
            # 优先级 2: GT Labels Only (Oracle)
            elif USE_GT_LABELS_ONLY:
                if len(current_gt_anns) == 0:
                    current_prompt = "object ." 
                else:
                    unique_cat_ids = set([ann['category_id'] for ann in current_gt_anns])
                    unique_names = [id_to_name.get(cid, str(cid)) for cid in unique_cat_ids]
                    current_prompt = " . ".join(unique_names) + " ."
            
            # 优先级 3: 全量 Prompt (Standard)
            else:
                current_prompt = FULL_PROMPT
            # ==========================================================

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
            
            # 用于可视化的临时列表
            vis_boxes = []
            vis_logits = []
            vis_phrases = []

            # [Filter] 2. 过滤预测结果 (不在 LabelMap 的结果丢弃)
            for box, score, label in zip(boxes, logits, phrases):
                if label not in VALID_LABELS_SET:
                    continue
                
                vis_boxes.append(box)
                vis_logits.append(score)
                vis_phrases.append(label)

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
            for ann in current_gt_anns:
                gt_boxes_xyxy.append(coco_box_to_xyxy(ann['bbox']))
                gt_labels.append(id_to_name.get(ann['category_id'], str(ann['category_id'])))

            t_pred_boxes = torch.tensor(pred_boxes_xyxy) if pred_boxes_xyxy else torch.empty((0, 4))
            t_pred_scores = torch.tensor(pred_scores) if pred_scores else torch.empty((0,))
            t_gt_boxes = torch.tensor(gt_boxes_xyxy) if gt_boxes_xyxy else torch.empty((0, 4))
            
            evaluator.update(scene_name, t_pred_boxes, t_pred_scores, pred_labels, t_gt_boxes, gt_labels)

            # 保存 JSON
            base_name = os.path.basename(file_name)
            # 如果文件名里有 .enc，保存结果时也可以去掉，或者保留，这里保留原始文件名逻辑
            json_save_path = os.path.join(args.output_dir, "vis_" + os.path.splitext(base_name)[0] + ".json")
            with open(json_save_path, "w", encoding='utf-8') as f_json:
                json.dump({
                    "file_name": file_name, 
                    "prompt_used": current_prompt, # [Debug] 记录实际使用的 Prompt
                    "height": img_h, 
                    "width": img_w, 
                    "objects": json_results
                }, f_json, indent=4, ensure_ascii=False)
            
            # 绘制并保存图片
            if len(vis_boxes) > 0:
                t_vis_boxes = torch.stack(vis_boxes)
                t_vis_logits = torch.stack(vis_logits)
            else:
                t_vis_boxes = torch.empty((0, 4))
                t_vis_logits = torch.empty((0,))
            
            annotated_frame = custom_annotate(image_source, t_vis_boxes, t_vis_logits, vis_phrases)
            cv2.imwrite(os.path.join(args.output_dir, "vis_" + base_name), annotated_frame)
            
            torch.cuda.empty_cache()

    evaluator.print_results()
    print(f"\n✅ 全部完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()