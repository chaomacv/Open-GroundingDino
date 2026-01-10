import os
import torch
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict

# ================= ⚙️ 基础配置 =================
CONFIG_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/config/cfg_odvg.py"
CHECKPOINT_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/logs/railway_4gpu_80_10_10/checkpoint_best_regular.pth"
TEST_JSON_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/test_split_coco.json"
IMAGE_ROOT = "/opt/data/private/xjx/RailMind/高速铁路无人机图像/FilteredLabeled"
LABEL_MAP_FILE = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/label_map.json"
BERT_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/GroundingDINO/weights/bert-base-uncased"

# GT JSON 所在的文件夹 (用于加载 Ground Truth)
DIR_GT = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/mycode/vis_gt_results"

# 匹配 IoU 阈值
IOU_THRESHOLD = 0.8
# ===============================================

def load_model(model_config_path, model_checkpoint_path, device="cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.text_encoder_type = BERT_PATH
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
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

def load_gt_detections(json_path):
    """加载真实标注 GT"""
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    normalized_objs = []
    if 'objects' not in data: return []
    for obj in data['objects']:
        if 'box_pixel_xyxy' in obj:
            box = obj['box_pixel_xyxy']
        elif 'bbox' in obj:
             continue
        else:
            continue
        label = obj.get('label', 'unknown')
        normalized_objs.append({'label': label, 'box': box})
    return normalized_objs

def compare_single_pair(objs_gt, objs_pred):
    """内存中对比一对结果"""
    stats = {'matched': 0, 'missed': 0, 'extra': 0}
    matched_pred_indices = set()
    
    for gt in objs_gt:
        best_iou = -1
        best_idx = -1
        for idx, pred in enumerate(objs_pred):
            if idx in matched_pred_indices: continue
            if gt['label'] != pred['label']: continue
            iou = compute_iou(gt['box'], pred['box'])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        
        if best_iou >= IOU_THRESHOLD:
            matched_pred_indices.add(best_idx)
            stats['matched'] += 1
        else:
            stats['missed'] += 1
            
    stats['extra'] = len(objs_pred) - len(matched_pred_indices)
    return stats

def run_evaluation(model, device, text_prompt, images_info, threshold, gt_map):
    """执行一次完整的评估循环"""
    total_stats = {'gt': 0, 'pred': 0, 'matched': 0, 'missed': 0, 'extra': 0}
    
    # 遍历所有图片
    # 为了速度，这里不再用 tqdm 显示详细进度条，只在外部显示轮次
    for img_info in images_info:
        file_name = img_info['file_name']
        full_image_path = os.path.join(IMAGE_ROOT, file_name)
        
        # 1. 找到对应的 GT 文件
        # 假设文件名 gt_xxx.json 对应 xxx.jpg
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        gt_filename = f"gt_{base_name}.json"
        
        # 如果没有 GT 文件，跳过对比
        if base_name not in gt_map:
            continue
            
        gt_path = os.path.join(DIR_GT, gt_filename)
        objs_gt = load_gt_detections(gt_path)
        
        if not os.path.exists(full_image_path): continue

        # 2. 模型推理
        image_pil, image = load_image(full_image_path)
        image = image.to(device)
        img_w, img_h = image_pil.size

        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=text_prompt,
                box_threshold=threshold,
                text_threshold=threshold,
                device=device
            )

        # 3. 格式化预测结果
        objs_pred = []
        for box, score, label in zip(boxes, logits, phrases):
            box_norm = box.tolist()
            cx, cy, w, h = box_norm
            x1 = int((cx - w/2) * img_w)
            y1 = int((cy - h/2) * img_h)
            x2 = int((cx + w/2) * img_w)
            y2 = int((cy + h/2) * img_h)
            objs_pred.append({
                'label': label,
                'score': score.item(),
                'box': [x1, y1, x2, y2]
            })

        # 4. 对比
        res = compare_single_pair(objs_gt, objs_pred)
        
        # 5. 累加
        total_stats['gt'] += len(objs_gt)
        total_stats['pred'] += len(objs_pred)
        total_stats['matched'] += res['matched']
        total_stats['missed'] += res['missed']
        total_stats['extra'] += res['extra']

    # 计算指标
    recall = total_stats['matched'] / total_stats['gt'] if total_stats['gt'] > 0 else 0
    precision = total_stats['matched'] / total_stats['pred'] if total_stats['pred'] > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return recall, precision, f1, total_stats

def main():
    # 1. 初始化
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 加载模型中...")
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device)
    
    # 2. 准备 Prompt
    with open(LABEL_MAP_FILE, 'r') as f:
        label_map = json.load(f)
    class_names = [str(name) for name in label_map.values() if isinstance(name, (str, int))]
    text_prompt = " . ".join(class_names) + " ."
    
    # 3. 读取测试集列表
    with open(TEST_JSON_PATH, 'r') as f:
        coco_data = json.load(f)
    images_info = coco_data['images']
    
    # 4. 预先建立 GT 索引
    gt_files = [f for f in os.listdir(DIR_GT) if f.startswith("gt_")]
    gt_map = {f[3:].replace(".json", ""): f for f in gt_files} # key: "123", value: "gt_123.json"
    
    print(f"📊 测试集包含 {len(images_info)} 张图片，找到 {len(gt_map)} 个 GT 文件。")
    print("🔄 开始阈值搜索 (0.10 -> 0.50)...")
    print("-" * 80)
    print(f"{'Threshold':<10} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6}")
    print("-" * 80)

    best_f1 = -1
    best_res = None
    best_thresh = -1
    
    # 5. 循环阈值
    # np.arange(0.1, 0.51, 0.05) 会生成 [0.1, 0.15, ..., 0.5]
    thresholds = np.arange(0.1, 0.51, 0.05)
    
    for thr in thresholds:
        thr = round(thr, 2) # 避免浮点数精度问题
        
        rec, prec, f1, stats = run_evaluation(model, device, text_prompt, images_info, thr, gt_map)
        
        print(f"{thr:<10} | {rec:<10.2%} | {prec:<10.2%} | {f1:<10.4f} | {stats['matched']:<6} | {stats['extra']:<6} | {stats['missed']:<6}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_res = (rec, prec, f1, stats)
            best_thresh = thr

    print("-" * 80)
    print("\n🏆 最佳结果 (Best Result):")
    print(f"🔥 最佳阈值 (Threshold): {best_thresh}")
    print(f"🔵 召回率 (Recall):    {best_res[0]:.2%}")
    print(f"🔴 精确率 (Precision): {best_res[1]:.2%}")
    print(f"⭐ F1-Score:          {best_res[2]:.4f}")
    
    tp, fp, fn = best_res[3]['matched'], best_res[3]['extra'], best_res[3]['missed']
    print(f"📦 详情: 正确检测(TP)={tp}, 误检(FP)={fp}, 漏检(FN)={fn}")

if __name__ == "__main__":
    main()