import argparse
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 引入 GroundingDINO 的必要模块
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

# ================= 配置区域 =================
# 1. 配置文件路径 (用你训练用的那个 cfg_odvg.py)
CONFIG_PATH = "config/cfg_odvg.py"

# 2. 权重文件路径 (用那个最好的!)
CHECKPOINT_PATH = "logs/railway_4gpu_run/checkpoint_best_regular.pth"

# 3. 想要测试的图片路径
IMAGE_PATH = "/opt/data/private/xjx/RailMind/高速铁路无人机图像/FilteredLabeled/你的某张测试图.jpg" 

# 4. 输出路径
OUTPUT_PATH = "prediction_result.jpg"

# 5. 你的标签映射 (必须和 label_map.json 里的名字一致，或者直接写在这里)
# 如果你的 label_map.json 里是 {"1": "insulator", ...}，这里填英文名即可
# 这里的顺序不重要，关键是模型输出的时候能对上
# 简单起见，我们直接加载 label_map 文件也行，或者手动写个列表
LABEL_MAP_FILE = "label_map.json" 
# ===========================================

def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"模型加载完成: {load_res}")
    model.eval()
    return model

def plot_boxes(image_pil, boxes, logits, phrases):
    draw = ImageDraw.Draw(image_pil)
    # 尝试加载字体，如果报错就用默认的
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    w, h = image_pil.size
    
    # boxes 是归一化的 [cx, cy, w, h]，需要转回 [x1, y1, x2, y2]
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    for box, logit, phrase in zip(xyxy, logits, phrases):
        x1, y1, x2, y2 = box
        # 只有置信度 > 0.35 才画
        if logit > 0.35:
            # 画框 (红色，宽度3)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            # 写字
            label_text = f"{phrase} {logit:.2f}"
            draw.text((x1, y1), label_text, fill="red", font=font)
    
    return image_pil

# 辅助函数：坐标转换
def box_convert(boxes, in_fmt, out_fmt):
    # 简单的 cxcywh 转 xyxy 实现
    if in_fmt == "cxcywh" and out_fmt == "xyxy":
        cx, cy, w, h = boxes.unbind(-1)
        b = [(cx - 0.5 * w), (cy - 0.5 * h),
             (cx + 0.5 * w), (cy + 0.5 * h)]
        return torch.stack(b, dim=-1)
    return boxes

def main():
    # 1. 加载模型
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
    model = model.to("cuda")

    # 2. 加载图片
    image_pil, image = load_image(IMAGE_PATH)
    image = image.to("cuda")

    # 3. 推理
    # GroundingDINO 需要 text prompt。
    # 如果你是 Close-Set (闭集) 训练，这里其实不太需要 Prompt，
    # 但为了兼容代码，我们可以输入所有类别的拼接字符串，或者空着看效果
    # ⚠️ 既然是 ODVG 训练，它的 query 机制略有不同。
    # 最简单的验证方法是：直接跑 forward
    
    with torch.no_grad():
        outputs = model(image[None], captions=[""]) # captions 为空，因为我们不是检测文本
    
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, num_class)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # 4. 过滤结果
    # 取最大置信度的类别
    # 注意：这里需要 label_map 来把 ID 转回名字
    import json
    with open(LABEL_MAP_FILE, 'r') as f:
        label_map = json.load(f)
    
    # 确保是 {id: name}
    if isinstance(list(label_map.values())[0], int):
         id_to_name = {v: k for k, v in label_map.items()}
    else:
         id_to_name = {int(k): v for k, v in label_map.items()}

    # 过滤
    filt_mask = logits.max(dim=1)[0] > 0.35
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]
    
    #以此获取类别名
    pred_phrases = []
    for logit in logits_filt:
        class_id = logit.argmax().item()
        # label_map 通常从 1 开始，或者 0 开始，取决于你的 dataset 实现
        # 这里可能需要 +1 或者直接取
        name = id_to_name.get(class_id, str(class_id))
        pred_phrases.append(name)
        
    scores = logits_filt.max(dim=1)[0]

    # 5. 画图
    res_image = plot_boxes(image_pil, boxes_filt, scores, pred_phrases)
    res_image.save(OUTPUT_PATH)
    print(f"✅ 预测完成！结果已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()