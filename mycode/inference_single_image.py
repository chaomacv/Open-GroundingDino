import os
import torch
import numpy as np
import cv2
from PIL import Image

# 引入 GroundingDINO 模块
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict, annotate

# ================= ⚙️ 配置区域 (随时修改这里) =================

# 1. 想找什么？(Prompt)
# 格式要求：英文单词，用 " . " (空格+点+空格) 分隔，最后也要加点
# 示例： "insulator . nut_missing . bird_nest ."
TEXT_PROMPT = "rustypaint . corrosion . guard_rust . rustyfence . rustypole . rustyplate ."

# 2. 只有一张图，路径填这里
IMAGE_PATH = "/opt/data/private/xjx/RailMind/高速铁路无人机图像/FilteredLabeled/声屏障-仅缺陷标注-检测框/60752222094958_0009_Z_9.JPG"

# 3. 结果保存路径
OUTPUT_IMAGE_PATH = "result_single.jpg"

# 4. 模型配置 (保持不变)
CONFIG_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/config/cfg_odvg.py"
CHECKPOINT_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/logs/0113/model3_only_fullneg/checkpoint_best_regular.pth"
BERT_PATH = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/GroundingDINO/weights/bert-base-uncased"

# 5. 阈值 (根据效果微调)
BOX_THRESHOLD = 0.35   # 框的置信度阈值
TEXT_THRESHOLD = 0.35  # 文本匹配阈值

# =========================================================

def load_model(model_config_path, model_checkpoint_path, device="cuda"):
    args = SLConfig.fromfile(model_config_path)
    
    # 强制使用本地 BERT
    print(f"🔄 强制使用本地 BERT 路径: {BERT_PATH}")
    args.text_encoder_type = BERT_PATH
    
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
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

def main():
    # 1. 检查图片是否存在
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 错误: 找不到图片 {IMAGE_PATH}")
        return

    # 2. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device)

    # 3. 加载图片
    print(f"🖼️ 正在处理图片: {IMAGE_PATH}")
    image_source, image = load_image(IMAGE_PATH)

    # 4. 推理
    print(f"🔍 检测目标 Prompt: {TEXT_PROMPT}")
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=device
    )

    # 5. 打印检测结果到终端
    if len(boxes) > 0:
        print(f"✅ 检测到 {len(boxes)} 个目标:")
        for phrase, logit in zip(phrases, logits):
            print(f"   - {phrase}: {logit:.2f}")
    else:
        print("⚠️ 未检测到任何目标。")

    # 6. 画图并保存
    annotated_frame = annotate(image_source=np.asarray(image_source), boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_frame)
    print(f"💾 结果已保存至: {os.path.abspath(OUTPUT_IMAGE_PATH)}")

if __name__ == "__main__":
    main()