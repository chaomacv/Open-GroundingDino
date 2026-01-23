import json
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ================= ⚙️ 配置区域 =================

# 1. 输入文件路径
IMG_PATH = "/opt/data/private/xjx/RailMind/高速铁路无人机图像/FilteredLabeled/声屏障-仅缺陷标注-检测框/60752222094958_0009_Z_9.JPG"
JSON_PATH = "/opt/data/private/xjx/RailMind/高速铁路无人机图像/FilteredLabeled/声屏障-仅缺陷标注-检测框/60752222094958_0009_Z_9.json"

# 2. 输出保存路径 (保存到当前目录下)
OUTPUT_PATH = "vis_result_60752222094958_0009_Z_9.jpg"

# 3. 字体路径 (为了在图片上显示中文，必须指定一个支持中文的字体文件)
# 如果是 Ubuntu/Debian 系统，通常在 /usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf
# 或者你可以上传一个 simhei.ttf 到同级目录
FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf" 
# 如果找不到字体，脚本会回退到默认字体（中文可能会显示为方框）

# ===============================================

def cv2_to_pil(cv2_img):
    """ 将 OpenCV 图片转换为 PIL 图片 """
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_img):
    """ 将 PIL 图片转换为 OpenCV 图片 """
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

def parse_annotation(json_data):
    """ 
    解析 JSON 数据，提取 bbox 和 label 
    支持 LabelMe 格式 ('shapes') 和 自定义 Detection 格式
    """
    objects = []
    
    # 模式 1: LabelMe 格式 (shapes -> points)
    if "shapes" in json_data:
        for shape in json_data["shapes"]:
            label = shape.get("label", "unknown")
            points = np.array(shape.get("points", []))
            if len(points) > 0:
                x_min = min(points[:, 0])
                y_min = min(points[:, 1])
                x_max = max(points[:, 0])
                y_max = max(points[:, 1])
                objects.append({"bbox": [x_min, y_min, x_max, y_max], "label": label})
    
    # 模式 2: ODVG/Detection 格式 (detection -> instances)
    elif "detection" in json_data and "instances" in json_data["detection"]:
        for inst in json_data["detection"]["instances"]:
            label = str(inst.get("label", "unknown")) # 可能是数字ID
            # 尝试获取 category name 如果有的话
            if "category" in inst:
                label = inst["category"]
                
            bbox = inst["bbox"] # 通常是 [x, y, w, h] 或 [x1, y1, x2, y2]
            # 这里简单判断一下，如果 w, h 比较小可能需要转换，暂定为 xyxy
            # 如果是 xywh: x2 = x + w, y2 = y + h
            # 假设是 xyxy (GroundingDINO常用)
            objects.append({"bbox": bbox, "label": label})
            
    # 模式 3: 通用 Objects 格式
    elif "objects" in json_data:
        for obj in json_data["objects"]:
            label = obj.get("label", "obj")
            bbox = obj.get("bbox") # [x1, y1, x2, y2]
            objects.append({"bbox": bbox, "label": label})

    return objects

def main():
    if not os.path.exists(IMG_PATH):
        print(f"❌ 错误: 找不到图片文件 {IMG_PATH}")
        return
    if not os.path.exists(JSON_PATH):
        print(f"❌ 错误: 找不到标注文件 {JSON_PATH}")
        return

    print(f"🖼️ 读取图片: {IMG_PATH}")
    # OpenCV 读取图片 (处理中文路径可能需要 np.fromfile 技巧，但在 Linux 通常直接支持)
    image = cv2.imread(IMG_PATH)
    if image is None:
        print("❌ 读取图片失败，可能是文件损坏或路径编码问题")
        return

    print(f"📖 读取 JSON: {JSON_PATH}")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 解析数据
    objects = parse_annotation(json_data)
    print(f"🔍 发现 {len(objects)} 个标注目标")

    # 转为 PIL 以便绘制中文
    pil_image = cv2_to_pil(image)
    draw = ImageDraw.Draw(pil_image)
    
    # 加载字体
    try:
        font = ImageFont.truetype(FONT_PATH, size=40)
    except:
        print("⚠️ 未找到指定字体，使用默认字体 (中文可能无法显示)")
        font = ImageFont.load_default()

    # 绘制循环
    for obj in objects:
        bbox = obj["bbox"]
        label = obj["label"]
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # 1. 画框 (红色, 线宽 5)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=5)
        
        # 2. 画标签背景和文字
        # 计算文字大小
        if hasattr(font, 'getbbox'):
            text_bbox = font.getbbox(label)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        else:
            text_w, text_h = draw.textsize(label, font)

        # 绘制文字背景 (红色实心)
        draw.rectangle([x1, y1 - text_h - 10, x1 + text_w + 10, y1], fill=(255, 0, 0))
        # 绘制文字 (白色)
        draw.text((x1 + 5, y1 - text_h - 10), label, fill=(255, 255, 255), font=font)

    # 转回 OpenCV 并保存
    result_img = pil_to_cv2(pil_image)
    cv2.imwrite(OUTPUT_PATH, result_img)
    print(f"✅ 可视化结果已保存至: {os.path.abspath(OUTPUT_PATH)}")

if __name__ == "__main__":
    main()