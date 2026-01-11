import os
import glob
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ================= 配置区域 =================
# 数据集根目录
dataset_root = "/opt/data/private/xjx/RailMind/高速铁路无人机图像/FilteredLabeled"
# ===========================================

def parse_xml_to_dict(xml_file):
    """解析 VOC 格式 XML 返回 LabelMe 需要的字段"""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 1. 获取图片基本信息
    # 注意：有时候 XML 里的 filename 不带后缀或后缀不对，最好在保存时单独校验
    # 这里我们只取 filename 字符串用于 LabelMe 的 imagePath
    filename_node = root.find('filename')
    filename = filename_node.text if filename_node is not None else os.path.basename(xml_file).replace('.xml', '.jpg')
    
    # 处理尺寸
    size_node = root.find('size')
    if size_node is not None:
        width = int(size_node.find('width').text)
        height = int(size_node.find('height').text)
    else:
        # 如果 XML 没写尺寸，默认置 0，LabelMe 打开时通常会自动读取图片尺寸
        width = 0
        height = 0

    # 2. 构建 LabelMe 基础结构
    json_dict = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": filename,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    # 3. 提取 Object (检测框)
    for obj in root.findall('object'):
        name_node = obj.find('name')
        if name_node is None:
            continue
        label = name_node.text
        
        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue
        
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # LabelMe 的 rectangle 类型使用两个点：[左上, 右下]
        points = [
            [xmin, ymin],
            [xmax, ymax]
        ]

        shape = {
            "label": label,
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": "rectangle", # XML 转过来的框设为 rectangle
            "flags": {},
            "mask": None
        }
        json_dict['shapes'].append(shape)

    return json_dict

def main():
    print(f"开始扫描目录: {dataset_root}")
    
    # 1. 查找所有 XML 文件
    xml_files = glob.glob(os.path.join(dataset_root, "**", "*.xml"), recursive=True)
    print(f"共发现 {len(xml_files)} 个 XML 文件。")
    
    success_count = 0
    fail_count = 0

    for xml_path in tqdm(xml_files, desc="转换中"):
        try:
            # 解析 XML
            json_content = parse_xml_to_dict(xml_path)
            
            # === 核心修改点：直接保存在当前 XML 的同级目录下 ===
            current_dir = os.path.dirname(xml_path)
            xml_filename = os.path.basename(xml_path)
            # 替换后缀 .xml -> .json
            json_filename = os.path.splitext(xml_filename)[0] + ".json"
            output_path = os.path.join(current_dir, json_filename)

            # 保存 JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, indent=2, ensure_ascii=False)
            
            success_count += 1

        except Exception as e:
            print(f"\n[Error] 处理文件失败: {xml_path}")
            print(f"原因: {e}")
            fail_count += 1

    print("\n" + "="*30)
    print(f"处理完成！")
    print(f"成功转换: {success_count} 个")
    print(f"失败: {fail_count} 个")
    print(f"JSON 文件已保存在原 XML 文件旁边。")
    print("="*30)

if __name__ == "__main__":
    main()