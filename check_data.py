# check_data.py
import json
import os
import sys
import torch
from datasets.odvg import ODVGDataset, make_coco_transforms

# 模拟 args 参数类
class Args:
    def __init__(self):
        self.data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        self.data_aug_max_size = 1333
        self.data_aug_scales2_resize = [400, 500, 600]
        self.data_aug_scales2_crop = [384, 600]
        self.fix_size = False
        self.strong_aug = False
        self.max_labels = 80

def check_dataset():
    #Paths - 请根据你的实际路径修改这里
    root_dir = "/opt/data/private/xjx/RailMind/高速铁路无人机图像/FilteredLabeled"
    anno_file = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/train_split.jsonl"
    label_map_file = "/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino/label_map.json"

    print(f"正在检查数据集...")
    print(f"Root: {root_dir}")
    
    # 1. 检查 Label Map
    try:
        with open(label_map_file, 'r') as f:
            label_map = json.load(f)
        print(f"Label Map 加载成功，共 {len(label_map)} 个类别")
    except Exception as e:
        print(f"错误：无法读取 label_map.json: {e}")
        return

    # 2. 实例化数据集 (这会触发 __init__)
    try:
        args = Args()
        dataset = ODVGDataset(
            root=root_dir,
            anno=anno_file,
            label_map_anno=label_map_file,
            max_labels=80,
            transforms=make_coco_transforms('train', False, False, args)
        )
        print(f"Dataset 初始化成功，共 {len(dataset)} 张图片")
    except Exception as e:
        print(f"错误：Dataset 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 遍历检查每张图片 (这会触发 __getitem__)
    print("开始遍历读取图片，检查路径和标签...")
    from tqdm import tqdm
    
    # 我们只检查前 500 张，或者你可以检查全部
    for i in tqdm(range(len(dataset))):
        try:
            # 获取原始元数据，检查路径是否存在
            meta = dataset.metas[i]
            rel_path = meta["filename"]
            abs_path = os.path.join(root_dir, rel_path)
            
            if not os.path.exists(abs_path):
                print(f"\n[致命错误] 找不到图片文件！")
                print(f"JSON中路径: {rel_path}")
                print(f"系统绝对路径: {abs_path}")
                print("提示: 这可能是由于路径中的中文字符导致的编码问题。")
                break
            
            # 尝试读取数据
            img, target = dataset[i]
            
            # 检查 Label 是否在 Map 中
            # 这里的 target['labels'] 已经是转换后的索引了，如果 dataset[i] 没报错，说明 Label 映射是成功的
            
        except KeyError as e:
            print(f"\n[数据错误] 图片 {meta.get('filename')} 中的 Label ID 在 label_map.json 中找不到！")
            print(f"报错信息: {e}")
            print(f"对应的数据内容: {meta['detection']}")
            break
        except Exception as e:
            print(f"\n[未知错误] 在处理第 {i} 张图片时崩溃")
            print(f"文件名: {dataset.metas[i].get('filename')}")
            import traceback
            traceback.print_exc()
            break

    print("\n检查结束。如果上面没有红色报错，说明数据读取层是没问题的。")

if __name__ == "__main__":
    check_dataset()