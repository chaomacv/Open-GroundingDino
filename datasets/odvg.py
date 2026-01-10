from torchvision.datasets.vision import VisionDataset
import os.path
from typing import Callable, Optional
import json
from PIL import Image
import torch
import random
import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import datasets.transforms as T

class ODVGDataset(VisionDataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        anno (string): Path to json annotation file.
        label_map_anno (string):  Path to json label mapping file. Only for Object Detection
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        anno: str,
        label_map_anno: str = None,
        max_labels: int = 80,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        only_train_positives: bool = False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.dataset_mode = "OD" if label_map_anno else "VG"
        self.max_labels = max_labels
        self.only_train_positives = only_train_positives
        
        if self.dataset_mode == "OD":
            self.load_label_map(label_map_anno)
        self._load_metas(anno)
        self.get_dataset_info()

    def load_label_map(self, label_map_anno):
        with open(label_map_anno, 'r') as file:
            raw_map = json.load(file)
        
        # [核心修复] 自动检测并翻转 Label Map
        # 目标格式: {"0": "insulator", "11": "fastener", ...}
        # 如果读入的是 {"insulator": 0, ...}，则需要翻转
        first_val = list(raw_map.values())[0]
        if isinstance(first_val, int):
            # 翻转字典: value(int) -> key(str ID), key(str Name) -> value(str Name)
            self.label_map = {str(v): k for k, v in raw_map.items()}
        else:
            # 假设已经是 ID->Name 格式
            self.label_map = raw_map
            
        self.label_index = set(self.label_map.keys())

    def _load_metas(self, anno):
        with  open(anno, 'r')as f:
            self.metas = [json.loads(line) for line in f]

    def get_dataset_info(self):
        print(f"  == total images: {len(self)}")
        if self.dataset_mode == "OD":
            print(f"  == total labels: {len(self.label_map)}")
            print(f"  == Training Mode: {'Only Positives (No Negatives)' if self.only_train_positives else 'Full Labels (With Negatives)'}")

    def __getitem__(self, index: int):
        meta = self.metas[index]
        rel_path = meta["filename"]
        abs_path = os.path.join(self.root, rel_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{abs_path} not found.")
        image = Image.open(abs_path).convert('RGB')
        w, h = image.size
        
        if self.dataset_mode == "OD":
            anno = meta["detection"]
            instances = [obj for obj in anno["instances"]]
            boxes = [obj["bbox"] for obj in instances]
            
            # 获取正样本 ID (字符串格式)
            ori_classes = [str(obj["label"]) for obj in instances]
            pos_labels = set(ori_classes)
            
            # 采样逻辑
            if self.only_train_positives:
                vg_labels = list(pos_labels)
                random.shuffle(vg_labels)
            else:
                neg_labels = list(self.label_index.difference(pos_labels))
                num_neg_to_sample = self.max_labels - len(pos_labels)
                if num_neg_to_sample > 0:
                    if len(neg_labels) > num_neg_to_sample:
                        neg_labels = random.sample(neg_labels, num_neg_to_sample)
                else:
                    neg_labels = []
                vg_labels = list(pos_labels) + neg_labels
                random.shuffle(vg_labels)

            # 构建 Prompt
            # 此时 self.label_map["11"] 应该能正确返回 "fastener"
            caption_list = [self.label_map[lb] for lb in vg_labels]
            
            caption_dict = {item: index for index, item in enumerate(caption_list)}

            if len(caption_list) > 0:
                caption = ' . '.join(caption_list) + ' .'
            else:
                caption = "" 

            classes = [caption_dict[self.label_map[str(obj["label"])]] for obj in instances]
            
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            classes = torch.tensor(classes, dtype=torch.int64)
            
        elif self.dataset_mode == "VG":
            anno = meta["grounding"]
            instances = [obj for obj in anno["regions"]]
            boxes = [obj["bbox"] for obj in instances]
            caption_list = [obj["phrase"] for obj in instances]
            c = list(zip(boxes, caption_list))
            random.shuffle(c)
            boxes[:], caption_list[:] = zip(*c)
            uni_caption_list  = list(set(caption_list))
            label_map = {}
            for idx in range(len(uni_caption_list)):
                label_map[uni_caption_list[idx]] = idx
            classes = [label_map[cap] for cap in caption_list]
            caption = ' . '.join(uni_caption_list) + ' .'
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            classes = torch.tensor(classes, dtype=torch.int64)
            caption_list = uni_caption_list
            
        target = {}
        target["orig_size"] = torch.as_tensor([int(h), int(w)]) 
        target["image_id"] = torch.tensor([index])
        
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["cap_list"] = caption_list
        target["caption"] = caption
        target["boxes"] = boxes
        target["labels"] = classes

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    

    def __len__(self) -> int:
        return len(self.metas)


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
    
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i*data_aug_scale_overlap) for i in scales]
        max_size = int(max_size*data_aug_scale_overlap)
        scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ])

        if strong_aug:
            import datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),
                normalize,
            ])
        
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'eval_debug', 'train_reg', 'test']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_odvg(image_set, args, datasetinfo):
    img_folder = datasetinfo["root"]
    ann_file = datasetinfo["anno"]
    label_map = datasetinfo["label_map"] if "label_map" in datasetinfo else None
    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    
    only_train_positives = getattr(args, 'only_train_positives', False)
    
    print(img_folder, ann_file, label_map)
    print(f"Dataset build: only_train_positives={only_train_positives}")
    
    dataset = ODVGDataset(
        img_folder, 
        ann_file, 
        label_map, 
        max_labels=args.max_labels,
        transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args),
        only_train_positives=only_train_positives 
    )
    return dataset


if __name__=="__main__":
    pass