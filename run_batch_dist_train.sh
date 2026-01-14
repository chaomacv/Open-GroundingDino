#!/bin/bash

# ================= 配置区域 =================
# 显卡数量
GPU_NUM=4
# 基础路径
ROOT_DIR="/opt/data/private/xjx/RailMind/agent/RailwayCARS/relatedResearch/Open-GroundingDino"
IMG_ROOT="/opt/data/private/xjx/RailMind/高速铁路无人机图像/FilteredLabeled"
CONFIG_FILE="${ROOT_DIR}/config/cfg_odvg.py"
OUTPUT_BASE="${ROOT_DIR}/logs/0113"

# 预训练模型路径 (从你的 train_dist.sh 中提取)
PRETRAIN_MODEL="${ROOT_DIR}/../GroundingDINO/weights/groundingdino_swint_ogc.pth"
TEXT_ENCODER="${ROOT_DIR}/../GroundingDINO/weights/bert-base-uncased"

# 遇到错误立即停止
set -e

# ================= 1. 动态生成数据集 JSON =================
echo "正在生成临时数据集配置文件..."

# [配置文件 A] 标准数据 (给模型1 & 2 用)
cat > ${ROOT_DIR}/temp_dataset_std.json <<EOF
{
  "train": [
    {
      "root": "${IMG_ROOT}",
      "anno": "${ROOT_DIR}/train_split.jsonl",
      "label_map": "${ROOT_DIR}/label_map.json",
      "dataset_mode": "odvg"
    }
  ],
  "val": [
    {
      "root": "${IMG_ROOT}",
      "anno": "${ROOT_DIR}/val_split_coco.json",
      "label_map": "${ROOT_DIR}/label_map.json",
      "dataset_mode": "coco"
    }
  ]
}
EOF

# [配置文件 B] Only数据 (给模型3 & 4 用)
cat > ${ROOT_DIR}/temp_dataset_only.json <<EOF
{
  "train": [
    {
      "root": "${IMG_ROOT}",
      "anno": "${ROOT_DIR}/train_split_only.jsonl",
      "label_map": "${ROOT_DIR}/label_map_only.json",
      "dataset_mode": "odvg"
    }
  ],
  "val": [
    {
      "root": "${IMG_ROOT}",
      "anno": "${ROOT_DIR}/val_split_coco_only.json",
      "label_map": "${ROOT_DIR}/label_map_only.json",
      "dataset_mode": "coco"
    }
  ]
}
EOF

echo "配置文件生成完毕，开始 4卡 分布式训练..."
echo "====================================================="

# 定义一个函数来运行训练，保持代码整洁
run_training() {
    local DATASET_JSON=$1
    local EXP_NAME=$2
    local POS_FLAG=$3  # True 或 False
    
    echo ">>> 开始训练任务: ${EXP_NAME}"
    echo "    数据集: ${DATASET_JSON}"
    echo "    only_train_positives: ${POS_FLAG}"

    # 直接调用 torch.distributed.launch，模仿 train_dist.sh 但注入了 extra options
    # 注意：为了防止端口冲突，这里每次任务可以使用不同的端口，或者依赖系统回收
    # 这里我们添加了 only_train_positives 到 --options 中
    
    python -m torch.distributed.launch --nproc_per_node="${GPU_NUM}" --master_port=29500 main.py \
        --output_dir "${OUTPUT_BASE}/${EXP_NAME}" \
        -c "${CONFIG_FILE}" \
        --datasets "${ROOT_DIR}/${DATASET_JSON}" \
        --pretrain_model_path "${PRETRAIN_MODEL}" \
        --options text_encoder_type="${TEXT_ENCODER}" only_train_positives=${POS_FLAG}

    echo ">>> ${EXP_NAME} 训练完成!"
    echo "-----------------------------------------------------"
    
    # 稍微暂停几秒，让显存完全释放
    sleep 5
}

# ================= 2. 执行四个训练任务 =================

# --- 模型一: 标准数据 + 全标签 (False) ---
run_training "temp_dataset_std.json"  "model1_std_fullneg" "False"

# --- 模型二: 标准数据 + 仅正样本 (True) ---
run_training "temp_dataset_std.json"  "model2_std_posonly" "True"

# --- 模型三: Only数据 + 全标签 (False) ---
run_training "temp_dataset_only.json" "model3_only_fullneg" "False"

# --- 模型四: Only数据 + 仅正样本 (True) ---
run_training "temp_dataset_only.json" "model4_only_posonly" "True"

echo "所有 4 个分布式训练任务已完成！"