#!/bin/bash

# 定义 GPU ID
GPU_ID=2

# 定义要遍历的 nostd 和 std 模型目录
model_dirs=(
    "path/to/checkpoints/drugs_none_bs32_std_aug_rigid/"
)

# 定义要遍历的 inference_steps 数量
inference_steps_list=(20 10 5)

# 遍历每个模型目录
for model_dir in "${model_dirs[@]}"; do
    # 获取模型类型（std 或 nostd）
    if [[ "$model_dir" == *"nostd"* ]]; then
        model_type="nostd"
    else
        model_type="std"
    fi

    # 遍历每个 inference_steps
    for steps in "${inference_steps_list[@]}"; do
        # 输出文件路径
        output_file="${model_dir}drugs_default/drugs_${steps}steps.pkl"

        # 运行命令
        CUDA_VISIBLE_DEVICES=$GPU_ID python generate_confs.py \
            --test_csv path/to/DRUGS/test_smiles.csv \
            --inference_steps $steps \
            --model_dir $model_dir \
            --out $output_file \
            --tqdm \
            --batch_size 128 \
            --no_energy \
            --dec none
    done
done