#!/bin/bash

GPU_ID=2

model_dirs=(
    "path/to/checkpoints/drugs_none_bs32_std_aug_rigid/"
)

inference_steps_list=(20 10 5)

for model_dir in "${model_dirs[@]}"; do
    if [[ "$model_dir" == *"nostd"* ]]; then
        model_type="nostd"
    else
        model_type="std"
    fi

    for steps in "${inference_steps_list[@]}"; do
        output_file="${model_dir}XL_default/XL_${steps}steps.pkl"

        CUDA_VISIBLE_DEVICES=$GPU_ID python generate_confs.py \
            --test_csv path/to/XL/test_smiles.csv \
            --inference_steps $steps \
            --model_dir $model_dir \
            --out $output_file \
            --tqdm \
            --batch_size 128 \
            --no_energy \
            --dec none
    done
done