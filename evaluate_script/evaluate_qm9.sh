#!/bin/bash

model_dirs=(
    "/path/tocheckpoints/qm9_none_bs32_std_rigid/"
)

inference_steps_list=(20 10 5)

for model_dir in "${model_dirs[@]}"; do
    if [[ "$model_dir" == *"nostd"* ]]; then
        model_type="nostd"
    else
        model_type="std"
    fi

    for steps in "${inference_steps_list[@]}"; do
        confs_file="${model_dir}qm9_default/qm9_${steps}steps.pkl"

        python evaluate_confs.py \
            --confs $confs_file \
            --test_csv /path/toQM9/test_smiles.csv \
            --true_mols /path/toQM9/test_mols.pkl \
            --n_workers 500
    done
done