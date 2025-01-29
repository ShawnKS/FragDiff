#!/bin/bash

# 定义要遍历的 nostd 和 std 模型目录
model_dirs=(
    # "/path/tocheckpoints/drugs_none_bs256_std_aug_rigid/"
    # "/path/tocheckpoints/drugs_none_bs256_nostd_aug_rigid/"
    "/path/tocheckpoints/drugs_none_bs32_std_rigid_500/"
    "/path/tocheckpoints/drugs_none_bs256_nostd_rigid/"
    # drugs_none_bs256_nostd_aug_rigid
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
        # 输入文件路径
        confs_file="${model_dir}drugs_default/drugs_${steps}steps.pkl"

        # 运行命令
        python evaluate_confs.py \
            --confs $confs_file \
            --test_csv /path/toDRUGS/test_smiles.csv \
            --true_mols /path/toDRUGS/test_mols.pkl \
            --n_workers 500
    done
done


# python evaluate_confs.py --confs /path/tocheckpoints/drugs_none_bs256_nostd_aug_rigid/drugs_default/drugs_20steps.pkl --test_csv /path/toDRUGS/test_smiles.csv --true_mols /path/toDRUGS/test_mols.pkl --n_workers 10;

# python evaluate_confs.py --confs /path/tocheckpoints/drugs_none_bs256_nostd_aug_rigid/drugs_default/drugs_10steps.pkl --test_csv /path/toDRUGS/test_smiles.csv --true_mols /path/toDRUGS/test_mols.pkl --n_workers 10;

# python evaluate_confs.py --confs /path/tocheckpoints/drugs_none_bs256_nostd_aug_rigid/drugs_default/drugs_5steps.pkl --test_csv /path/toDRUGS/test_smiles.csv --true_mols /path/toDRUGS/test_mols.pkl --n_workers 10;


# /path/tocheckpoints/drugs_none_bs256_nostd_aug_rigid/drugs_default

# python evaluate_confs.py --confs /path/tocheckpoints/drugs_frag_all_bs128/drugs_default/drugs_20steps.pkl --test_csv /path/toDRUGS/test_smiles.csv --true_mols /path/toDRUGS/test_mols.pkl --n_workers 10;

# python evaluate_confs.py --confs /path/tocheckpoints/drugs_frag_all_bs128/drugs_default/drugs_10steps.pkl --test_csv /path/toDRUGS/test_smiles.csv --true_mols /path/toDRUGS/test_mols.pkl --n_workers 10;

# python evaluate_confs.py --confs /path/tocheckpoints/drugs_frag_conn_bs128/drugs_default/drugs_20steps.pkl --test_csv /path/toDRUGS/test_smiles.csv --true_mols /path/toDRUGS/test_mols.pkl --n_workers 10

# CUDA_VISIBLE_DEVICES=7 python generate_confs.py --test_csv /path/toDRUGS/test_smiles.csv --inference_steps 20 --model_dir /path/tocheckpoints/drugs_frag_all_bs128/ --out /path/tocheckpoints/drugs_frag_all_bs128/drugs_default/drugs_20steps.pkl --tqdm --batch_size 128 --no_energy --dec frag_all ; 
#  CUDA_VISIBLE_DEVICES=7 python generate_confs.py --test_csv /path/toDRUGS/test_smiles.csv --inference_steps 10 --model_dir /path/tocheckpoints/drugs_frag_all_bs128/ --out /path/tocheckpoints/drugs_frag_all_bs128/drugs_default/drugs_10steps.pkl --tqdm --batch_size 128 --no_energy --dec frag_all  ;

# CUDA_VISIBLE_DEVICES=7 python generate_confs.py --test_csv /path/toDRUGS/test_smiles.csv --inference_steps 20 --model_dir /path/tocheckpoints/drugs_frag_conn_bs128/ --out /path/tocheckpoints/drugs_frag_conn_bs128/drugs_default/drugs_20steps.pkl --tqdm --batch_size 128 --no_energy --dec frag_conn ; 
#  CUDA_VISIBLE_DEVICES=7 python generate_confs.py --test_csv /path/toDRUGS/test_smiles.csv --inference_steps 10 --model_dir /path/tocheckpoints/drugs_frag_conn_bs128/ --out /path/tocheckpoints/drugs_frag_conn_bs128/drugs_default/drugs_10steps.pkl --tqdm --batch_size 128 --no_energy --dec frag_conn  ;

# CUDA_VISIBLE_DEVICES=7 python generate_confs.py --test_csv /path/toDRUGS/test_smiles.csv --inference_steps 20 --model_dir /path/tocheckpoints/drugs_br_bs128/ --out /path/tocheckpoints/drugs_br_bs128/drugs_default/drugs_20steps.pkl --tqdm --batch_size 128 --no_energy --dec br ; 
#  CUDA_VISIBLE_DEVICES=7 python generate_confs.py --test_csv /path/toDRUGS/test_smiles.csv --inference_steps 10 --model_dir /path/tocheckpoints/drugs_br_bs128/ --out /path/tocheckpoints/drugs_br_bs128/drugs_default/drugs_10steps.pkl --tqdm --batch_size 128 --no_energy --dec br  ;