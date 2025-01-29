#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
datasets=("drugs")

limit_train_mols_values=(0)
zs=(10)

for limit in "${limit_train_mols_values[@]}"
do
for z in "${zs[@]}"
do
for dataset in "${datasets[@]}"
do
    if [ "$dataset" == "qm9" ]; then
        data_dir="/path/to/data/QM9/qm9/"
        cache="/path/to/data/QM9/cache"
        std_pickles="/path/to/data/QM9/standardized_pickles"
        split_path="/path/to/data/QM9/split.npy"
        in_node_features=44
    elif [ "$dataset" == "drugs" ]; then
        data_dir="/path/to/data/DRUGS/drugs/"
        cache="/path/to/data/DRUGS/cache"
        std_pickles="/path/to/data/DRUGS/standardized_pickles"
        split_path="/path/to/data/DRUGS/split.npy"
        restart_dir="/path/to/data/checkpoints/drugs_none_bs32_std/aug_rigid/30nepochs_500_restart"
        in_node_features=74
    fi
        python train.py --data_dir "$data_dir" --std_pickles "$std_pickles" --split_path "$split_path" --dataset "$dataset" --dec none --batch_size 32 --n_epochs 250 --in_node_features "$in_node_features" --rigid --aug --z "$z" --num_workers 32 --limit_train_mols "$limit" --restart_dir "$restart_dir";
done
done
done

zs=(10)
for limit in "${limit_train_mols_values[@]}"
do
for z in "${zs[@]}"
do
for dataset in "${datasets[@]}"
do
    if [ "$dataset" == "qm9" ]; then
        data_dir="/path/to/data/QM9/qm9/"
        cache="/path/to/data/QM9/cache"
        std_pickles="/path/to/data/QM9/standardized_pickles"
        split_path="/path/to/data/QM9/split.npy"
        in_node_features=44
    elif [ "$dataset" == "drugs" ]; then
        data_dir="/path/to/data/DRUGS/drugs/"
        cache="/path/to/data/DRUGS/cache"
        std_pickles="/path/to/data/DRUGS/standardized_pickles"
        split_path="/path/to/data/DRUGS/split.npy"
        in_node_features=74
    fi

        python train.py --data_dir "$data_dir" --split_path "$split_path" --dataset "$dataset" --dec norecap --batch_size 32 --n_epochs 500 --in_node_features "$in_node_features" --rigid --aug --z "$z" --num_workers 32 --limit_train_mols "$limit" ;
        python train.py --data_dir "$data_dir" --split_path "$split_path" --dataset "$dataset" --dec nobrics --batch_size 32 --n_epochs 500 --in_node_features "$in_node_features" --rigid --aug --z "$z" --num_workers 32 --limit_train_mols "$limit";
        python train.py --data_dir "$data_dir" --split_path "$split_path" --dataset "$dataset" --dec nobr --batch_size 32 --n_epochs 500 --in_node_features "$in_node_features" --rigid --aug --z "$z" --num_workers 32 --limit_train_mols "$limit";
        python train.py --data_dir "$data_dir" --split_path "$split_path" --dataset "$dataset" --dec none --batch_size 32 --n_epochs 500 --in_node_features "$in_node_features" --rigid --aug --z "$z" --num_workers 32 --limit_train_mols "$limit";
done
done
done
