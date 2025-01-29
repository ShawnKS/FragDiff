## Enhancing Molecular Conformer Generation via Fragment-Augmented Diffusion Pretraining

Anonymous code repository for our TMLR submission `Enhancing Molecular Conformer Generation via Fragment-Augmented Diffusion Pretraining`.

## Setting up Conda Environment

Create a new [Conda](https://docs.anaconda.com/anaconda/install/index.html) environment using `environment.yml`:

```sh
conda env create -f environment.yml
conda activate fragdiff  # Updated environment name
```

Install `e3nn`: `pip install e3nn`.

## Datasets Preparation
To train, generate and evaluate conformers, first download the dataset directory from [this shared Drive](https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7?usp=sharing).
### Data Preprocessing with /data_tools
**We provide improved preprocessing scripts to address memory and version dependency issues in the original implementations of TorDiff and GeoDiff**:

1. **Fragment-Augmented Diffusion Processing**:
   ```sh
   # Step 1: Featurization
   python data_tools/train_data_dump.py

   # Step 2: Fragmentation
   python data_tools/frag_data_dump.py
   ```

2. **GeoDiff Processing**:
   ```sh
   # Step 1: GeoDiff Featurization
   python data_tools/geodiff_data_dump.py 

   # Step 2: GeoDiff Fragmentation
   python data_tools/geodiff_frag_dump.py 
   ```

### Computational-Aided Data Augmentation
Example conformer matching command (preserved from original):
```sh
python standardize_confs.py \
  --out_dir data/DRUGS/standardized_pickles \
  --root data/DRUGS/drugs/ \
  --confs_per_mol 30 \
  --worker_id 0 \
  --jobs_per_worker 1000 &
```

## Training Workflows
### Fragment-Augmented Diffusion Training
After setting data paths in `refactor_train.py`:
```sh
# New training
python refactor_train.py --log_dir workdir/new_training

# Resume training
python refactor_train.py --log_dir workdir/resumed_training \
  --restart_dir workdir/previous_training
```

### GeoDiff Training
Configure training via YAML files:
```sh
# New training
python train.py ./config/xx.yml

# Resume training (with iteration 15000)
python train.py ./config/xx.yml -- --resume --resume_iter 15000
```

## Conformers Generation and Evaluation
Generation command (TorDiff):
```sh
python generate_confs.py \
  --test_csv DRUGS/test_smiles.csv \
  --inference_steps 20 \
  --model_dir workdir/drugs_default \
  --out conformers_20steps.pkl \
  --tqdm \
  --batch_size 128 \
  --no_energy
```

Evaluation command (TorDiff):
```sh
python evaluate_confs.py \
  --confs workdir/drugs_default/drugs_steps20.pkl \
  --test_csv data/DRUGS/test_smiles.csv \
  --true_mols data/DRUGS/test_mols.pkl \
  --n_workers 10
```

Generation command (GeoDiff):
```sh
python ./Geodiff/test.py .../logs/../checkpoints/xx.pt --start_idx <start_idx> --end_idx <end_idx>
```

Evaluation command (GeoDiff):
```sh
python eval_covmat.py .../logs/../samples_../sample_xx.pkl --start_idx <start_idx> --end_idx <end_idx>
```

## Fragment-Augmented Boltzmann Generator
Updated environment name in commands:
```sh
# Training
python train.py \
  --boltzmann_training \
  --boltzmann_weight \
  --sigma_min 0.1 \
  --temp 250 \
  --adjust_temp \
  --log_dir workdir/boltz_T250 \
  ...  # other parameters preserved

# Testing
python test_boltzmann.py \
  --model_dir workdir/boltz_T250 \
  --temp 250 \
  --model_steps 20 \
  --original_model_dir workdir/drugs_seed_boltz/ \
  --out boltzmann.out
```

---
Full implementation details will be released with the final paper.