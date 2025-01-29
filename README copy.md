# Fragment-Augmented Diffusion for Molecular Conformation Generation

Anonymous code repository for our ICLR submission `Fragment-Augmented Diffusion for Molecular Conformation Generation`.

## Setting up Conda Environment

Create a new [Conda](https://docs.anaconda.com/anaconda/install/index.html) environment using `environment.yml`.

```sh
conda env create -f environment.yml
conda activate fadiff
```

Install `e3nn` : `pip install e3nn`.

## Datasets Preparation
To train, generate and evaluate conformers, first download the dataset directory from [this shared Drive](https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7?usp=sharing).

### Computational-Aided Data Augmentation via Conformer Matching

Use conformer matching using `standardize_confs.py` to create computational-aided augmentation. Here's an example command:

```sh
python standardize_confs.py \
  --out_dir data/DRUGS/standardized_pickles \
  --root data/DRUGS/drugs/ \
  --confs_per_mol 30 \
  --worker_id 0 \
  --jobs_per_worker 1000 &
```
<!-- ## Training Model -->
## Training, Generation, and Evaluation
**Note:** While the provided code examples utilize the **DRUGS** dataset, you can easily train the models on other datasets such as **QM9** and **XL** by replacing the dataset-specific parameters and paths accordingly.

To train the model using augmented data with a minimum fragment size of 30 and specific decomposition rules, use the following command. Set the `--dec` parameter to `none` to use all possible fragmentations, `"norecap"` to remove RECAP rules, `"nobrics"` to remove BRICS rules, or `"nobr"` to remove both RECAP and BRICS augmentations. For example, to remove RECAP rules:

```sh
python train.py \
  --data_dir "/path/to/data/DRUGS/drugs/" \
  --split_path "/path/to/data/DRUGS/split.npy" \
  --dataset "drugs" \
  --dec "norecap" \
  --batch_size 32 \
  --n_epochs 250 \
  --in_node_features 74 \
  --rigid \
  --aug \
  --z 30 \
  --num_workers 32 \
  --limit_train_mols 0
```

**Parameters Explained:**  
`--data_dir` specifies the directory containing the dataset, `--split_path` points to the data split file, and `--dataset` names the dataset (e.g., `drugs`). The `--dec` flag sets the decomposition rule (`none` to use all possible fragmentations, `"norecap"` to remove RECAP rules, `"nobrics"` to remove BRICS rules, or `"nobr"` to remove both). `--batch_size` defines the number of samples per batch, `--n_epochs` sets the number of training epochs, and `--in_node_features` indicates the number of input node features. The `--rigid` flag enables rigid molecular fragmentation, `--aug` activates data augmentation, `--z` sets the minimum fragment size allowed, `--num_workers` determines the number of worker threads for data loading, and `--limit_train_mols` limits the number of training molecules (`0` for no limit).

## Conformers Generation and Evaluation

An example of generating GEOM-DRUGS conformers by running:

```sh
python generate_confs.py --test_csv DRUGS/test_smiles.csv --inference_steps 20 --model_dir workdir/drugs_default --out conformers_20steps.pkl --tqdm --batch_size 128 --no_energy
```

This script saves `conformers_20steps.pkl`, a dictionary with the SMILE as the key and the RDKit molecules with generated conformers as the value. By default, it generates `2*num_confs` conformers per row in `smiles.csv`; specify a fixed number with `--confs_per_mol`.

To evaluate the generated conformers, ensure you have `test_smiles.csv` and `test_mols.pkl` downloaded. Then, generate conformers with the trained model:

```sh
python generate_confs.py \
  --test_csv data/DRUGS/test_smiles.csv \
  --inference_steps 20 \
  --model_dir workdir/drugs_default \
  --out workdir/drugs_default/drugs_20steps.pkl \
  --tqdm \
  --batch_size 128 \
  --no_energy
```

Next, evaluate the conformers using:

```sh
python evaluate_confs.py \
  --confs workdir/drugs_default/drugs_steps20.pkl \
  --test_csv data/DRUGS/test_smiles.csv \
  --true_mols data/DRUGS/test_mols.pkl \
  --n_workers 10
```

## Fragment-Augmented Boltzmann Generator

To train and test the fragment-augmented Boltzmann generator at 250K:

1. **Train the Generator**

   ```sh
   python train.py \
     --boltzmann_training \
     --boltzmann_weight \
     --sigma_min 0.1 \
     --temp 250 \
     --adjust_temp \
     --log_dir workdir/boltz_T250 \
     --cache data/cache/boltz10k \
     --split_path data/DRUGS/split_boltz_10k.npy \
     --restart_dir workdir/drugs_seed_boltz/
   ```

2. **Test the Generator**

   ```sh
   python test_boltzmann.py \
     --model_dir workdir/boltz_T250 \
     --temp 250 \
     --model_steps 20 \
     --original_model_dir workdir/drugs_seed_boltz/ \
     --out boltzmann.out
   ```

**Parameters Explained:**  
`--boltzmann_training` enables Boltzmann training mode, `--boltzmann_weight` applies Boltzmann weighting, and `--sigma_min` sets the minimum sigma value for diffusion steps. `--temp` specifies the temperature for the Boltzmann distribution, while `--adjust_temp` allows temperature adjustments during training. `--log_dir` defines the directory to save logs and models, `--cache` points to cached data, and `--split_path` indicates the path to the data split file. `--restart_dir` allows restarting training from a checkpoint. During testing, `--model_steps` sets the number of inference steps, `--original_model_dir` refers to the directory of the original trained model, and `--out` specifies the output file for test results.

---

For further customization and advanced configurations, refer to the `utils/parsing.py` file and the script comments. We will provide a more specific and detailed code logic introduction and further refinements after the public release of the article.
