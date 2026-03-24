# Process Predictor

This is the reference implementation for our paper: "Leveraging the Transformer Hawkes Process for Time Predictions in Predictive Process Monitoring"

---

## Project structure

```
thpppm/
├── main.py                   # entry script
├── test.py                   # test script
├── train.sh                  # train all datasets
├── test.sh                   # test all datasets                     
├── environment.yml           # conda environment
├── data/                     # Datasets
│   └──... csv 
├── configs/
│   └── default.yaml          # Default hyper-parameters
├── preprocess/
│   ├── __init__.py
│   └── dataset.py            # CSV loading, feature engineering, Dataset & DataLoader
├── trainer/
│   ├── __init__.py
│   └── train.py              # train_epoch, eval_epoch, train_model
├── utils/
│   ├── __init__.py
│   └── reproducibility.py    # set_all_seeds
├── transformer/              # model
│   ├── __init__.py
│   ├── model.py
│   ├── Constants.py
│   └── Layers.py
├── Utils.py                  # helpers
├── saved_models/             # checkpoints (generated)
├── results/                  # results (generated)
└── runs/                     # TensorBoard logs (generated)
└── additional_results/       # additional results mentioned in the paper
```

## Quick start

### 1. Install dependencies

```bash
conda env create -f environment.yml
conda activate thpppm
```

### 2. Prepare data

Your data directory should contain the following CSV files for each cross-validation fold (an example of BPI2012A can be found under data/):

```
data/
├── train_<fold>_variation0_<name>.csv
├── test_<fold>_variation0_<name>.csv
└── <name>.csv          # full dataset (used to build activity vocabulary)

Example:
data/
├── train_fold0_variation0_BPI_Challenge_2012_A.csv
├── test_fold0_variation0_BPI_Challenge_2012_A.csv
└── BPI_Challenge_2012_A.csv      
```
You can also follow benchmark paper's repository to generate other datasets: https://gitlab.citius.gal/efren.rama/pmdlcompararator

### 3. Train & Test

```bash
bash train.sh
```

One "screen session" named with the name of dataset is created, where you can attach the session using:
```bash
screen -r <session-name>
```



## TensorBoard

While training (or after):

```bash
tensorboard --logdir=runs
# Open http://localhost:6006
```
