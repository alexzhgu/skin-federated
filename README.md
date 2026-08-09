# Privacy-Preserving Federated Learning for Distributed Skin Disease Classification

Honours project (REIT4881), University of Queensland.

Evaluates FedAvg and FedProx for multi-class skin disease classification under
non-IID data, compared against a centralised baseline. A subsampled skin disease
image dataset is partitioned across simulated clients using Dirichlet label
skew, and classification accuracy, convergence, and communication cost are
measured across levels of heterogeneity.

## Dataset

Source: `muhammadabdulsami/massive-skin-disease-balanced-dataset` (Kaggle),
~262k images across 35 classes. Downloaded via `kagglehub`.

Experiments run on a stratified subsample of 30 classes / ~20k images
(`sub_manifest.csv`), which preserves the original train/val/test split per
image. The full dataset is used only for the context baseline.

## Environment

- Python 3.14, PyTorch with CUDA (run on A100 GPU(s))
- `pip install torch torchvision pandas numpy scikit-learn pillow tqdm imagehash matplotlib kagglehub`

## Pipeline

Run in order. Each script reads a manifest CSV with columns `filepath, class,
split` and writes to a per-stage output directory.

### 1. Subsample

Builds the 30-class / 20k-image manifest from the full split.

```
python subsample.py --manifest runs/<baseline_ts>/split_manifest.csv \
    --out sub_manifest.csv --classes 30 --total 20000
```

### 2. Leakage check

Perceptual-hash check for near-duplicate images crossing the train/val/test
split. Reports cross-split collisions (leakage) and cross-class collisions
(label noise).

```
python phash.py --manifest sub_manifest.csv --workers 16
```

### 3. Partition check

Prints per-client class distribution, size ratio, and empty-cell counts for
each Dirichlet alpha, to confirm the intended skew before training.

```
python partition.py sub_manifest.csv
```

### 4. Centralised baseline

ResNet-18 (ImageNet-pretrained), trained on the pooled training set. The
best-validation checkpoint is evaluated on the held-out test set. This is the
upper-bound reference for the federated runs.

```
sbatch baseline.sh                       # uses sub_manifest.csv
```

Output: `runs/<timestamp>/` with `config.json`, `train_log.csv`, `best.pt`,
`test_results.json`.

### 5. Federated training

FedAvg and FedProx over simulated clients. The training split is partitioned
per run (IID or Dirichlet); val/test are held central and identical across
runs. Reads the same manifest as the baseline.

```
sbatch --job-name=<tag> federated.sh --algo fedavg  --partition iid --seed 42
sbatch --job-name=<tag> federated.sh --algo fedavg  --partition dirichlet --alpha 0.1 --seed 42
sbatch --job-name=<tag> federated.sh --algo fedprox --partition dirichlet --alpha 0.1 --mu 0.001 --seed 42
```

Key arguments: `--algo {fedavg,fedprox}`, `--partition {iid,dirichlet}`,
`--alpha`, `--mu`, `--clients`, `--frac`, `--rounds`, `--local-epochs`,
`--seed`, `--out-dir`.

Output per run (in `--out-dir`): `<tag>_report.txt`, `<tag>_log.csv`,
`<tag>_model.pt`, where the tag encodes algo, partition, alpha, mu, client
count, and seed.

### 6. Aggregate

Reads all runs in a directory, computes mean/std per cell over seeds, runs the
paired FedAvg-vs-FedProx comparison, and writes the figures.

```
python aggregate.py --fed-dir fed_runs --baseline runs/<baseline_ts> --out-dir results
```

Output (in `results/`): `summary.csv`, `grid.csv`, `paired.csv`, and figures
for accuracy vs alpha, validation curves, accuracy vs communication, and (if
present) accuracy vs client count.

## Experimental grid

Main sweep, 3 seeds (42/43/44) per cell:

- Partitions: IID, Dirichlet alpha in {1.0, 0.5, 0.1}
- Algorithms: FedAvg; FedProx (mu selected on validation at alpha=0.1)
- Fixed: 10 clients, frac 0.5, 100 rounds, 5 local epochs, ResNet-18

FedProx mu was tuned at alpha=0.1 over {0.001, 0.01, 0.1, 1.0}; mu=0.001 was
selected. A client-count sweep (5, 10, 50 clients at alpha=0.1) tests how
heterogeneity severity scales with per-client data volume.

The seed controls partition draw, client selection, initialisation, and
shuffle. FedAvg and FedProx are
compared at matched seeds so the difference isolates the algorithm.

## Repository layout

```
subsample.py     build the subsampled manifest
phash.py         perceptual-hash leakage check
partition.py     Dirichlet / IID partition, standalone summary
baseline.py      centralised ResNet-18 baseline
federated.py     FedAvg / FedProx simulation
aggregate.py     tables and figures
*.sh             SLURM submission scripts
sub_manifest.csv the 30-class / 20k-image split used for all experiments
```

## Notes

- Run outputs (`runs/`, `fed_runs/`, `less_nodes/`, checkpoints, logs) and the
  dataset are not tracked. `sub_manifest.csv` is tracked as it defines the
  experiments.
