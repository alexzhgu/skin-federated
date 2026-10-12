#!/bin/bash
#SBATCH --job-name=fed
#SBATCH --qos=grind
#SBATCH --partition=a100-grind
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs
source ~/miniconda3/bin/activate baseline

export TORCH_HOME=$HOME/torch_cache

ALGO=fedavg
PARTITION=iid
ALPHA=0.5
MU=0.1

python federated.py \
  --manifest sample_manifest_rangpur.csv \
  --path-col filepath \
  --label-col class \
  --algo $ALGO \
  --partition $PARTITION \
  --alpha $ALPHA \
  --mu $MU \
  --workers 4 \
  --out-dir fed_runs
