#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --qos=grind
#SBATCH --partition=a100-grind
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=baseline/%x_%j.out
#SBATCH --error=baseline/%x_%j.err
mkdir -p baseline
source ~/miniconda3/bin/activate baseline
export TORCH_HOME=$HOME/torch_cache
export PYTHONUNBUFFERED=1
python baseline.py \
    --manifest sub_manifest.csv \
    --workers $SLURM_CPUS_PER_TASK \
    "$@"
