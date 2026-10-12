#!/bin/bash
#SBATCH --job-name=federated
#SBATCH --qos=grind
#SBATCH --partition=a100-grind
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=a100-1,a100-2,a100-3
#SBATCH --cpus-per-task=4
#SBATCH --output=federated/%x_%j.out
#SBATCH --error=federated/%x_%j.err
mkdir -p federated
source ~/miniconda3/bin/activate baseline
export TORCH_HOME=$HOME/torch_cache
python federated.py \
    --manifest sub_manifest.csv \
    --path-col filepath --label-col class \
    --workers $SLURM_CPUS_PER_TASK \
    "$@"
