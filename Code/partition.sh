#!/bin/bash
#SBATCH --job-name=partition
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
mkdir -p logs
source ~/miniconda3/bin/activate baseline
python partition.py runs/20260707_163655/split_manifest.csv
