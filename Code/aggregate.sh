#!/bin/bash
#SBATCH --job-name=aggregate
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
mkdir -p logs
source ~/miniconda3/bin/activate baseline
python aggregate.py --fed-dir fed_runs --baseline baseline/baseline_553175.out "$@"
