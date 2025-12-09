#!/bin/bash
#SBATCH --job-name=rf-bench
#SBATCH -o rf_bench.o%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --account=s1189

module load anaconda
conda activate terratorch-tune

python3 rf_benchmark_simple.py
