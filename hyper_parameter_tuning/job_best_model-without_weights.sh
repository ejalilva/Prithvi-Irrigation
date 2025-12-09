#!/usr/local/bin/bash
#SBATCH --job-name=prithvi-best
#SBATCH -o prithvi_best.o%j
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=122G
#SBATCH --time=11:59:00
#SBATCH --account=s1189
#SBATCH --constraint=rome

module load anaconda
conda activate terratorch-tune

echo "Starting training at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

python3 train_best_model_v5_noWeights.py

echo "Finished at $(date)"
