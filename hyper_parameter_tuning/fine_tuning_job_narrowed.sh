#!/usr/local/bin/bash
#SBATCH --job-name=prithvi-tune
#SBATCH -o irrigation_prithvi_gpu.o%j
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1                 # Single task - PyTorch Lightning handles distribution
#SBATCH --cpus-per-task=24         # 12 per GPU × 2 GPUs
#SBATCH --mem-per-gpu=122G         # Use full default allocation
#SBATCH --time=11:59:00
#SBATCH --account=s1189
#SBATCH --constraint=rome

ulimit -s unlimited

module load anaconda
conda activate terratorch-tune

# Print job info
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# Run without --exclusive flag (was causing memory conflicts)
python3 optuna_hyperparameter_tuning_omm_fixed.py

echo "Job finished at $(date)"
