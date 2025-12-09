#!/bin/bash
set -e  # Exit on error
# Local test script for terratorch model

# # Set PROJ paths
# export PROJ_LIB="/home/ejalilva/.conda/envs/terratorch-tune/lib/python3.12/site-packages/rasterio/proj_data"
# export PROJ_DATA="$PROJ_LIB"
# export PYPROJ_DATADIR="/home/ejalilva/.conda/envs/terratorch-tune/lib/python3.12/site-packages/pyproj/proj_dir/share/proj"

# # Disable albumentations update check
# export NO_ALBUMENTATIONS_UPDATE='1'

# echo "Environment setup:"
# echo "  PROJ_LIB: $PROJ_LIB"
# echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Activate conda environment
ml load anaconda
conda activate terratorch-tune


# Launch parallel trials on available GPUs
CUDA_VISIBLE_DEVICES=0 python optuna_hyperparameter_tuning.py &
CUDA_VISIBLE_DEVICES=1 python optuna_hyperparameter_tuning.py &
# Add more GPUs as needed:
# CUDA_VISIBLE_DEVICES=2 python optuna_hyperparameter_tuning.py &
# CUDA_VISIBLE_DEVICES=3 python optuna_hyperparameter_tuning.py &

# Wait for all trials
wait
echo "All trials complete"