#!/usr/bin/env python
# resume_optuna_search.py

import os
import sys
import numpy as np
import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import TrialState
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Disable albumentations version check
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Fix PROJ paths BEFORE any imports
PROJ_PATH = '/home/ejalilva/.conda/envs/terratorch-tune/lib/python3.12/site-packages/rasterio/proj_data'
os.environ['PROJ_LIB'] = PROJ_PATH
os.environ['PROJ_DATA'] = PROJ_PATH
PYPROJ_PATH = '/home/ejalilva/.conda/envs/terratorch-tune/lib/python3.12/site-packages/pyproj/proj_dir/share/proj'
os.environ['PYPROJ_DATADIR'] = PYPROJ_PATH

torch.set_float32_matmul_precision('medium')  # Enable Tensor Cores

# Get absolute path of the local package
local_package_path = os.path.abspath(os.path.join(os.getcwd(), '../..', 'terratorch'))

# Add the path to system path if it's not already there
if local_package_path not in sys.path:
    sys.path.insert(0, local_package_path)

# If you had previously imported terratorch, reload it
import importlib
if 'terratorch' in sys.modules:
    importlib.reload(sys.modules['terratorch'])

import terratorch
from terratorch.datamodules import MultiTemporalCropClassificationDataModule
from terratorch.datasets import MultiTemporalCropClassification
from terratorch.tasks import SemanticSegmentationTask
from terratorch.datasets.transforms import FlattenTemporalIntoChannels, UnflattenTemporalFromChannels

import albumentations as A
from albumentations.pytorch import ToTensorV2

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Import your custom callbacks
from loss_callback import LossTrackerCallback
from confusionMatrix_callback_withVal import ConfusionMatrixCallback

# Configuration - KEEP ALL the ORIGINAL SETTINGS
DATASET_PATH = '/discover/nobackup/ejalilva/data/prithvi/datasets--ibm-nasa-geospatial--multi-temporal-irrigation-classificaction/snapshots/04b439f179e52a7b144f69676210eecd30c39cfc/'
STUDY_NAME = "prithvi_tuning_20251112_231341"  # Use the SAME study name as before
N_TRIALS = 50  # Total desired trials
MAX_EPOCHS_TUNING = 20  # Same as your original
FINAL_EPOCHS = 60
N_GPUS = 1

# Check GPU availability
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    print(f"Using {N_GPUS} GPU(s) per trial")

def create_datamodule(batch_size=16, num_workers=23):
    """Create the data module with specified batch size."""
    transforms = [
        terratorch.datasets.transforms.FlattenTemporalIntoChannels(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(),
        terratorch.datasets.transforms.UnflattenTemporalFromChannels(n_timesteps=3),
    ]
    
    return MultiTemporalCropClassificationDataModule(
        batch_size=batch_size,
        data_root=DATASET_PATH,
        train_transform=transforms,
        val_transform=transforms,
        test_transform=transforms,
        reduce_zero_label=False,
        expand_temporal_dimension=True,
        use_metadata=True,
        num_workers=num_workers
    )

def objective(trial):
    """YOUR EXACT SAME OBJECTIVE FUNCTION FROM THE ORIGINAL SCRIPT"""
    
    # Hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.01, 0.5, log=True)
    head_dropout = trial.suggest_float('head_dropout', 0.0, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    decoder_channels = trial.suggest_categorical('decoder_channels', [128, 256, 512])
    
    # Additional hyperparameters you might want to tune
    optimizer_type = trial.suggest_categorical('optimizer', ['AdamW', 'SGD', 'Adam'])
    use_scheduler = trial.suggest_categorical('use_scheduler', [True, False])
    
    # Class weights
    use_class_weights = trial.suggest_categorical('use_class_weights', [True, False])
    if use_class_weights:
        weight_scale = trial.suggest_float('weight_scale', 0.5, 2.0)
        class_weights = [2.28 * weight_scale, 1.02 * weight_scale, 
                        1.37 * weight_scale, 0.54 * weight_scale]
    else:
        class_weights = None
    
    # Architecture choices
    freeze_backbone = trial.suggest_categorical('freeze_backbone', [True, False])
    backbone_model = trial.suggest_categorical('backbone', 
        ['prithvi_eo_v2_300_tl', 'prithvi_eo_v2_300'])
    
    print(f"\nTrial {trial.number} hyperparameters:")
    print(f"  lr: {lr:.6f}")
    print(f"  weight_decay: {weight_decay:.4f}")
    print(f"  head_dropout: {head_dropout:.4f}")
    print(f"  batch_size: {batch_size}")
    
    # Set seed for reproducibility
    pl.seed_everything(42 + trial.number)
    
    # Create run name
    run_name = f"trial_{trial.number:03d}"
    
    # Setup directories
    output_dir = os.path.join("optuna_outputs", STUDY_NAME, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs",
        version=f"trial_{trial.number}"
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        monitor="val/Multiclass_Jaccard_Index",
        mode="max",
        save_top_k=1,
        filename=f"trial_{trial.number}" + "-{epoch:02d}-{val_loss:.4f}"
    )
    
    pruning_callback = PyTorchLightningPruningCallback(
        trial, monitor="val/Multiclass_Jaccard_Index"
    )
    
    early_stopping = EarlyStopping(
        monitor="val/Multiclass_Jaccard_Index",
        mode="max",
        patience=5,
        min_delta=0.001
    )
    
    loss_tracker = LossTrackerCallback()
    
    callbacks = [checkpoint_callback, pruning_callback, early_stopping, loss_tracker]
    
    try:
        # Create data module
        data_module = create_datamodule(batch_size=batch_size, num_workers=11)
        
        # Trainer
        trainer = pl.Trainer(
            accelerator="auto",
            devices=N_GPUS,
            precision="bf16-mixed",
            logger=logger,
            max_epochs=MAX_EPOCHS_TUNING,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            enable_checkpointing=True,
            callbacks=callbacks,
            default_root_dir=output_dir,
            strategy='ddp_find_unused_parameters_true' if N_GPUS > 1 else 'auto',
            enable_model_summary=False,
            enable_progress_bar=True
        )
        
        # Model arguments
        model_args = {
            "decoder": "UperNetDecoder",
            "backbone_pretrained": True,
            "backbone": backbone_model,
            "backbone_in_channels": 6,
            "backbone_features_only": True,
            "backbone_coords_encoding": ["time", "location"],
            "rescale": True,
            "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            "backbone_num_frames": 3,
            "num_classes": 4,
            "head_dropout": head_dropout,
            "decoder_channels": decoder_channels,
            "decoder_scale_modules": True,
            "necks": [
                {
                    "name": "SelectIndices",
                    "indices": [5, 11, 17, 23]
                },
                {
                    "name": "ReshapeTokensToImage",
                    "effective_time_dim": 3
                }
            ]
        }
        
        # Create optimizer parameters
        if optimizer_type == 'SGD':
            optimizer_hparams = {
                "weight_decay": weight_decay,
                "momentum": 0.9
            }
        else:
            optimizer_hparams = {"weight_decay": weight_decay}
        
        # Create model
        model = SemanticSegmentationTask(
            model_args=model_args,
            plot_on_val=False,
            class_weights=class_weights,
            loss="ce",
            lr=lr,
            optimizer=optimizer_type,
            optimizer_hparams=optimizer_hparams,
            ignore_index=-1,
            freeze_backbone=freeze_backbone,
            freeze_decoder=False,
            model_factory="EncoderDecoderFactory",
        )
        
        # Train
        trainer.fit(model, datamodule=data_module)
        
        # Get the best validation score
        val_score = trainer.callback_metrics.get("val/Multiclass_Jaccard_Index", 0.0)
        if torch.is_tensor(val_score):
            val_score = val_score.item()
        
        # Save trial results
        trial_results = {
            "trial_number": trial.number,
            "val_jaccard": val_score,
            "hyperparameters": trial.params,
            "epochs_trained": trainer.current_epoch + 1
        }
        
        with open(os.path.join(output_dir, "trial_results.json"), "w") as f:
            json.dump(trial_results, f, indent=2)
        
        print(f"Trial {trial.number} completed. Val Jaccard: {val_score:.4f}")
        
        return val_score
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        return 0.0

def resume_or_create_study():
    """Resume existing study or create new one"""
    
    db_path = "optuna_study.db"
    
    if os.path.exists(db_path):
        print(f"Found existing study database at {db_path}")
        
        # Load existing study
        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=f"sqlite:///{db_path}"
        )
        
        # Print current status
        n_trials = len(study.trials)
        n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        
        print(f"Resuming study '{STUDY_NAME}':")
        print(f"  Total trials so far: {n_trials}")
        print(f"  Completed: {n_complete}")
        print(f"  Pruned: {n_pruned}")
        
        if n_complete > 0:
            print(f"  Best value: {study.best_value:.4f}")
            print(f"  Best trial: #{study.best_trial.number}")
        
        remaining = N_TRIALS - n_trials
        if remaining <= 0:
            print(f"Study already has {n_trials} trials (target was {N_TRIALS})")
            print("Study is complete!")
            return study, 0
        
        print(f"Will run {remaining} more trials to reach {N_TRIALS} total")
        
    else:
        print(f"No existing study found. Creating new study '{STUDY_NAME}'")
        
        # Create new study
        study = optuna.create_study(
            study_name=STUDY_NAME,
            direction='maximize',
            storage=f"sqlite:///{db_path}",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1
            ),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        remaining = N_TRIALS
        print(f"Will run {remaining} trials")
    
    return study, remaining

def main():
    """Main function to run or resume optimization"""
    
    print("="*60)
    print("Optuna Hyperparameter Optimization")
    print("="*60)
    
    # Resume or create study
    study, remaining_trials = resume_or_create_study()
    
    if remaining_trials > 0:
        # Run optimization
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=11.5*3600,  # 11.5 hours safety margin
            catch=(Exception,),
            show_progress_bar=True
        )
    
    # Print final results
    print("\n" + "="*60)
    print("Optimization completed!")
    print("="*60)
    
    print(f"Total trials: {len(study.trials)}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Save final results
    output_dir = os.path.join("optuna_outputs", STUDY_NAME)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/final_results.json")

if __name__ == "__main__":
    main()