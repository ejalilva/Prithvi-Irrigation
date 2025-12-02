#!/usr/bin/env python
# coding: utf-8

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
# Fix PROJ paths BEFORE any imports
PROJ_PATH = '/home/ejalilva/.conda/envs/terratorch-tune/lib/python3.12/site-packages/rasterio/proj_data'
os.environ['PROJ_LIB'] = PROJ_PATH
os.environ['PROJ_DATA'] = PROJ_PATH

PYPROJ_PATH = '/home/ejalilva/.conda/envs/terratorch-tune/lib/python3.12/site-packages/pyproj/proj_dir/share/proj'
os.environ['PYPROJ_DATADIR'] = PYPROJ_PATH

print(f"Set PROJ_LIB to: {os.environ['PROJ_LIB']}")
print(f"Set PYPROJ_DATADIR to: {os.environ['PYPROJ_DATADIR']}")
os.environ['HF_DATASETS_OFFLINE'] = '1'

# Verify the file exists
proj_db = os.path.join(PROJ_PATH, 'proj.db')
if os.path.exists(proj_db):
    print(f"✓ proj.db found at: {proj_db}")
else:
    print(f"✗ proj.db NOT found at: {proj_db}")

# Clean up any conflicting GEOSpyD paths
if 'PATH' in os.environ:
    path_parts = os.environ['PATH'].split(':')
    clean_path = [p for p in path_parts if 'GEOSpyD' not in p]
    os.environ['PATH'] = ':'.join(clean_path)

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

# Configuration
DATASET_PATH = '/discover/nobackup/ejalilva/data/prithvi/datasets--ibm-nasa-geospatial--multi-temporal-irrigation-classificaction/snapshots/04b439f179e52a7b144f69676210eecd30c39cfc/'
STUDY_NAME = f"prithvi_tuning_V4"  # New version - fresh database needed
N_TRIALS = 50  # Number of trials
MAX_EPOCHS_TUNING = 20  # Fewer epochs for tuning
FINAL_EPOCHS = 60  # Full training with best params
N_GPUS = 1  # Single GPU per trial for memory efficiency

# Check GPU availability
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Using {N_GPUS} GPU(s) per trial")


def create_datamodule(batch_size=16, num_workers=4):
    """Create the data module with specified batch size.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of data loading workers (keep low for DDP)
    """
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
        use_metadata=False,
        num_workers=num_workers,
        pin_memory=True,              # Faster GPU transfer
        persistent_workers=True if num_workers > 0 else False,  # Reduce worker spawn overhead
    )


def get_neck_indices(backbone_model):
    """Get the correct neck indices for each backbone model."""
    if 'v2_300' in backbone_model:
        return [5, 11, 17, 23]  # indices for prithvi_eo_v2_300
    elif 'v2_600' in backbone_model:
        return [7, 15, 23, 31]  # indices for prithvi_eo_v2_600
    else:
        # Default to 300 indices
        return [5, 11, 17, 23]


def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    
    # All parameters must have CONSISTENT value spaces across all trials
    # Optuna doesn't support dynamic value spaces
    
#     backbone_model = trial.suggest_categorical('backbone', 
#         ['prithvi_eo_v2_300_tl', 'prithvi_eo_v2_600_tl'])
    
#     # Use consistent parameter spaces - adjust values after suggestion if needed
#     batch_size_suggested = trial.suggest_categorical('batch_size', [8, 16, 32])
#     decoder_channels_suggested = trial.suggest_categorical('decoder_channels', [128, 256, 512])
    
#     # Cap values for larger backbone to prevent OOM
#     if backbone_model == 'prithvi_eo_v2_600_tl':
#         batch_size = min(batch_size_suggested, 16)  # Cap at 16 for 600 model
#         decoder_channels = min(decoder_channels_suggested, 256)  # Cap at 256 for 600 model
#         if batch_size != batch_size_suggested or decoder_channels != decoder_channels_suggested:
#             print(f"  Note: Adjusted batch_size {batch_size_suggested}->{batch_size}, "
#                   f"decoder_channels {decoder_channels_suggested}->{decoder_channels} for 600 model")
#     else:
#         batch_size = batch_size_suggested
#         decoder_channels = decoder_channels_suggested
    
#     # Other hyperparameters to tune
#     lr = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
#     weight_decay = trial.suggest_float('weight_decay', 0.01, 0.5, log=True)
#     head_dropout = trial.suggest_float('head_dropout', 0.0, 0.5)
    
#     # Additional hyperparameters
#     optimizer_type = trial.suggest_categorical('optimizer', ['AdamW', 'SGD', 'Adam'])
#     use_scheduler = trial.suggest_categorical('use_scheduler', [True, False])
    
#     # Class weights - always suggest weight_scale, only use it if use_class_weights is True
#     use_class_weights = trial.suggest_categorical('use_class_weights', [True, False])
#     weight_scale = trial.suggest_float('weight_scale', 0.5, 2.0)  # Always suggest for consistency
#     if use_class_weights:
#         class_weights = [2.28 * weight_scale, 1.02 * weight_scale, 
#                         1.37 * weight_scale, 0.54 * weight_scale]
#     else:
#         class_weights = None
    
#     # Architecture choices
#     freeze_backbone = trial.suggest_categorical('freeze_backbone', [True, False])

    # FIXED - don't waste trials on these
    freeze_backbone = False
    use_scheduler = False
    use_class_weights = False
    class_weights = None
    batch_size = 16
    decoder_channels = 512
    optimizer_type = 'AdamW'
    
    # SEARCH - the important ones with narrowed ranges
    backbone_model = trial.suggest_categorical('backbone', 
        ['prithvi_eo_v2_300_tl', 'prithvi_eo_v2_600_tl'])
    head_dropout = trial.suggest_float('head_dropout', 0.35, 0.5)  # Narrowed
    weight_decay = trial.suggest_float('weight_decay', 0.2, 0.5)   # Narrowed
    lr = trial.suggest_float('lr', 3e-5, 2e-4, log=True)           # Narrowed around best


    
    print(f"\nTrial {trial.number} hyperparameters:")
    print(f"  backbone: {backbone_model}")
    print(f"  lr: {lr:.6f}")
    print(f"  weight_decay: {weight_decay:.4f}")
    print(f"  head_dropout: {head_dropout:.4f}")
    print(f"  batch_size: {batch_size}")
    print(f"  decoder_channels: {decoder_channels}")
    print(f"  optimizer: {optimizer_type}")
    print(f"  freeze_backbone: {freeze_backbone}")
    
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
    
    # Pruning callback for early stopping of bad trials
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
        # Create data module with reduced workers for memory efficiency
        data_module = create_datamodule(batch_size=batch_size, num_workers=4)
        
        # Trainer - single GPU for tuning trials
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
            strategy='auto',  # Use 'auto' for single GPU
            enable_model_summary=False,
            enable_progress_bar=True
        )
        
        # Get correct neck indices for this backbone
        neck_indices = get_neck_indices(backbone_model)
        
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
                    "indices": neck_indices
                },
                {
                    "name": "ReshapeTokensToImage",
                    "effective_time_dim": 3
                }
            ]
        }
        
        # Create optimizer parameters based on optimizer type
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
        
        # Add scheduler by overriding configure_optimizers
        if use_scheduler:
            original_configure_optimizers = model.configure_optimizers

            def configure_optimizers_with_scheduler():
                opt_config = original_configure_optimizers()
                if isinstance(opt_config, dict):
                    optimizer = opt_config["optimizer"]
                else:
                    optimizer = opt_config

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=MAX_EPOCHS_TUNING
                )

                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch"
                    }
                }

            model.configure_optimizers = configure_optimizers_with_scheduler
        
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
            "val_loss": trainer.callback_metrics.get("val_loss", 0.0).item() if "val_loss" in trainer.callback_metrics else 0.0,
            "hyperparameters": trial.params,
            "epochs_trained": trainer.current_epoch + 1
        }
        
        with open(os.path.join(output_dir, "trial_results.json"), "w") as f:
            json.dump(trial_results, f, indent=2)
        
        # Enhanced cleanup to prevent memory accumulation
        del model, trainer, data_module
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()
        gc.collect()  # Call twice for thorough cleanup
        
        print(f"Trial {trial.number} completed. Val Jaccard: {val_score:.4f}")
        
        return val_score
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error too
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        return 0.0


def train_best_model(study):
    """Train the model with best hyperparameters for full epochs."""
    
    print("\n" + "="*50)
    print("Training with best hyperparameters")
    print("="*50)
    
    best_params = study.best_params
    print(f"Best parameters: {json.dumps(best_params, indent=2)}")
    print(f"Best validation Jaccard: {study.best_value:.4f}")
    
    # Extract best hyperparameters
    lr = best_params['lr']
    weight_decay = best_params['weight_decay']
    head_dropout = best_params['head_dropout']
    batch_size = best_params['batch_size']
    decoder_channels = best_params['decoder_channels']
    optimizer_type = best_params['optimizer']
    use_scheduler = best_params['use_scheduler']
    use_class_weights = best_params['use_class_weights']
    freeze_backbone = best_params['freeze_backbone']
    backbone_model = best_params['backbone']
    
    if use_class_weights:
        weight_scale = best_params.get('weight_scale', 1.0)
        class_weights = [2.28 * weight_scale, 1.02 * weight_scale, 
                        1.37 * weight_scale, 0.54 * weight_scale]
    else:
        class_weights = None
    
    # Set seed
    pl.seed_everything(0)
    
    # Setup directories for final model
    final_output_dir = os.path.join("optuna_outputs", STUDY_NAME, "best_model")
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=final_output_dir,
        name="final_training"
    )
    
    # Class names
    class_names = ["Water", "Natural", "Irrigated", "Rainfed"]
    
    # Callbacks for final training
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(final_output_dir, "checkpoints"),
        monitor="val/Multiclass_Jaccard_Index",
        mode="max",
        filename="best_model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        save_last=True
    )
    
    loss_tracker = LossTrackerCallback()
    conf_matrix_callback = ConfusionMatrixCallback(
        class_names=class_names,
        save_dir=os.path.join(final_output_dir, "confusion_matrices")
    )
    
    early_stopping = EarlyStopping(
        monitor="val/Multiclass_Jaccard_Index",
        mode="max",
        patience=10,
        min_delta=0.0001
    )
    
    # Data module - can use more workers for final training
    data_module = create_datamodule(batch_size=batch_size, num_workers=8)
    
    # Determine number of GPUs for final training
    available_gpus = torch.cuda.device_count()
    final_gpus = min(2, available_gpus)  # Use up to 2 GPUs for final training
    
    # Trainer for final training
    trainer = pl.Trainer(
        accelerator="auto",
        devices=final_gpus,
        precision="bf16-mixed",
        num_nodes=1,
        logger=logger,
        max_epochs=FINAL_EPOCHS,
        check_val_every_n_epoch=2,
        log_every_n_steps=10,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, loss_tracker, conf_matrix_callback, early_stopping],
        default_root_dir=final_output_dir,
        strategy='ddp_find_unused_parameters_true' if final_gpus > 1 else 'auto'
    )
    
    # Get correct neck indices for the best backbone
    neck_indices = get_neck_indices(backbone_model)
    
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
                "indices": neck_indices
            },
            {
                "name": "ReshapeTokensToImage",
                "effective_time_dim": 3
            }
        ]
    }
    
    # Optimizer parameters
    if optimizer_type == 'SGD':
        optimizer_hparams = {
            "weight_decay": weight_decay,
            "momentum": 0.9
        }
    else:
        optimizer_hparams = {"weight_decay": weight_decay}
    
    # Create model with best hyperparameters
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
    
    # Add scheduler for final training
    if use_scheduler:
        original_configure_optimizers = model.configure_optimizers

        def configure_optimizers_with_scheduler():
            opt_config = original_configure_optimizers()
            if isinstance(opt_config, dict):
                optimizer = opt_config["optimizer"]
            else:
                optimizer = opt_config

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=FINAL_EPOCHS
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }

        model.configure_optimizers = configure_optimizers_with_scheduler
    
    # Train final model
    trainer.fit(model, datamodule=data_module)
    
    # Test the model
    test_results = trainer.test(model, datamodule=data_module, ckpt_path='best')
    
    # Save final results
    final_results = {
        "best_hyperparameters": best_params,
        "best_val_jaccard_from_study": study.best_value,
        "final_val_jaccard": trainer.callback_metrics.get("val/Multiclass_Jaccard_Index", 0.0).item() 
            if "val/Multiclass_Jaccard_Index" in trainer.callback_metrics else 0.0,
        "test_results": test_results,
        "checkpoint_path": checkpoint_callback.best_model_path
    }
    
    with open(os.path.join(final_output_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Results saved to: {final_output_dir}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print("="*50)
    
    return model, trainer, data_module


def main():
    """Main function to run the hyperparameter optimization."""
    
    print("="*60)
    print("Starting Optuna Hyperparameter Optimization")
    print("="*60)
    print(f"Study name: {STUDY_NAME}")
    print(f"Number of trials: {N_TRIALS}")
    print(f"Max epochs per trial: {MAX_EPOCHS_TUNING}")
    print(f"Final training epochs: {FINAL_EPOCHS}")
    print(f"GPUs per trial: {N_GPUS}")
    print("="*60)
    
    # Create study with SQLite storage for persistence
    # Use study name in database file to avoid conflicts with old studies
    db_path = f'sqlite:///optuna_study_{STUDY_NAME}.db'
    print(f"Using database: {db_path}")
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction='maximize',
        storage=db_path,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        ),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=None,
        catch=(Exception,),
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    # Print statistics
    print("\n" + "="*50)
    print("Optimization completed!")
    print("="*50)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len(pruned_trials)}")
    print(f"Number of complete trials: {len(complete_trials)}")
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Jaccard): {trial.value:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save study results
    study_dir = os.path.join("optuna_outputs", STUDY_NAME)
    os.makedirs(study_dir, exist_ok=True)
    
    # Save all trials data
    import pandas as pd
    df = study.trials_dataframe()
    df.to_csv(os.path.join(study_dir, "all_trials.csv"), index=False)
    
    # Save study summary
    summary = {
        "study_name": STUDY_NAME,
        "n_trials": N_TRIALS,
        "n_complete": len(complete_trials),
        "n_pruned": len(pruned_trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number
    }
    
    with open(os.path.join(study_dir, "study_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Create visualizations
    try:
        import optuna.visualization as vis
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(study_dir, "optimization_history.html"))
        
        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(os.path.join(study_dir, "param_importances.html"))
        
        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(study_dir, "parallel_coordinate.html"))
        
        print(f"\nVisualizations saved to {study_dir}")
    except Exception as e:
        print(f"Could not create visualizations: {e}")
    
    # Train final model with best parameters
    print("\nTraining final model with best hyperparameters...")
    model, trainer, data_module = train_best_model(study)
    
    print("\nHyperparameter optimization and final training complete!")


if __name__ == "__main__":
    main()