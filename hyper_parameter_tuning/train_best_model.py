#!/usr/bin/env python
"""Train model with best Optuna hyperparameters."""

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['PROJ_LIB'] = '/home/ejalilva/.conda/envs/terratorch-tune/lib/python3.12/site-packages/rasterio/proj_data'
os.environ['PROJ_DATA'] = os.environ['PROJ_LIB']
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['PROJ_NETWORK'] = 'OFF'  # Disable PROJ network access
import pyproj
pyproj.network.set_network_enabled(False)

import sys
import torch
import json
import numpy as np
torch.set_float32_matmul_precision('medium')

# Local terratorch
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..', 'terratorch')))

import terratorch
from terratorch.datamodules import MultiTemporalCropClassificationDataModule
from terratorch.tasks import SemanticSegmentationTask
import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from loss_callback import LossTrackerCallback
from confusionMatrix_callback_withVal import ConfusionMatrixCallback

# ============== LOAD BEST PARAMS FROM OPTUNA ==============
import optuna
study = optuna.load_study(
    study_name="prithvi_tuning_V4",
    storage="sqlite:///optuna_study_prithvi_tuning_V4.db"
)
BEST_PARAMS = study.best_params

BEST_PARAMS['freeze_backbone'] = False
BEST_PARAMS['use_scheduler'] = False
BEST_PARAMS['use_class_weights'] = True
BEST_PARAMS['class_weights'] = 1
BEST_PARAMS['batch_size'] = 16
BEST_PARAMS['decoder_channels'] = 512
BEST_PARAMS['optimizer'] = 'AdamW'
print(f"Loaded best params (trial {study.best_trial.number}, Jaccard={study.best_value:.4f})")

DATASET_PATH = '/discover/nobackup/ejalilva/data/prithvi/datasets--ibm-nasa-geospatial--multi-temporal-irrigation-classificaction-openet/snapshots/04b439f179e52a7b144f69676210eecd30c39cfc/'
base_weights = [29.7, 2.1, 3.2, 5.5] # Use sqrt for softer weighting
OUTPUT_DIR = 'best_model_training'
MAX_EPOCHS = 120

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs:", num_gpus)
    
# ============== SETUP ==============
pl.seed_everything(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data
transforms = [
    terratorch.datasets.transforms.FlattenTemporalIntoChannels(),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2(),
    terratorch.datasets.transforms.UnflattenTemporalFromChannels(n_timesteps=3),
]

data_module = MultiTemporalCropClassificationDataModule(
    batch_size=BEST_PARAMS['batch_size'],
    data_root=DATASET_PATH,
    train_transform=transforms,
    val_transform=transforms,
    test_transform=transforms,
    reduce_zero_label=False,
    expand_temporal_dimension=True,
    use_metadata=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

# Neck indices based on backbone
neck_indices = [7, 15, 23, 31] if '600' in BEST_PARAMS['backbone'] else [5, 11, 17, 23]

# Model
model = SemanticSegmentationTask(
    model_args={
        "decoder": "UperNetDecoder",
        "backbone_pretrained": True,
        "backbone": BEST_PARAMS['backbone'],
        "backbone_in_channels": 6,
        "backbone_features_only": True,
        "backbone_coords_encoding": ["time", "location"],
        "rescale": True,
        "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
        "backbone_num_frames": 3,
        "num_classes": 4,
        "head_dropout": BEST_PARAMS['head_dropout'],
        "decoder_channels": BEST_PARAMS['decoder_channels'],
        "decoder_scale_modules": True,
        "necks": [
            {"name": "SelectIndices", "indices": neck_indices},
            {"name": "ReshapeTokensToImage", "effective_time_dim": 3}
        ]
    },
    plot_on_val=False,
    class_weights=[np.sqrt(w) * BEST_PARAMS['class_weights'] for w in base_weights] if BEST_PARAMS['use_class_weights'] else None,
    loss="ce",
    lr=BEST_PARAMS['lr'],
    optimizer=BEST_PARAMS['optimizer'],
    optimizer_hparams={"weight_decay": BEST_PARAMS['weight_decay']},
    ignore_index=-1,
    freeze_backbone=BEST_PARAMS['freeze_backbone'],
    freeze_decoder=False,
    model_factory="EncoderDecoderFactory",
)

# Callbacks
callbacks = [
    ModelCheckpoint(
        dirpath=os.path.join(OUTPUT_DIR, "checkpoints"),
        monitor="val/Multiclass_Jaccard_Index",
        mode="max",
        filename="best-{epoch:02d}-{val/Multiclass_Jaccard_Index:.4f}",
        save_top_k=3,
        save_last=True
    ),
    EarlyStopping(
        monitor="val/Multiclass_Jaccard_Index",
        mode="max",
        patience=15,
        min_delta=0.001
    ),
    LossTrackerCallback(),
    ConfusionMatrixCallback(
        class_names=["Water", "Natural", "Irrigated", "Rainfed"],
        save_dir=os.path.join(OUTPUT_DIR, "confusion_matrices")
    ),
]

# Trainer
trainer = pl.Trainer(
    accelerator="auto",
    devices=num_gpus,
    precision="bf16-mixed",
    logger=TensorBoardLogger(save_dir=OUTPUT_DIR, name="logs"),
    max_epochs=MAX_EPOCHS,
    check_val_every_n_epoch=2,
    log_every_n_steps=10,
    callbacks=callbacks,
    default_root_dir=OUTPUT_DIR,
    strategy='ddp_find_unused_parameters_true'
)

# ============== TRAIN ==============
print(f"Training with params: {json.dumps(BEST_PARAMS, indent=2)}")
trainer.fit(model, datamodule=data_module)

# ============== TEST ==============
print("\nRunning test evaluation...")
test_results = trainer.test(model, datamodule=data_module, ckpt_path='best')

# Save results
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump({
        "best_params": BEST_PARAMS,
        "test_results": test_results,
        "best_checkpoint": trainer.checkpoint_callback.best_model_path
    }, f, indent=2)

# ============== SAVE MODEL ==============
# Option 1: Already saved by ModelCheckpoint (best_model_training/checkpoints/)
best_ckpt = trainer.checkpoint_callback.best_model_path
print(f"Best checkpoint saved at: {best_ckpt}")

# Option 2: Save final model state dict (smaller, for inference only)
torch.save(model.model.state_dict(), os.path.join(OUTPUT_DIR, "model_state_dict.pt"))

# Option 3: Save entire model with hyperparameters (for resuming training)
trainer.save_checkpoint(os.path.join(OUTPUT_DIR, "final_model.ckpt"))

print(f"\nDone! Results saved to {OUTPUT_DIR}/")
print(f"Best checkpoint: {best_ckpt}")
print(f"State dict: {OUTPUT_DIR}/model_state_dict.pt")
print(f"Final checkpoint: {OUTPUT_DIR}/final_model.ckpt")
