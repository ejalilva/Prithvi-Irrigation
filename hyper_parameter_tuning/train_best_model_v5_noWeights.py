#!/usr/bin/env python
"""Train model with V5 Trial 30 params + class weights."""

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['PROJ_LIB'] = '/home/ejalilva/.conda/envs/terratorch-tune/lib/python3.12/site-packages/rasterio/proj_data'
os.environ['PROJ_DATA'] = os.environ['PROJ_LIB']
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['PROJ_NETWORK'] = 'OFF'
import pyproj
pyproj.network.set_network_enabled(False)

import sys
import torch
import json
import numpy as np
torch.set_float32_matmul_precision('medium')

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

# ============== LOAD TRIAL 30 FROM V5 ==============
import optuna
study = optuna.load_study(
    study_name="prithvi_tuning_V5",
    storage="sqlite:///optuna_study_prithvi_tuning_V5.db"
)
trial_30 = study.trials[30]
print(f"Loaded trial 30: Jaccard={trial_30.value:.4f}")
print(f"Params: {trial_30.params}")

# ============== CONFIG ==============
DATASET_PATH = '/discover/nobackup/ejalilva/data/prithvi/datasets--ibm-nasa-geospatial--multi-temporal-irrigation-classificaction-openet/snapshots/04b439f179e52a7b144f69676210eecd30c39cfc/'
base_weights = [29.7, 2.1, 3.2, 5.5]
OUTPUT_DIR = 'best_model_V5_without_weights'
MAX_EPOCHS = 120

# Fixed params (from V5 analysis)
batch_size = 8
backbone_model = 'prithvi_eo_v2_600_tl'
decoder_channels = 512
optimizer_type = 'AdamW'

# From trial 30
lr = trial_30.params['lr']
weight_decay = trial_30.params['weight_decay']
head_dropout = trial_30.params['head_dropout']

# Class weights - sqrt baseline (scale=1.0)
# class_weights = [np.sqrt(w) for w in base_weights]
class_weights = None
print(f"\nConfig:")
print(f"  lr: {lr:.6f}")
print(f"  weight_decay: {weight_decay:.4f}")
print(f"  head_dropout: {head_dropout:.4f}")
print(f"  class_weights: {class_weights}")

# ============== SETUP ==============
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

pl.seed_everything(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)

transforms = [
    terratorch.datasets.transforms.FlattenTemporalIntoChannels(),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2(),
    terratorch.datasets.transforms.UnflattenTemporalFromChannels(n_timesteps=3),
]

data_module = MultiTemporalCropClassificationDataModule(
    batch_size=batch_size,
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

neck_indices = [7, 15, 23, 31]  # 600 model

model = SemanticSegmentationTask(
    model_args={
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
            {"name": "SelectIndices", "indices": neck_indices},
            {"name": "ReshapeTokensToImage", "effective_time_dim": 3}
        ]
    },
    plot_on_val=False,
    class_weights=class_weights,
    loss="ce",
    lr=lr,
    optimizer=optimizer_type,
    optimizer_hparams={"weight_decay": weight_decay},
    ignore_index=-1,
    freeze_backbone=False,
    freeze_decoder=False,
    model_factory="EncoderDecoderFactory",
)

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
    strategy='ddp_find_unused_parameters_true' if num_gpus > 1 else 'auto'
)

# ============== TRAIN ==============
trainer.fit(model, datamodule=data_module)

# ============== TEST ==============
print("\nRunning test evaluation...")
test_results = trainer.test(model, datamodule=data_module, ckpt_path='best')

# Save results
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump({
        "trial_30_params": trial_30.params,
        "trial_30_jaccard": trial_30.value,
        "class_weights": class_weights,
        "test_results": test_results,
        "best_checkpoint": trainer.checkpoint_callback.best_model_path
    }, f, indent=2)

print(f"\nDone! Results saved to {OUTPUT_DIR}/")