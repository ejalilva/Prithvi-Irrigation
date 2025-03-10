#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
import cv2
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


# In[2]:


# Get absolute path of the local package
local_package_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'terratorch'))

# Add the path to system path if it's not already there
if local_package_path not in sys.path:
    sys.path.insert(0, local_package_path)

# If you had previously imported terratorch, you'll need to reload it
import importlib
if 'terratorch' in sys.modules:
    importlib.reload(sys.modules['terratorch'])

import terratorch

# Verify the import
# print(f"terratorch package location: {terratorch.__file__}")

from terratorch.datamodules import MultiTemporalCropClassificationDataModule
from terratorch.datasets import MultiTemporalCropClassification
from terratorch.tasks import SemanticSegmentationTask
from terratorch.datasets.transforms import FlattenTemporalIntoChannels, UnflattenTemporalFromChannels


# In[6]:


import albumentations as A
from albumentations import Compose, Flip
from albumentations.pytorch import ToTensorV2


# In[7]:


import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


# In[8]:


if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs:", num_gpus)


DATASET_PATH = '../../data/prithvi/datasets--ibm-nasa-geospatial--multi-temporal-irrigation-classificaction/snapshots/04b439f179e52a7b144f69676210eecd30c39cfc/' 


# In[11]:


# those are the recommended transforms for this task
transforms = [
    terratorch.datasets.transforms.FlattenTemporalIntoChannels(),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # A.RandomBrightnessContrast(p=0.5),
    A.Normalize(),
    ToTensorV2(),
    terratorch.datasets.transforms.UnflattenTemporalFromChannels(n_timesteps=3),
]


# In[21]:


# this datamodule allows access to the dataset in its various splits.
datamodule = MultiTemporalCropClassificationDataModule(
    batch_size=16, # 16 images at a time
    data_root=DATASET_PATH,
    train_transform=transforms, # transformation to apply to the data during training
    val_transform=transforms, # transformation to apply to the data during validation
    test_transform=transforms, # transformation to apply to the data during test
    expand_temporal_dimension=True,
)


# In[22]:


# checking for the dataset means and stds
datamodule.means, datamodule.stds


# In[23]:


# checking datasets train split size
datamodule.setup("fit")
train_dataset = datamodule.train_dataset
len(train_dataset)


# In[24]:


# checking datasets available bands
train_dataset.all_band_names


# In[25]:


# checking datasets classes
train_dataset.class_names


# In[26]:


# ploting a few samples
for i in range(5):
    train_dataset.plot(train_dataset[i])


# In[27]:


# checking datasets validation split size
val_dataset = datamodule.val_dataset
len(val_dataset)


# In[28]:


# checking datasets testing split size
datamodule.setup("test")
test_dataset = datamodule.test_dataset
len(test_dataset)


# In[29]:


pl.seed_everything(0)

# Logger
logger = TensorBoardLogger(
    save_dir="output",
    name="tutorial"
)

# colab will kill the kernel after ~24 epochs, therefore stopping after two and pull the correct checkpoint after
max_epochs = 1 if 'google.colab' in sys.modules else 50 

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",                 # Directory to save the checkpoints
    mode="max",                             # Maximize variable
    monitor="val/Multiclass_Jaccard_Index", # Variable to monitor
    filename="epoch-{epoch:02d}",           # Filename format
    save_top_k=-1,                          # Save all checkpoints
    every_n_epochs=50,                      # Save every epoch
    save_on_train_epoch_end=True,            # Ensure saving after each epoch
    save_last=True  # This will always keep the last epoch checkpoint.

)

# Trainer
trainer = pl.Trainer(
    accelerator="auto",
    # strategy="auto",
    devices="auto",
    num_nodes=1,
    logger = logger,
    max_epochs=max_epochs,
    check_val_every_n_epoch=2,
    log_every_n_steps=10,
    enable_checkpointing=True,
    callbacks=[checkpoint_callback],
    default_root_dir="root_dir",
    strategy='ddp_find_unused_parameters_true' 
)

# DataModule
data_module = MultiTemporalCropClassificationDataModule(
    batch_size=16, # increased to 16 from 8
    data_root=DATASET_PATH,
    train_transform=transforms,
    val_transform=transforms,
    test_transform=transforms,
    reduce_zero_label=True,
    expand_temporal_dimension=True,
    use_metadata=False, # Multicropclassification dataset has metadata for location and time
    num_workers=23
)


# In[ ]:


# Model
model = SemanticSegmentationTask(
    model_args={
        "decoder": "UperNetDecoder",
        "backbone_pretrained": True,
        "backbone": "prithvi_eo_v2_300_tl", # Model can be either prithvi_eo_v2_300, prithvi_eo_v2_300_tl, prithvi_eo_v2_600, prithvi_eo_v2_600_tl
        "backbone_in_channels": 6,
        "backbone_features_only": True,
        # "backbone_coords_encoding": ["time", "location"], # this must be used to use time and location metadata
        "rescale": True,
        "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
        "backbone_num_frames": 3,
        "num_classes": 4,
        "head_dropout": 0.3, # increased to 0.3 from 0.1 this is to drop a fraction of neurons at each iteration (here 30%) to avoid overfitting
        "decoder_channels": 256,
        "decoder_scale_modules": True,
        "necks": [
            {
                "name": "SelectIndices",
                #"indices": [2, 5, 8, 11] # indices for prithvi_vit_100
                "indices": [5, 11, 17, 23] # indices for prithvi_eo_v2_300
                # "indices": [7, 15, 23, 31] # indices for prithvi_eo_v2_600
            },
            {
                "name": "ReshapeTokensToImage",
                "effective_time_dim": 3
            }
        ]

    },
    plot_on_val=False,
    class_weights=[42.59142538,  9.41862502,  3.79541504,  1.64779206], # water, irrigated, rainfed, natural
    loss="ce",
    lr=1.0e-4, # decreased to 1e-4 from 2e-4
    optimizer="AdamW",
    optimizer_hparams={"weight_decay": 0.3}, # increased to 0.3 from 0.1
    ignore_index=-1,
    freeze_backbone=False,
    freeze_decoder=False,
    model_factory="EncoderDecoderFactory",
)

# Training
trainer.fit(model, datamodule=data_module)


# In[ ]:


trainer.test(model, datamodule=data_module, ckpt_path='checkpoints/epoch-epoch=49-v1.ckpt')


# In[ ]:


# now we can use the model for predictions and ploting!
best_ckpt_path = 'checkpoints/epoch-epoch=49-v1.ckpt'
model = SemanticSegmentationTask.load_from_checkpoint(
    best_ckpt_path,
    model_args=model.hparams.model_args,
    model_factory=model.hparams.model_factory
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = data_module.test_dataloader()
model.to(device)
with torch.no_grad():
    batch = next(iter(test_loader))
    images = batch["image"].to(device)
    masks = batch["mask"].numpy()
    other_keys = batch.keys() - {"image", "mask", "filename"}
    rest = {k: batch[k].to(device) for k in other_keys}

    outputs = model(images, **rest)
    preds = torch.argmax(outputs.output, dim=1).cpu().numpy()

for i in range(5):
    sample = {key: batch[key][i] for key in batch}
    sample["prediction"] = preds[i]
    test_dataset.plot(sample)

