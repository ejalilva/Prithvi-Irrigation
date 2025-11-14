# test_multi_gpu.py
import torch
import lightning.pytorch as pl
from terratorch.tasks import SemanticSegmentationTask

def test_multi_gpu():
    print(f"GPUs available: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    
    # Minimal trainer test
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        fast_dev_run=True,  # Just run 1 batch
        max_epochs=1
    )
    
    print("Trainer initialized successfully")
    
if __name__ == "__main__":
    test_multi_gpu()