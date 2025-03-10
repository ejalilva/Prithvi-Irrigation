from terratorch.tasks import SemanticSegmentationTask
import torch

class ExtendedSemanticSegmentationTask(SemanticSegmentationTask):
    """
    Extends the SemanticSegmentationTask to collect predictions and targets
    for confusion matrix computation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
    
    def training_step(self, batch, batch_idx):
        # Get the original training step result
        result = super().training_step(batch, batch_idx)
        
        # Extract predictions and targets
        with torch.no_grad():
            x, y = batch
            # Forward pass
            y_hat = self.forward(x)
            # Store predictions and targets
            self.train_step_outputs.append({
                'preds': y_hat.detach(),
                'targets': y.detach()
            })
        
        return result
    
    def validation_step(self, batch, batch_idx):
        # Get the original validation step result
        result = super().validation_step(batch, batch_idx)
        
        # Extract predictions and targets
        with torch.no_grad():
            x, y = batch
            # Forward pass
            y_hat = self.forward(x)
            # Store predictions and targets
            self.val_step_outputs.append({
                'preds': y_hat.detach(),
                'targets': y.detach()
            })
        
        return result
    
    def test_step(self, batch, batch_idx):
        # Get the original test step result
        result = super().test_step(batch, batch_idx)
        
        # Extract predictions and targets
        with torch.no_grad():
            x, y = batch
            # Forward pass
            y_hat = self.forward(x)
            # Store predictions and targets
            self.test_step_outputs.append({
                'preds': y_hat.detach(),
                'targets': y.detach()
            })
        
        return result
    
    def on_train_epoch_end(self):
        # Call the parent's on_train_epoch_end if it exists
        if hasattr(super(), 'on_train_epoch_end'):
            super().on_train_epoch_end()
    
    def on_validation_epoch_end(self):
        # Call the parent's on_validation_epoch_end if it exists
        if hasattr(super(), 'on_validation_epoch_end'):
            super().on_validation_epoch_end()
    
    def on_test_epoch_end(self):
        # Call the parent's on_test_epoch_end if it exists
        if hasattr(super(), 'on_test_epoch_end'):
            super().on_test_epoch_end()
