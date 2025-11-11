import matplotlib.pyplot as plt
import lightning.pytorch as pl
import torch

class LossTrackerCallback(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.train_logged = False
        self.val_logged = False
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called at the end of each training batch."""
        # Capture loss directly from training outputs if available
        if isinstance(outputs, dict) and 'loss' in outputs:
            if not hasattr(self, 'current_train_loss'):
                self.current_train_loss = []
            self.current_train_loss.append(outputs['loss'].detach().item())
        elif isinstance(outputs, torch.Tensor):
            if not hasattr(self, 'current_train_loss'):
                self.current_train_loss = []
            self.current_train_loss.append(outputs.detach().item())
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch."""
        # Debug: Print all available metrics
        print(f"Available metrics at train epoch end: {list(trainer.callback_metrics.keys())}")
        
        # Try to get loss from various possible metric names
        possible_keys = [
            "train_loss", "loss", "train/loss", "training_loss", 
            "loss/train", "train/ce_loss", "ce_loss", "cross_entropy_loss"
        ]
        
        loss_found = False
        for key in possible_keys:
            if key in trainer.callback_metrics:
                epoch_loss = trainer.callback_metrics[key].item()
                self.train_losses.append(epoch_loss)
                self.epochs.append(trainer.current_epoch)
                print(f"Epoch {trainer.current_epoch}: Train Loss = {epoch_loss}")
                loss_found = True
                self.train_logged = True
                break
        
        # If we couldn't find the loss in callback_metrics, use our captured batch losses
        if not loss_found and hasattr(self, 'current_train_loss') and self.current_train_loss:
            epoch_loss = sum(self.current_train_loss) / len(self.current_train_loss)
            self.train_losses.append(epoch_loss)
            self.epochs.append(trainer.current_epoch)
            print(f"Epoch {trainer.current_epoch}: Train Loss = {epoch_loss} (calculated from batches)")
            self.train_logged = True
        
        # Reset batch loss tracking for next epoch
        self.current_train_loss = []
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Called at the end of each validation batch."""
        # Capture loss directly from validation outputs if available
        if isinstance(outputs, dict) and 'loss' in outputs:
            if not hasattr(self, 'current_val_loss'):
                self.current_val_loss = []
            self.current_val_loss.append(outputs['loss'].detach().item())
        elif isinstance(outputs, torch.Tensor):
            if not hasattr(self, 'current_val_loss'):
                self.current_val_loss = []
            self.current_val_loss.append(outputs.detach().item())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of each validation epoch."""
        # Debug: Print all available metrics
        print(f"Available metrics at val epoch end: {list(trainer.callback_metrics.keys())}")
        
        # Try to get loss from various possible metric names
        possible_keys = [
            "val_loss", "val/loss", "validation_loss", "loss/val", 
            "val/ce_loss", "val/cross_entropy_loss", "validation/loss"
        ]
        
        loss_found = False
        for key in possible_keys:
            if key in trainer.callback_metrics:
                epoch_val_loss = trainer.callback_metrics[key].item()
                while len(self.val_losses) < len(self.train_losses) - 1:
                    # Fill in missing values if we missed some validation epochs
                    self.val_losses.append(self.val_losses[-1] if self.val_losses else 0)
                self.val_losses.append(epoch_val_loss)
                print(f"Epoch {trainer.current_epoch}: Val Loss = {epoch_val_loss}")
                loss_found = True
                self.val_logged = True
                break
        
        # If we couldn't find the loss in callback_metrics, use our captured batch losses
        if not loss_found and hasattr(self, 'current_val_loss') and self.current_val_loss:
            epoch_val_loss = sum(self.current_val_loss) / len(self.current_val_loss)
            while len(self.val_losses) < len(self.train_losses) - 1:
                # Fill in missing values if we missed some validation epochs
                self.val_losses.append(self.val_losses[-1] if self.val_losses else 0)
            self.val_losses.append(epoch_val_loss)
            print(f"Epoch {trainer.current_epoch}: Val Loss = {epoch_val_loss} (calculated from batches)")
            self.val_logged = True
        
        # Reset batch loss tracking for next epoch
        self.current_val_loss = []
    
    def on_fit_end(self, trainer, pl_module):
        """Called at the end of training."""
        if not self.train_logged:
            print("Warning: No training losses were logged throughout training")
        if not self.val_logged:
            print("Warning: No validation losses were logged throughout training")
            
        self.plot_loss()
    
    def plot_loss(self, filename="loss_plot.png"):
        """Plot training and validation loss and save as PNG."""
        if not self.train_losses:
            print("No loss values collected. Cannot generate plot.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label="Train Loss", marker="o", linestyle="-", linewidth=2)
        
        if self.val_losses:
            # Ensure val_losses has same length as train_losses
            if len(self.val_losses) < len(self.train_losses):
                print(f"Warning: Only {len(self.val_losses)} validation losses available for {len(self.train_losses)} epochs")
                # Pad with last value
                last_val = self.val_losses[-1] if self.val_losses else 0
                self.val_losses.extend([last_val] * (len(self.train_losses) - len(self.val_losses)))
            
            plt.plot(self.epochs, self.val_losses[:len(self.epochs)], label="Validation Loss", marker="s", linestyle="-", linewidth=2)
        
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Loss plot saved as {filename}")
        
        return plt.gcf()  # Return the figure for further customization if needed