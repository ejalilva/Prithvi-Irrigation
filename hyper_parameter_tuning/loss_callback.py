import lightning.pytorch as pl
import matplotlib.pyplot as plt

class LossTrackerCallback(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch."""
        if "train_loss" in trainer.callback_metrics:
            epoch_loss = trainer.callback_metrics["train_loss"].item()
            self.train_losses.append(epoch_loss)
            self.epochs.append(trainer.current_epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of each validation epoch."""
        if "val_loss" in trainer.callback_metrics:
            epoch_val_loss = trainer.callback_metrics["val_loss"].item()
            self.val_losses.append(epoch_val_loss)

    def plot_loss(self):
        """Plot training and validation loss after training."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.epochs, self.train_losses, label="Train Loss", marker="o")
        if self.val_losses:
            plt.plot(self.epochs, self.val_losses, label="Validation Loss", marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss vs. Epochs")
        plt.legend()
        plt.grid()
        plt.show()
