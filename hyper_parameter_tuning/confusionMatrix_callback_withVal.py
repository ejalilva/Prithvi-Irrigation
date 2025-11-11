import lightning.pytorch as pl
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ConfusionMatrixCallback(pl.Callback):
    """
    Callback to collect predictions and ground truth tensors during the final epoch
    and generate a confusion matrix.
    """
    def __init__(self, class_names=None, save_dir="confusion_matrices"):
        super().__init__()
        self.save_dir = save_dir
        self.class_names = class_names or ["Class 0", "Class 1", "Class 2", "Class 3"]
        self.val_predictions = []
        self.val_ground_truths = []
        self.train_predictions = []
        self.train_ground_truths = []
        self.is_final_epoch = False
        
        # Create the save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Create a log file
        self.log_file = os.path.join(self.save_dir, "confusion_matrix_debug.log")
        with open(self.log_file, "w") as f:
            f.write(f"Debug log started at {self.log_file}\n")
            f.write(f"Save directory: {self.save_dir}\n")
            f.write(f"Working directory: {os.getcwd()}\n")
            f.write(f"Python executable: {sys.executable}\n")
            f.write(f"Python version: {sys.version}\n")
    
    def log_message(self, message):
        """Write a message to the log file"""
        with open(self.log_file, "a") as f:
            f.write(f"{message}\n")
    
    def on_fit_start(self, trainer, pl_module):
        self.log_message(f"fit_start - trainer: {trainer.__class__.__name__}, pl_module: {pl_module.__class__.__name__}")
        self.log_message(f"Max epochs set to {trainer.max_epochs}")
    
    def on_train_epoch_start(self, trainer, pl_module):
        # Check if this is the final epoch
        current_epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs
        self.log_message(f"train_epoch_start - epoch {current_epoch+1}/{max_epochs}")
        
        # Set flag if this is the final epoch
        self.is_final_epoch = (current_epoch == max_epochs - 1)
        
        if self.is_final_epoch:
            self.log_message("Last epoch detected! Starting collection for confusion matrix")
            self.train_predictions = []
            self.train_ground_truths = []
    
    def on_validation_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs
        self.log_message(f"val_epoch_start - epoch {current_epoch+1}/{max_epochs}")
        
        # Check if this is the final epoch (in case validation is called separately)
        self.is_final_epoch = (current_epoch == max_epochs - 1)
        
        # Reset the lists at the start of the final validation epoch
        if self.is_final_epoch:
            self.log_message("Collecting validation data for confusion matrix")
            self.val_predictions = []
            self.val_ground_truths = []
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only collect during the final epoch
        if not self.is_final_epoch:
            return
        
        # Log information about the batch and outputs for debugging
        if batch_idx == 0:
            self.log_message(f"train_batch_end - batch {batch_idx}, outputs type: {type(outputs)}")
            if isinstance(outputs, dict):
                self.log_message(f"  Output keys: {list(outputs.keys())}")
            if isinstance(batch, dict):
                self.log_message(f"  Batch is dict with keys {list(batch.keys())}")
        
        try:
            # Forward pass to get predictions (since outputs typically just has the loss)
            with torch.no_grad():
                # Copy the batch to the same device as the model
                device = pl_module.device
                x = batch["image"].to(device)
                
                # Handle other inputs that might be needed for the forward pass
                other_keys = [k for k in batch.keys() if k not in ["image", "mask", "filename"]]
                rest = {k: batch[k].to(device) for k in other_keys}
                
                # Get model predictions
                model_output = pl_module(x, **rest)
                
                # Extract logits from outputs
                if hasattr(model_output, "output"):
                    logits = model_output.output
                elif isinstance(model_output, dict) and "output" in model_output:
                    logits = model_output["output"]
                else:
                    logits = model_output
                
                # Get class predictions
                pred = torch.argmax(logits, dim=1)
                
                # Extract ground truth
                mask = batch["mask"]
                
                # Store predictions and ground truths
                self.train_predictions.append(pred.detach().cpu())
                self.train_ground_truths.append(mask.detach().cpu())
        except Exception as e:
            self.log_message(f"Error in train_batch_end: {str(e)}")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Only collect during the final epoch
        if not self.is_final_epoch:
            return
        
        # Log information about the batch and outputs for debugging
        if batch_idx == 0:
            self.log_message(f"val_batch_end - batch {batch_idx}, outputs type: {type(outputs)}")
            if isinstance(outputs, dict):
                self.log_message(f"  Validation output keys: {list(outputs.keys())}")
            if isinstance(batch, dict):
                self.log_message(f"  Validation batch is dict with keys {list(batch.keys())}")
        
        try:
            # First, try to extract predictions from outputs (the normal way)
            pred = None
            
            # Check if outputs contains predictions directly
            if hasattr(outputs, "output"):
                logits = outputs.output
                pred = torch.argmax(logits, dim=1)
            elif isinstance(outputs, dict) and "output" in outputs:
                logits = outputs["output"]
                pred = torch.argmax(logits, dim=1)
            elif isinstance(outputs, dict) and "preds" in outputs:
                pred = outputs["preds"]
            
            # If we couldn't get predictions from outputs, do a forward pass
            if pred is None:
                self.log_message(f"  Could not extract predictions from validation outputs, doing forward pass")
                # Forward pass to get predictions
                with torch.no_grad():
                    # Copy the batch to the same device as the model
                    device = pl_module.device
                    x = batch["image"].to(device)
                    
                    # Handle other inputs that might be needed for the forward pass
                    other_keys = [k for k in batch.keys() if k not in ["image", "mask", "filename"]]
                    rest = {k: batch[k].to(device) for k in other_keys}
                    
                    # Get model predictions
                    model_output = pl_module(x, **rest)
                    
                    # Extract logits from outputs
                    if hasattr(model_output, "output"):
                        logits = model_output.output
                    elif isinstance(model_output, dict) and "output" in model_output:
                        logits = model_output["output"]
                    else:
                        logits = model_output
                    
                    # Get class predictions
                    pred = torch.argmax(logits, dim=1)
            
            # Extract ground truth
            mask = batch["mask"]
            
            # Store predictions and ground truths
            self.val_predictions.append(pred.detach().cpu())
            self.val_ground_truths.append(mask.detach().cpu())
            
            # Log the shapes for the first batch
            if batch_idx == 0:
                self.log_message(f"  Validation shapes - Predictions: {pred.shape}, Ground Truth: {mask.shape}")
                
        except Exception as e:
            self.log_message(f"Error in validation_batch_end: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
    
    def on_train_epoch_end(self, trainer, pl_module):
        self.log_message(f"train_epoch_end - epoch {trainer.current_epoch+1}/{trainer.max_epochs}")
        
        if not self.is_final_epoch:
            return
        
        self.log_message("Creating training confusion matrix")
        self._create_confusion_matrix(self.train_predictions, self.train_ground_truths, "train")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_message(f"val_epoch_end - epoch {trainer.current_epoch+1}/{trainer.max_epochs}")
        
        if not self.is_final_epoch:
            return
        
        self.log_message("Creating validation confusion matrix")
        self._create_confusion_matrix(self.val_predictions, self.val_ground_truths, "val")
    
    def on_fit_end(self, trainer, pl_module):
        self.log_message("fit_end - Training complete")
        self.log_message(f"Check for confusion matrices in: {self.save_dir}")
        self.log_message(f"Files in {self.save_dir}:")
        self.log_message("\n".join(os.listdir(self.save_dir)))
    
    def _create_confusion_matrix(self, predictions, ground_truths, prefix):
        """Create and save a confusion matrix from collected predictions and ground truths"""
        try:
            # Check if we have collected any predictions
            if not predictions or not ground_truths:
                self.log_message(f"Collected data: {len(predictions)} batches of outputs, {len(ground_truths)} batches of targets")
                self.log_message(f"No {prefix} data collected for confusion matrix.")
                return
            
            # Add additional debugging
            self.log_message(f"Creating {prefix} confusion matrix from {len(predictions)} prediction batches")
            for i, pred_batch in enumerate(predictions[:2]):  # Log info for first 2 batches only
                self.log_message(f"  Batch {i}: Prediction shape {pred_batch.shape}, dtype {pred_batch.dtype}")
                
            # Concatenate all tensors
            self.log_message(f"Concatenating {len(predictions)} tensors...")
            all_preds = torch.cat(predictions, dim=0)
            all_gt = torch.cat(ground_truths, dim=0)
            
            self.log_message(f"Concatenated tensor shapes - Predictions: {all_preds.shape}, Ground Truth: {all_gt.shape}")
            
            # Flatten if not already flattened
            all_preds = all_preds.numpy().flatten()
            all_gt = all_gt.numpy().flatten()
            
            self.log_message(f"Flattened shapes - Predictions: {all_preds.shape}, Ground Truth: {all_gt.shape}")
            
            # Filter out ignored indices (typically -1)
            valid_indices = (all_gt >= 0)
            all_preds = all_preds[valid_indices]
            all_gt = all_gt[valid_indices]
            
            self.log_message(f"After filtering ignored indices: {len(all_preds)} valid pixels")
            
            # Check if we have enough data
            if len(all_preds) == 0 or len(all_gt) == 0:
                self.log_message(f"After filtering, no valid {prefix} data for confusion matrix.")
                return
            
            # Calculate the confusion matrix
            cm = confusion_matrix(all_gt, all_preds)
            
            # Save raw confusion matrix
            np.save(os.path.join(self.save_dir, f"{prefix}_confusion_matrix.npy"), cm)
            
            # Create a visualization
            class_subset = self.class_names[:cm.shape[0]]
            
            # Calculate and display metrics
            self.log_message(f"{prefix.capitalize()} Confusion Matrix:")
            self.log_message(f"Shape: {cm.shape}")
            
            # Calculate per-class metrics
            for i, class_name in enumerate(class_subset):
                if np.sum(cm[i, :]) > 0:  # Avoid division by zero
                    precision = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
                    recall = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    self.log_message(f"Class {i} ({class_name}): Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            
            # Calculate overall accuracy
            accuracy = np.sum(np.diag(cm)) / np.sum(cm)
            self.log_message(f"Overall Accuracy: {accuracy:.4f}")
            
            # Plot and save the confusion matrix
            plt.figure(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_subset)
            disp.plot(cmap=plt.cm.Blues, values_format='d')
            plt.title(f"{prefix.capitalize()} Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"{prefix}_confusion_matrix.png"), dpi=300)
            plt.close()
            
            # Also save a normalized version
            plt.figure(figsize=(10, 8))
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_subset)
            disp.plot(cmap=plt.cm.Blues, values_format='.2f')
            plt.title(f"{prefix.capitalize()} Normalized Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"{prefix}_confusion_matrix_normalized.png"), dpi=300)
            plt.close()
            
            self.log_message(f"Successfully created {prefix} confusion matrix")
            
        except Exception as e:
            self.log_message(f"Error creating {prefix} confusion matrix: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())

# Need to import sys for executable and version information
import sys