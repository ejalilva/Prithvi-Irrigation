#!/usr/bin/env python
"""
Simple Random Forest Benchmark
Just rasterio + numpy + sklearn. No PyTorch.
"""

import os
import glob
import numpy as np
import rasterio
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, jaccard_score, f1_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import joblib
import json

# ============== CONFIGURATION ==============
TRAIN_DIR = 'training_chips'
VAL_DIR = 'validation_chips'
OUTPUT_DIR = 'rf_outputs'

CLASS_NAMES = ["Water", "Natural", "Irrigated", "Rainfed"]

RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': 42,
    'verbose': 1
}

MAX_TRAIN_PIXELS = 5_000_000  # Set to None for all pixels


def load_chips(directory):
    """Load all chips from directory. Returns X (n_pixels, 18), y (n_pixels,)"""
    
    # Find all merged tifs
    image_files = sorted(glob.glob(os.path.join(directory, '*_merged.tif')))
    
    if not image_files:
        # Try other patterns
        image_files = sorted(glob.glob(os.path.join(directory, '*.tif')))
        image_files = [f for f in image_files if 'mask' not in f]
    
    print(f"  Found {len(image_files)} chips")
    
    all_X, all_y = [], []
    
    for img_path in tqdm(image_files):
        # Find corresponding mask
        mask_path = img_path.replace('_merged.tif', '.mask.tif')
        if not os.path.exists(mask_path):
            mask_path = img_path.replace('.tif', '_mask.tif')
        if not os.path.exists(mask_path):
            mask_path = img_path.replace('.tif', '.mask.tif')
        
        if not os.path.exists(mask_path):
            print(f"  Warning: No mask for {img_path}")
            continue
        
        # Read image (18 bands)
        with rasterio.open(img_path) as src:
            img = src.read()  # (18, H, W)
        
        # Read mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # (H, W)
        
        # Reshape: (18, H, W) -> (H*W, 18)
        n_bands, H, W = img.shape
        X = img.reshape(n_bands, -1).T  # (H*W, 18)
        y = mask.flatten()  # (H*W,)
        
        # Keep only valid pixels (0-3)
        valid = (y >= 0) & (y < len(CLASS_NAMES))
        X = X[valid]
        y = y[valid]
        
        all_X.append(X)
        all_y.append(y)
    
    return np.concatenate(all_X), np.concatenate(all_y)


def main():
    print("=" * 50)
    print("Random Forest Benchmark")
    print("=" * 50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)
    
    # Load data
    print(f"\nLoading training data from {TRAIN_DIR}...")
    X_train, y_train = load_chips(TRAIN_DIR)
    
    print(f"\nLoading validation data from {VAL_DIR}...")
    X_val, y_val = load_chips(VAL_DIR)
    
    # Subsample if needed
    if MAX_TRAIN_PIXELS and len(y_train) > MAX_TRAIN_PIXELS:
        print(f"\nSubsampling train: {len(y_train):,} -> {MAX_TRAIN_PIXELS:,}")
        idx = np.random.choice(len(y_train), MAX_TRAIN_PIXELS, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    
    print(f"\nData loaded:")
    print(f"  Train: {len(y_train):,} pixels, {X_train.shape[1]} features")
    print(f"  Val:   {len(y_val):,} pixels")
    print(f"  Classes: {dict(zip(CLASS_NAMES, np.bincount(y_train.astype(int), minlength=4)))}")
    
    # Train
    print(f"\nTraining RF...")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    
    # Save model
    joblib.dump(rf, os.path.join(OUTPUT_DIR, 'rf_model.joblib'))
    
    # Evaluate
    print("\nEvaluating...")
    y_pred = rf.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    jaccard = jaccard_score(y_val, y_pred, average='macro')
    f1 = f1_score(y_val, y_pred, average='macro')
    
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Mean Jaccard: {jaccard:.4f}")
    print(f"Mean F1:      {f1:.4f}")
    
    # Per-class
    per_class = jaccard_score(y_val, y_pred, average=None)
    print(f"\nPer-class Jaccard:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {per_class[i]:.4f}")
    
    # Classification report
    print(f"\n{classification_report(y_val, y_pred, target_names=CLASS_NAMES, digits=4)}")
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap='Blues')
    plt.title('Validation Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Save results
    results = {
        'accuracy': float(acc),
        'mean_jaccard': float(jaccard),
        'mean_f1': float(f1),
        'per_class_jaccard': {CLASS_NAMES[i]: float(per_class[i]) for i in range(4)},
        'train_pixels': len(y_train),
        'val_pixels': len(y_val),
    }
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
