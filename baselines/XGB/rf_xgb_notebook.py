# %% [markdown]
# # RF and XGBoost Benchmark for Irrigation Classification

# %% Imports
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

# XGBoost
from xgboost import XGBClassifier

# %% Configuration
TRAIN_DIR = 'training_chips'
VAL_DIR = 'validation_chips'
OUTPUT_DIR = 'benchmark_outputs'

CLASS_NAMES = ["Water", "Natural", "Irrigated", "Rainfed"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)

# %% Find training chips
train_images = sorted(glob.glob(os.path.join(TRAIN_DIR, '*_merged.tif')))
print(f"Found {len(train_images)} training chips")

# %% Load ONE chip to verify structure
test_img_path = train_images[0]
test_mask_path = test_img_path.replace('_merged.tif', '.mask.tif')

with rasterio.open(test_img_path) as src:
    test_img = src.read()
    print(f"Image shape: {test_img.shape}")  # (18, H, W)

with rasterio.open(test_mask_path) as src:
    test_mask = src.read(1)
    print(f"Mask shape: {test_mask.shape}")
    print(f"Unique values: {np.unique(test_mask)}")

# %% Load TRAINING data
all_X_train, all_y_train = [], []

for img_path in tqdm(train_images, desc="Loading training"):
    mask_path = img_path.replace('_merged.tif', '.mask.tif')
    if not os.path.exists(mask_path):
        continue
    
    with rasterio.open(img_path) as src:
        img = src.read()  # (18, H, W)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # (H, W)
    
    X = img.reshape(img.shape[0], -1).T  # (H*W, 18)
    y = mask.flatten()
    
    valid = (y >= 0) & (y < 4)
    all_X_train.append(X[valid])
    all_y_train.append(y[valid])

X_train = np.concatenate(all_X_train)
y_train = np.concatenate(all_y_train)
print(f"Training: {X_train.shape[0]:,} pixels, {X_train.shape[1]} features")

# %% Load VALIDATION data
val_images = sorted(glob.glob(os.path.join(VAL_DIR, '*_merged.tif')))
print(f"Found {len(val_images)} validation chips")

all_X_val, all_y_val = [], []

for img_path in tqdm(val_images, desc="Loading validation"):
    mask_path = img_path.replace('_merged.tif', '.mask.tif')
    if not os.path.exists(mask_path):
        continue
    
    with rasterio.open(img_path) as src:
        img = src.read()
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    
    X = img.reshape(img.shape[0], -1).T
    y = mask.flatten()
    
    valid = (y >= 0) & (y < 4)
    all_X_val.append(X[valid])
    all_y_val.append(y[valid])

X_val = np.concatenate(all_X_val)
y_val = np.concatenate(all_y_val)
print(f"Validation: {X_val.shape[0]:,} pixels")

# %% (Optional) Subsample if too large
MAX_PIXELS = 5_000_000

if len(y_train) > MAX_PIXELS:
    idx = np.random.choice(len(y_train), MAX_PIXELS, replace=False)
    X_train, y_train = X_train[idx], y_train[idx]
    print(f"Subsampled to {MAX_PIXELS:,}")

# %% Class distribution
unique, counts = np.unique(y_train, return_counts=True)
for c, n in zip(unique, counts):
    print(f"  {CLASS_NAMES[int(c)]}: {n:,} ({100*n/len(y_train):.1f}%)")

# %% ========== RANDOM FOREST ==========
print("\n" + "="*50)
print("Training Random Forest...")
print("="*50)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rf.fit(X_train, y_train)

# %% RF Evaluation
y_pred_rf = rf.predict(X_val)

print("\nRF Results:")
print(f"  Accuracy:     {accuracy_score(y_val, y_pred_rf):.4f}")
print(f"  Mean Jaccard: {jaccard_score(y_val, y_pred_rf, average='macro'):.4f}")
print(f"  Mean F1:      {f1_score(y_val, y_pred_rf, average='macro'):.4f}")

# %% ========== XGBOOST ==========
print("\n" + "="*50)
print("Training XGBoost...")
print("="*50)

# Calculate class weights for imbalanced data
class_counts = np.bincount(y_train.astype(int), minlength=4)
class_weights = len(y_train) / (4 * class_counts)
sample_weights = class_weights[y_train.astype(int)]

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=4,
    n_jobs=-1,
    random_state=42,
    verbosity=1
)
xgb.fit(X_train, y_train, sample_weight=sample_weights)

# %% XGBoost Evaluation
y_pred_xgb = xgb.predict(X_val)

print("\nXGBoost Results:")
print(f"  Accuracy:     {accuracy_score(y_val, y_pred_xgb):.4f}")
print(f"  Mean Jaccard: {jaccard_score(y_val, y_pred_xgb, average='macro'):.4f}")
print(f"  Mean F1:      {f1_score(y_val, y_pred_xgb, average='macro'):.4f}")

# %% Comparison Table
print("\n" + "="*50)
print("COMPARISON")
print("="*50)
print(f"{'Metric':<15} {'RF':>10} {'XGBoost':>10}")
print("-"*35)
print(f"{'Accuracy':<15} {accuracy_score(y_val, y_pred_rf):>10.4f} {accuracy_score(y_val, y_pred_xgb):>10.4f}")
print(f"{'Mean Jaccard':<15} {jaccard_score(y_val, y_pred_rf, average='macro'):>10.4f} {jaccard_score(y_val, y_pred_xgb, average='macro'):>10.4f}")
print(f"{'Mean F1':<15} {f1_score(y_val, y_pred_rf, average='macro'):>10.4f} {f1_score(y_val, y_pred_xgb, average='macro'):>10.4f}")

# Per-class
print("\nPer-class Jaccard:")
rf_jac = jaccard_score(y_val, y_pred_rf, average=None)
xgb_jac = jaccard_score(y_val, y_pred_xgb, average=None)
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:<10} {rf_jac[i]:>10.4f} {xgb_jac[i]:>10.4f}")

# %% Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay(confusion_matrix(y_val, y_pred_rf), display_labels=CLASS_NAMES).plot(ax=axes[0], cmap='Blues')
axes[0].set_title('Random Forest')

ConfusionMatrixDisplay(confusion_matrix(y_val, y_pred_xgb), display_labels=CLASS_NAMES).plot(ax=axes[1], cmap='Greens')
axes[1].set_title('XGBoost')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=300)
plt.show()

# %% Feature Importance Comparison
band_names = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
feature_names = [f"{b}_t{t+1}" for t in range(3) for b in band_names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RF importance
rf_imp = rf.feature_importances_
idx = np.argsort(rf_imp)[::-1]
axes[0].bar(range(18), rf_imp[idx])
axes[0].set_xticks(range(18))
axes[0].set_xticklabels([feature_names[i] for i in idx], rotation=45, ha='right')
axes[0].set_title('RF Feature Importance')

# XGB importance
xgb_imp = xgb.feature_importances_
idx = np.argsort(xgb_imp)[::-1]
axes[1].bar(range(18), xgb_imp[idx])
axes[1].set_xticks(range(18))
axes[1].set_xticklabels([feature_names[i] for i in idx], rotation=45, ha='right')
axes[1].set_title('XGBoost Feature Importance')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300)
plt.show()

# %% Save models and results
joblib.dump(rf, os.path.join(OUTPUT_DIR, 'rf_model.joblib'))
joblib.dump(xgb, os.path.join(OUTPUT_DIR, 'xgb_model.joblib'))

results = {
    'rf': {
        'accuracy': float(accuracy_score(y_val, y_pred_rf)),
        'mean_jaccard': float(jaccard_score(y_val, y_pred_rf, average='macro')),
        'mean_f1': float(f1_score(y_val, y_pred_rf, average='macro')),
    },
    'xgboost': {
        'accuracy': float(accuracy_score(y_val, y_pred_xgb)),
        'mean_jaccard': float(jaccard_score(y_val, y_pred_xgb, average='macro')),
        'mean_f1': float(f1_score(y_val, y_pred_xgb, average='macro')),
    }
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUTPUT_DIR}/")
# %% Visualize predictions for sample chips
import matplotlib.patches as mpatches

def plot_chip_prediction(img_path, mask_path, model, n_cols=5):
    """Plot T0, T1, T2, Ground Truth, Prediction for one chip"""
    
    with rasterio.open(img_path) as src:
        img = src.read()  # (18, H, W)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # (H, W)
    
    H, W = mask.shape
    
    # Extract RGB for each timestep (bands 0-5 = t1, 6-11 = t2, 12-17 = t3)
    # Assuming RGB = bands 2,1,0 (RED, GREEN, BLUE)
    t0_rgb = np.moveaxis(img[2::-1], 0, -1)  # (H, W, 3)
    t1_rgb = np.moveaxis(img[8:5:-1], 0, -1)
    t2_rgb = np.moveaxis(img[14:11:-1], 0, -1)
    
    # Normalize for display
    def normalize(arr):
        arr = arr.astype(float)
        p2, p98 = np.percentile(arr, (2, 98))
        return np.clip((arr - p2) / (p98 - p2 + 1e-6), 0, 1)
    
    t0_rgb = normalize(t0_rgb)
    t1_rgb = normalize(t1_rgb)
    t2_rgb = normalize(t2_rgb)
    
    # Predict
    X = img.reshape(18, -1).T  # (H*W, 18)
    y_pred = model.predict(X).reshape(H, W)
    
    # Color map for classes
    colors = ['#2196F3', '#8B4513', '#228B22', '#90EE90']  # Water, non-Ag, Irrigated, Rainfed
    cmap = plt.cm.colors.ListedColormap(colors)
    
    return t0_rgb, t1_rgb, t2_rgb, mask, y_pred, cmap


# Select random chips to visualize
n_chips = 5
sample_indices = np.random.choice(len(val_images), n_chips, replace=False)

fig, axes = plt.subplots(n_chips, 5, figsize=(15, 3*n_chips))

# Color map
colors = ['#2196F3', '#8B4513', '#228B22', '#90EE90']
cmap = plt.cm.colors.ListedColormap(colors)

for row, idx in enumerate(sample_indices):
    img_path = val_images[idx]
    mask_path = img_path.replace('_merged.tif', '.mask.tif')
    
    with rasterio.open(img_path) as src:
        img = src.read()
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    
    H, W = mask.shape
    
    # RGB for each timestep (R=2, G=1, B=0)
    t0 = np.moveaxis(img[2::-1], 0, -1)
    t1 = np.moveaxis(img[8:5:-1], 0, -1)
    t2 = np.moveaxis(img[14:11:-1], 0, -1)
    
    # Normalize
    def norm(arr):
        p2, p98 = np.percentile(arr, (2, 98))
        return np.clip((arr - p2) / (p98 - p2 + 1e-6), 0, 1)
    
    t0, t1, t2 = norm(t0), norm(t1), norm(t2)
    
    # Predict (using RF, change to xgb if preferred)
    X = img.reshape(18, -1).T
    pred = rf.predict(X).reshape(H, W)
    
    # Plot
    axes[row, 0].imshow(t0)
    axes[row, 1].imshow(t1)
    axes[row, 2].imshow(t2)
    axes[row, 3].imshow(mask, cmap=cmap, vmin=0, vmax=3)
    axes[row, 4].imshow(pred, cmap=cmap, vmin=0, vmax=3)
    
    for ax in axes[row]:
        ax.axis('off')

# Titles
axes[0, 0].set_title('T0')
axes[0, 1].set_title('T1')
axes[0, 2].set_title('T2')
axes[0, 3].set_title('Ground Truth')
axes[0, 4].set_title('Predicted')

# Legend
patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, CLASS_NAMES)]
fig.legend(handles=patches, loc='center left', bbox_to_anchor=(0.01, 0.5))

plt.tight_layout()
plt.subplots_adjust(left=0.08)
plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_samples.png'), dpi=300, bbox_inches='tight')
plt.show()
