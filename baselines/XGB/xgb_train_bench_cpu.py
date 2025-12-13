# xgb_benchmark.py
import os
import glob
import numpy as np
import rasterio
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import json

# Config
BASE_DIR = '/discover/nobackup/ejalilva/data/prithvi/datasets--ibm-nasa-geospatial--multi-temporal-irrigation-classificaction-openet/snapshots/04b439f179e52a7b144f69676210eecd30c39cfc/'
OUTPUT_DIR = 'xgb_outputs'
CLASS_NAMES = ["Water", "Natural", "Irrigated", "Rainfed"]
MAX_PIXELS = 40_000_000

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)

def load_data(data_dir):
    images = sorted(glob.glob(os.path.join(data_dir, '*_merged.tif')))
    all_X, all_y = [], []
    for img_path in tqdm(images, desc=f"Loading {os.path.basename(data_dir)}"):
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
        all_X.append(X[valid])
        all_y.append(y[valid])
    return np.concatenate(all_X), np.concatenate(all_y)

X_train, y_train = load_data(os.path.join(BASE_DIR, 'training_chips'))
X_val, y_val = load_data(os.path.join(BASE_DIR, 'validation_chips'))

if len(y_train) > MAX_PIXELS:
    idx = np.random.choice(len(y_train), MAX_PIXELS, replace=False)
    X_train, y_train = X_train[idx], y_train[idx]
print(f"Training: {len(y_train):,} pixels")

class_counts = np.bincount(y_train.astype(int), minlength=4)
sample_weights = (len(y_train) / (4 * class_counts))[y_train.astype(int)]

# CPU version - still fast!
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=4,
    tree_method='hist',  # Fast histogram method
    device='cpu',        # Explicit CPU
    n_jobs=-1,           # All cores
    random_state=42,
    verbosity=1
)
xgb.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = xgb.predict(X_val)
results = {
    'accuracy': float(accuracy_score(y_val, y_pred)),
    'mean_jaccard': float(jaccard_score(y_val, y_pred, average='macro')),
    'mean_f1': float(f1_score(y_val, y_pred, average='macro')),
    'per_class_jaccard': [float(j) for j in jaccard_score(y_val, y_pred, average=None)]
}
print(f"\nAccuracy: {results['accuracy']:.4f}, Jaccard: {results['mean_jaccard']:.4f}, F1: {results['mean_f1']:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix(y_val, y_pred), display_labels=CLASS_NAMES).plot(ax=ax, cmap='Greens')
ax.set_title(f"XGBoost\nJaccard: {results['mean_jaccard']:.4f}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_confusion_matrix.png'), dpi=300)
plt.close()

joblib.dump(xgb, os.path.join(OUTPUT_DIR, 'xgb_model.joblib'))
with open(os.path.join(OUTPUT_DIR, 'xgb_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUTPUT_DIR}/")