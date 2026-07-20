"""
================================================================
  HyperGeo — Real Experiment Runner  (FIXED VERSION)
  Run this to get TRUE results for your IEEE paper.
================================================================

HOW TO RUN:
    1. Open Command Prompt in your D:\HyperGEO folder
    2. Run:  pip install numpy scipy scikit-learn matplotlib requests Pillow
    3. Run:  python run_real_experiment.py

The script will:
    - Download Indian Pines from a working mirror automatically
    - Run the full HyperGeo preprocessing pipeline
    - Train Random Forest and test on real ground truth
    - Print every number needed for your IEEE paper
    - Save results/hypergeo_real_results.json
    - Save results/mineral_map.png
================================================================
"""

import numpy as np
import scipy.io as sio
import os, sys, time, json, requests
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

os.makedirs('data',    exist_ok=True)
os.makedirs('results', exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Download Indian Pines (tries multiple mirrors)
# ═════════════════════════════════════════════════════════════════════════════

# Working mirrors — tried in order until one succeeds
DATA_MIRRORS = [
    # GitHub mirror 1
    "https://github.com/GiorgioMorales/HSI-BandSelection/raw/master/Data/Indian_pines_corrected.mat",
    # GitHub mirror 2
    "https://github.com/theholymath/hyperspectral_data/raw/master/Indian_pines_corrected.mat",
    # Hugging Face
    "https://huggingface.co/datasets/danaroth/indian_pines/resolve/main/Indian_pines_corrected.mat",
    # Original (blocked by some networks, worth a retry)
    "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
]

GT_MIRRORS = [
    "https://github.com/GiorgioMorales/HSI-BandSelection/raw/master/Data/Indian_pines_gt.mat",
    "https://github.com/theholymath/hyperspectral_data/raw/master/Indian_pines_gt.mat",
    "https://huggingface.co/datasets/danaroth/indian_pines/resolve/main/Indian_pines_gt.mat",
    "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
]

DATA_FILE = "data/Indian_pines_corrected.mat"
GT_FILE   = "data/Indian_pines_gt.mat"


def download_with_mirrors(mirrors, dest):
    """Try each mirror in order until one works."""
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if size > 100_000:   # file exists and is not empty/corrupted
            print(f"[Download] Already downloaded: {dest}  ({size//1024} KB)")
            return True
        else:
            print(f"[Download] Found incomplete file, re-downloading...")
            os.remove(dest)

    filename = os.path.basename(dest)
    for i, url in enumerate(mirrors):
        try:
            print(f"[Download] Trying mirror {i+1}/{len(mirrors)}: {url[:70]}...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                              'AppleWebKit/537.36 Chrome/120.0.0.0'
            }
            r = requests.get(url, headers=headers, stream=True, timeout=60)
            r.raise_for_status()

            total = int(r.headers.get('content-length', 0))
            downloaded = 0
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=32768):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  Progress: {pct:.1f}%  ({downloaded//1024} KB)", end='')

            print(f"\n[Download] ✓ Saved: {dest}  ({os.path.getsize(dest)//1024} KB)")
            return True

        except Exception as e:
            print(f"[Download] Mirror {i+1} failed: {e}")
            if os.path.exists(dest):
                os.remove(dest)
            continue

    return False


print("=" * 65)
print("  HyperGeo — Real Experiment Runner")
print("=" * 65)
print("\n[Step 1] Downloading Indian Pines dataset...")

ok1 = download_with_mirrors(DATA_MIRRORS, DATA_FILE)
ok2 = download_with_mirrors(GT_MIRRORS,   GT_FILE)

if not ok1 or not ok2:
    print("\n" + "="*65)
    print("  DOWNLOAD FAILED FROM ALL MIRRORS")
    print("="*65)
    print("""
  Please download the files MANUALLY:

  1. Open your browser and go to:
     https://github.com/GiorgioMorales/HSI-BandSelection/tree/master/Data

  2. Download these two files:
       Indian_pines_corrected.mat
       Indian_pines_gt.mat

  3. Place them inside:
       D:\\HyperGEO\\data\\

  4. Then run this script again.
""")
    sys.exit(1)

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Load Data
# ═════════════════════════════════════════════════════════════════════════════

print("\n[Step 2] Loading dataset...")
mat   = sio.loadmat(DATA_FILE)
gt_m  = sio.loadmat(GT_FILE)

# The corrected dataset variable name
cube  = mat['indian_pines_corrected'].astype(np.float32)   # (145,145,200)
gt    = gt_m['indian_pines_gt'].astype(np.int32)           # (145,145)

rows, cols, bands = cube.shape
print(f"  Cube shape : {cube.shape}")
print(f"  GT shape   : {gt.shape}")
print(f"  Value range: [{cube.min():.1f}, {cube.max():.1f}]")

CLASS_NAMES = {
    1: 'Alfalfa',              2: 'Corn-notill',
    3: 'Corn-mintill',         4: 'Corn',
    5: 'Grass-pasture',        6: 'Grass-trees',
    7: 'Grass-pasture-mowed',  8: 'Hay-windrowed',
    9: 'Oats',                10: 'Soybean-notill',
   11: 'Soybean-mintill',     12: 'Soybean-clean',
   13: 'Wheat',               14: 'Woods',
   15: 'Buildings-Grass-Drives', 16: 'Stone-Steel-Towers',
}

print(f"\n  Ground truth class distribution:")
for c in range(1, 17):
    n = int(np.sum(gt == c))
    print(f"    Class {c:2d} — {CLASS_NAMES[c]:<28}: {n:4d} pixels")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Preprocessing Pipeline (identical to HyperGeo)
# ═════════════════════════════════════════════════════════════════════════════

print("\n[Step 3] Running 4-stage preprocessing pipeline...")
t_pre = time.time()

# ── Stage 1: Min-Max Normalisation ───────────────────────────────────────────
print("  Stage 1/4 — Normalisation...")
t1 = time.time()
flat      = cube.reshape(-1, bands)
scaler    = MinMaxScaler()
flat_norm = scaler.fit_transform(flat)
cube_norm = flat_norm.reshape(rows, cols, bands).astype(np.float32)
print(f"    Done in {time.time()-t1:.2f}s  |  range: [0.000, 1.000]")

# ── Stage 2: Bad Band Removal ────────────────────────────────────────────────
# The corrected dataset already has water-vapour bands removed.
# We use all 200 bands.
print("  Stage 2/4 — Bad band removal...")
t2 = time.time()
cube_bb     = cube_norm          # all 200 bands are clean in corrected version
bands_clean = bands
print(f"    Done in {time.time()-t2:.2f}s  |  {bands} → {bands_clean} bands (already corrected)")

# ── Stage 3: Savitzky-Golay Smoothing ────────────────────────────────────────
print("  Stage 3/4 — Savitzky-Golay smoothing...")
t3       = time.time()
window   = 11
poly_deg = 3
flat_bb     = cube_bb.reshape(-1, bands_clean)
flat_smooth = savgol_filter(flat_bb, window_length=window,
                            polyorder=poly_deg, axis=1)
cube_smooth = flat_smooth.reshape(rows, cols, bands_clean).astype(np.float32)
print(f"    Done in {time.time()-t3:.2f}s  |  window={window}, degree={poly_deg}")

# ── Stage 4: PCA ─────────────────────────────────────────────────────────────
print("  Stage 4/4 — PCA dimensionality reduction...")
t4           = time.time()
N_COMPONENTS = 10
flat_s   = cube_smooth.reshape(-1, bands_clean)
pca      = PCA(n_components=N_COMPONENTS, random_state=42)
flat_pca = pca.fit_transform(flat_s)
cube_pca = flat_pca.reshape(rows, cols, N_COMPONENTS).astype(np.float32)
var_each = pca.explained_variance_ratio_ * 100
var_cum  = np.cumsum(var_each)
print(f"    Done in {time.time()-t4:.2f}s  |  {bands_clean} bands → {N_COMPONENTS} PCs")
print(f"    Variance retained: {var_cum[N_COMPONENTS-1]:.2f}%")
print(f"    Per-PC variance: {[f'{v:.2f}%' for v in var_each]}")

t_pre_total = time.time() - t_pre
print(f"\n  Preprocessing total: {t_pre_total:.2f}s")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Train/Test Split on Real Ground Truth
# ═════════════════════════════════════════════════════════════════════════════

print("\n[Step 4] Preparing ground truth train/test split...")

flat_pca_2d = cube_pca.reshape(-1, N_COMPONENTS)
gt_flat     = gt.flatten()

# Labelled pixels only (exclude background = 0)
mask   = gt_flat > 0
X_lab  = flat_pca_2d[mask]
y_lab  = gt_flat[mask]

print(f"  Total labelled pixels : {len(X_lab)}")

# Standard 80/20 stratified split — reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X_lab, y_lab,
    test_size=0.20,
    random_state=42,
    stratify=y_lab
)
print(f"  Training samples (80%): {len(X_train)}")
print(f"  Test samples     (20%): {len(X_test)}")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Train Random Forest
# ═════════════════════════════════════════════════════════════════════════════

print("\n[Step 5] Training Random Forest (200 trees, all CPU cores)...")
t_train_start = time.time()

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    verbose=0
)
rf.fit(X_train, y_train)

t_train = time.time() - t_train_start
print(f"  Training time: {t_train:.2f}s")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Predict and Compute All Metrics
# ═════════════════════════════════════════════════════════════════════════════

print("\n[Step 6] Computing metrics...")
t_pred_start = time.time()
y_pred = rf.predict(X_test)
t_pred = time.time() - t_pred_start
print(f"  Prediction time: {t_pred:.2f}s")

# Overall metrics
OA    = accuracy_score(y_test, y_pred) * 100
kappa = cohen_kappa_score(y_test, y_pred)

# Average Accuracy = mean per-class accuracy
classes = sorted(np.unique(y_test))
per_class_acc = []
for c in classes:
    m   = (y_test == c)
    acc = accuracy_score(y_test[m], y_pred[m]) * 100
    per_class_acc.append(acc)
AA = float(np.mean(per_class_acc))

# Per-class P, R, F1
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=classes, zero_division=0
)
mean_P  = float(np.mean(precision)) * 100
mean_R  = float(np.mean(recall))    * 100
mean_F1 = float(np.mean(f1))

t_total = t_pre_total + t_train + t_pred

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Generate and Save Mineral Map
# ═════════════════════════════════════════════════════════════════════════════

print("\n[Step 7] Generating full mineral classification map...")
t_map = time.time()

# Predict every pixel in the scene
y_all_pred = rf.predict(flat_pca_2d)
label_map  = y_all_pred.reshape(rows, cols)

# 17-colour palette (index 0 = black background)
PALETTE = [
    (0,   0,   0),     # 0  Background
    (255, 0,   0),     # 1  Alfalfa
    (0,   200, 0),     # 2  Corn-notill
    (0,   0,   255),   # 3  Corn-mintill
    (255, 255, 0),     # 4  Corn
    (255, 0,   255),   # 5  Grass-pasture
    (0,   255, 255),   # 6  Grass-trees
    (128, 0,   0),     # 7  Grass-pasture-mowed
    (0,   128, 0),     # 8  Hay-windrowed
    (0,   0,   128),   # 9  Oats
    (200, 200, 0),     # 10 Soybean-notill
    (200, 0,   200),   # 11 Soybean-mintill
    (0,   200, 200),   # 12 Soybean-clean
    (255, 128, 0),     # 13 Wheat
    (0,   255, 128),   # 14 Woods
    (128, 0,   255),   # 15 Buildings-Grass-Drives
    (255, 200, 128),   # 16 Stone-Steel-Towers
]

rgb_map = np.zeros((rows, cols, 3), dtype=np.uint8)
for idx, color in enumerate(PALETTE):
    rgb_map[label_map == idx] = color

try:
    from PIL import Image
    Image.fromarray(rgb_map, 'RGB').save('results/mineral_map.png')
except ImportError:
    import matplotlib.pyplot as plt
    plt.imsave('results/mineral_map.png', rgb_map)

print(f"  Saved: results/mineral_map.png  (time: {time.time()-t_map:.2f}s)")

# Also save the ground truth map for comparison
gt_rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
for idx, color in enumerate(PALETTE):
    gt_rgb[gt == idx] = color
try:
    from PIL import Image
    Image.fromarray(gt_rgb, 'RGB').save('results/ground_truth_map.png')
    print(f"  Saved: results/ground_truth_map.png")
except Exception:
    pass

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 8 — Save JSON Results
# ═════════════════════════════════════════════════════════════════════════════

results_dict = {
    "dataset"     : "Indian Pines corrected (200 bands)",
    "split"       : "80% train / 20% test, stratified, random_state=42",
    "n_train"     : int(len(X_train)),
    "n_test"      : int(len(X_test)),
    "OA_pct"      : round(OA,    2),
    "AA_pct"      : round(AA,    2),
    "kappa"       : round(kappa, 4),
    "mean_precision_pct" : round(mean_P,  2),
    "mean_recall_pct"    : round(mean_R,  2),
    "mean_f1"            : round(mean_F1, 4),
    "pca_n_components"   : N_COMPONENTS,
    "pca_cumvar_pct"     : round(float(var_cum[N_COMPONENTS-1]), 2),
    "pca_per_component"  : [round(float(v), 2) for v in var_each],
    "timing_seconds": {
        "preprocessing": round(t_pre_total, 2),
        "rf_training"  : round(t_train,     2),
        "rf_prediction": round(t_pred,      2),
        "total"        : round(t_total,     2),
    },
    "per_class": {}
}

for i, c in enumerate(classes):
    results_dict["per_class"][CLASS_NAMES[c]] = {
        "precision_pct": round(float(precision[i]) * 100, 2),
        "recall_pct"   : round(float(recall[i])    * 100, 2),
        "f1_pct"       : round(float(f1[i])        * 100, 2),
        "support"      : int(support[i]),
    }

with open('results/hypergeo_real_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)
print("\n  Saved: results/hypergeo_real_results.json")

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 9 — Print Everything for the IEEE Paper
# ═════════════════════════════════════════════════════════════════════════════

SEP = "=" * 65

print(f"""
{SEP}
  REAL RESULTS  —  Replace estimated values in IEEE paper
{SEP}

  TABLE IV — Overall Classification Performance:
  ┌──────────────────────────────────┬───────────────┐
  │ Metric                           │ Value         │
  ├──────────────────────────────────┼───────────────┤
  │ Overall Accuracy (OA)            │ {OA:.2f}%        │
  │ Average Accuracy (AA)            │ {AA:.2f}%        │
  │ Kappa Coefficient (κ)            │ {kappa:.4f}       │
  │ Mean Precision                   │ {mean_P:.2f}%        │
  │ Mean Recall                      │ {mean_R:.2f}%        │
  │ Mean F1-Score                    │ {mean_F1:.4f}       │
  │ RF Training Time                 │ {t_train:.1f}s          │
  └──────────────────────────────────┴───────────────┘

  TABLE III — Preprocessing:
    PCA cumulative variance @ 10 PCs : {var_cum[N_COMPONENTS-1]:.2f}%

  TABLE VI — Timing Breakdown:
    Preprocessing                    : {t_pre_total:.1f}s
    RF Training (200 trees)          : {t_train:.1f}s
    RF Prediction (21,025 pixels)    : {t_pred:.1f}s
    TOTAL pipeline                   : {t_total:.1f}s

  PCA Variance per component (for Figure 6):""")

for i, (v, cv) in enumerate(zip(var_each, var_cum)):
    print(f"    PC{i+1:2d}: {v:6.2f}%  (cumulative: {cv:.2f}%)")

print(f"""
  TABLE V — Per-class Precision, Recall, F1:
  {'Class':<30} {'Prec%':>7} {'Rec%':>7} {'F1%':>7} {'Support':>9}
  {'-'*58}""")
for i, c in enumerate(classes):
    print(f"  {CLASS_NAMES[c]:<30} {precision[i]*100:7.2f} {recall[i]*100:7.2f} {f1[i]*100:7.2f} {support[i]:9d}")

print(f"""
{SEP}
  FILES SAVED:
    results/hypergeo_real_results.json  ← all numbers in JSON
    results/mineral_map.png             ← real classification map
    results/ground_truth_map.png        ← ground truth for comparison
{SEP}

  NEXT STEPS:
    1. Copy the numbers above into your IEEE paper tables
    2. Replace Fig 9 in the paper with: results/mineral_map.png
    3. Take a screenshot of HyperGeo GUI with Indian Pines loaded
       and replace Fig 5 in the paper with that screenshot
    4. Update Fig 6 (PCA bar chart) using the per-PC values above
    5. You are ready to submit to IEEE!
{SEP}
""")
