"""
Hyperspectral Classification Engine
Methods: SAM (rule-based), SVM, Random Forest
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from typing import Optional, Tuple

from core.spectral_db import SpectralDatabase, spectral_db


class ClassificationResult:
    def __init__(self):
        self.label_map = None        # (rows, cols) int  — class index per pixel
        self.label_names = []        # list of class name strings
        self.label_colors = []       # list of hex color strings
        self.confidence_map = None   # (rows, cols) float 0-1
        self.method = ""
        self.report = ""

    @property
    def color_map(self):
        """Return (rows, cols, 3) uint8 RGB image of the mineral map."""
        if self.label_map is None:
            return None
        rows, cols = self.label_map.shape
        rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
        for idx, hex_color in enumerate(self.label_colors):
            mask = self.label_map == idx
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            rgb[mask] = [r, g, b]
        return rgb


class Classifier:
    """Unified classifier supporting SAM, SVM, and Random Forest."""

    def __init__(self, db: SpectralDatabase = None):
        self.db = db or spectral_db
        self._svm_model = None
        self._rf_model = None
        self._le = LabelEncoder()
        self._trained_wavelengths = None

    # ── SAM Classification ────────────────────────────────────────────────────

    def classify_sam(self, cube: np.ndarray, wavelengths: np.ndarray,
                     threshold_deg: float = 10.0,
                     progress_callback=None) -> ClassificationResult:
        """
        Spectral Angle Mapper — no training required.
        threshold_deg: pixels with min angle > threshold are labelled 'Unknown'.
        """
        rows, cols, bands = cube.shape
        ref_matrix, names, colors = self.db.build_reference_matrix(wavelengths)

        if progress_callback:
            progress_callback(10, "Building reference matrix...")

        flat = cube.reshape(-1, bands).astype(np.float32)

        # Vectorised SAM
        norms_flat = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-10
        norms_ref = np.linalg.norm(ref_matrix, axis=1, keepdims=True) + 1e-10

        flat_n = flat / norms_flat               # (N, bands)
        ref_n = ref_matrix / norms_ref           # (M, bands)

        dot = flat_n @ ref_n.T                   # (N, M)
        dot = np.clip(dot, -1, 1)
        angles = np.degrees(np.arccos(dot))      # (N, M)  — degrees

        if progress_callback:
            progress_callback(70, "Mapping mineral classes...")

        best_idx = np.argmin(angles, axis=1)
        best_angle = angles[np.arange(len(flat)), best_idx]

        # Unknown class
        label_names = names + ['Unknown']
        label_colors = colors + ['#2C2C2C']

        labels = best_idx.copy()
        labels[best_angle > threshold_deg] = len(names)  # unknown

        confidence = 1.0 - (best_angle / 90.0)
        confidence = np.clip(confidence, 0, 1)

        result = ClassificationResult()
        result.label_map = labels.reshape(rows, cols).astype(np.int32)
        result.confidence_map = confidence.reshape(rows, cols).astype(np.float32)
        result.label_names = label_names
        result.label_colors = label_colors
        result.method = "SAM"

        if progress_callback:
            progress_callback(100, "SAM classification complete")

        return result

    # ── Training data generation from reference library ───────────────────────

    def generate_training_data(self, wavelengths: np.ndarray,
                                n_samples_per_class: int = 300,
                                noise_level: float = 0.03):
        """
        Synthesise training data by adding Gaussian noise to reference spectra.
        Returns (X, y_encoded, label_names).
        """
        X_list, y_list = [], []
        names = self.db.get_names()

        for name in names:
            ref = self.db.interpolate_to(name, wavelengths)
            for _ in range(n_samples_per_class):
                noise = np.random.normal(0, noise_level, ref.shape).astype(np.float32)
                sample = np.clip(ref + noise, 0, 1)
                X_list.append(sample)
                y_list.append(name)

        X = np.array(X_list, dtype=np.float32)
        y_raw = np.array(y_list)
        y_enc = self._le.fit_transform(y_raw)
        self._trained_wavelengths = wavelengths.copy()
        return X, y_enc, list(self._le.classes_)

    # ── SVM Classification ────────────────────────────────────────────────────

    def train_svm(self, wavelengths: np.ndarray, progress_callback=None):
        if progress_callback:
            progress_callback(10, "Generating training data...")
        X, y, names = self.generate_training_data(wavelengths)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        if progress_callback:
            progress_callback(40, "Training SVM (this may take a moment)...")
        self._svm_model = SVC(kernel='rbf', C=10, gamma='scale',
                               probability=True, random_state=42)
        self._svm_model.fit(X_tr, y_tr)

        if progress_callback:
            progress_callback(80, "Evaluating SVM...")
        y_pred = self._svm_model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        report = classification_report(y_te, y_pred, target_names=names)

        if progress_callback:
            progress_callback(100, f"SVM trained — accuracy: {acc:.1%}")
        return acc, report, names

    def classify_svm(self, cube: np.ndarray, wavelengths: np.ndarray,
                     progress_callback=None) -> ClassificationResult:
        if self._svm_model is None:
            self.train_svm(wavelengths, progress_callback)

        rows, cols, bands = cube.shape
        flat = cube.reshape(-1, bands).astype(np.float32)

        if progress_callback:
            progress_callback(70, "Running SVM prediction...")

        probs = self._svm_model.predict_proba(flat)
        pred_enc = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)

        names = list(self._le.classes_)
        colors = [self.db.minerals[n].color if n in self.db.minerals else '#888888'
                  for n in names]

        result = ClassificationResult()
        result.label_map = pred_enc.reshape(rows, cols).astype(np.int32)
        result.confidence_map = confidence.reshape(rows, cols).astype(np.float32)
        result.label_names = names
        result.label_colors = colors
        result.method = "SVM"

        if progress_callback:
            progress_callback(100, "SVM classification complete")
        return result

    # ── Random Forest Classification ──────────────────────────────────────────

    def train_rf(self, wavelengths: np.ndarray, progress_callback=None):
        if progress_callback:
            progress_callback(10, "Generating training data...")
        X, y, names = self.generate_training_data(wavelengths)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        if progress_callback:
            progress_callback(40, "Training Random Forest...")
        self._rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=None,
            n_jobs=-1, random_state=42
        )
        self._rf_model.fit(X_tr, y_tr)

        if progress_callback:
            progress_callback(80, "Evaluating Random Forest...")
        y_pred = self._rf_model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        report = classification_report(y_te, y_pred, target_names=names)

        if progress_callback:
            progress_callback(100, f"RF trained — accuracy: {acc:.1%}")
        return acc, report, names

    def classify_rf(self, cube: np.ndarray, wavelengths: np.ndarray,
                    progress_callback=None) -> ClassificationResult:
        if self._rf_model is None:
            self.train_rf(wavelengths, progress_callback)

        rows, cols, bands = cube.shape
        flat = cube.reshape(-1, bands).astype(np.float32)

        if progress_callback:
            progress_callback(70, "Running Random Forest prediction...")

        probs = self._rf_model.predict_proba(flat)
        pred_enc = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)

        names = list(self._le.classes_)
        colors = [self.db.minerals[n].color if n in self.db.minerals else '#888888'
                  for n in names]

        result = ClassificationResult()
        result.label_map = pred_enc.reshape(rows, cols).astype(np.int32)
        result.confidence_map = confidence.reshape(rows, cols).astype(np.float32)
        result.label_names = names
        result.label_colors = colors
        result.method = "Random Forest"

        if progress_callback:
            progress_callback(100, "Random Forest classification complete")
        return result

    # ── Unified entry point ───────────────────────────────────────────────────

    def classify(self, cube: np.ndarray, wavelengths: np.ndarray,
                 method: str = 'rf', progress_callback=None) -> ClassificationResult:
        """
        method: 'sam' | 'svm' | 'rf'
        """
        method = method.lower()
        if method == 'sam':
            return self.classify_sam(cube, wavelengths, progress_callback=progress_callback)
        elif method == 'svm':
            return self.classify_svm(cube, wavelengths, progress_callback=progress_callback)
        elif method in ('rf', 'random_forest', 'randomforest'):
            return self.classify_rf(cube, wavelengths, progress_callback=progress_callback)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'sam', 'svm', or 'rf'.")

    # ── Save / Load models ────────────────────────────────────────────────────

    def save_models(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        if self._svm_model:
            joblib.dump(self._svm_model, os.path.join(directory, 'svm_model.pkl'))
        if self._rf_model:
            joblib.dump(self._rf_model, os.path.join(directory, 'rf_model.pkl'))
        if self._trained_wavelengths is not None:
            np.save(os.path.join(directory, 'wavelengths.npy'), self._trained_wavelengths)
        joblib.dump(self._le, os.path.join(directory, 'label_encoder.pkl'))

    def load_models(self, directory: str):
        svm_path = os.path.join(directory, 'svm_model.pkl')
        rf_path = os.path.join(directory, 'rf_model.pkl')
        le_path = os.path.join(directory, 'label_encoder.pkl')
        wl_path = os.path.join(directory, 'wavelengths.npy')

        if os.path.exists(svm_path):
            self._svm_model = joblib.load(svm_path)
        if os.path.exists(rf_path):
            self._rf_model = joblib.load(rf_path)
        if os.path.exists(le_path):
            self._le = joblib.load(le_path)
        if os.path.exists(wl_path):
            self._trained_wavelengths = np.load(wl_path)
