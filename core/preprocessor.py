"""
============================================================
  Module: core/preprocessor.py
  Layer:  Processing Layer
============================================================

PURPOSE:
    Cleans and prepares the raw hyperspectral data before it
    is fed into the machine learning model.

    Raw data from sensors has noise, redundant bands, and
    inconsistent value ranges. Preprocessing fixes all of this.

WHY IS PREPROCESSING IMPORTANT?
    Think of it like washing vegetables before cooking.
    The raw data "works" but preprocessing makes it much
    cleaner and better for ML classification.

STEPS IN THIS MODULE (in order):
    1. Normalize       → scale all values to 0–1 range
    2. Remove bad bands → drop water-vapor noise bands
    3. Smooth spectra  → reduce random sensor noise
    4. PCA             → reduce hundreds of bands to ~10 components

INPUT:
    HyperspectralImage object (from loader.py)

OUTPUT:
    - Preprocessed cube (numpy array)
    - Updated wavelengths array
    - PCA object (for later use if needed)
    - Explained variance ratio (shows how much info PCA kept)

HOW IT CONNECTS:
    loader.py → HyperspectralImage
             → preprocessor.run_pipeline()
             → returns cleaned cube + wavelengths
             → passed to classifier.py and visualizer.py
"""

import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# ── Step 1: Normalization ─────────────────────────────────────────────────────

def normalize(cube):
    """
    Scales all pixel values to the range [0, 1].

    WHY DO WE NORMALIZE?
        Different sensors produce different value ranges.
        For example, one sensor might give values 0–4095,
        another might give 0–1.0 (reflectance).
        Normalizing makes all data comparable and prevents
        the ML model from being biased by large numbers.

    HOW IT WORKS:
        For each pixel spectrum, we find the min and max values
        and scale everything between them to 0–1.
        (This is called Min-Max scaling.)

    Input : cube — numpy array of shape (rows, cols, bands)
    Output: cube — numpy array of shape (rows, cols, bands), values in [0, 1]
    """
    rows, cols, bands = cube.shape

    # Flatten the cube to 2D: each row is one pixel's spectrum
    # Shape: (rows*cols, bands)
    flat = cube.reshape(-1, bands)

    # MinMaxScaler normalizes each feature (band) across all pixels
    # After this, each band's values range from 0 to 1
    scaler = MinMaxScaler()
    flat_normalized = scaler.fit_transform(flat)

    # Reshape back to 3D cube
    return flat_normalized.reshape(rows, cols, bands).astype(np.float32)


# ── Step 2: Bad Band Removal ──────────────────────────────────────────────────

# These wavelength ranges are known to be noisy in most sensors
# They correspond to atmospheric water vapour absorption windows
# where the sensor data is unreliable
BAD_BAND_RANGES_NM = [
    (1340, 1460),   # Water vapour absorption zone 1
    (1790, 1960),   # Water vapour absorption zone 2
    (2450, 2600),   # Sensor edge noise
]

def remove_bad_bands(cube, wavelengths):
    """
    Removes spectral bands that are known to be unreliable.

    WHY REMOVE BANDS?
        Certain wavelength ranges are absorbed by water vapour
        in the atmosphere before reaching the sensor. The sensor
        still records values there, but they're mostly noise —
        not useful for mineral/material identification.

    HOW IT WORKS:
        We check each band's wavelength against the known bad ranges.
        Any band whose wavelength falls in a bad range is removed.

    Input:
        cube        — shape (rows, cols, bands)
        wavelengths — shape (bands,)

    Output:
        cleaned_cube        — shape (rows, cols, kept_bands)
        cleaned_wavelengths — shape (kept_bands,)
    """
    # Start with all bands marked as 'keep'
    keep_mask = np.ones(len(wavelengths), dtype=bool)

    # Mark bands within known bad ranges as 'remove'
    for low_nm, high_nm in BAD_BAND_RANGES_NM:
        bad = (wavelengths >= low_nm) & (wavelengths <= high_nm)
        keep_mask[bad] = False

    kept = np.sum(keep_mask)
    removed = len(wavelengths) - kept
    print(f"[Preprocessor] Bad band removal: kept {kept}, removed {removed} bands")

    # Return only the good bands
    return cube[:, :, keep_mask], wavelengths[keep_mask]


# ── Step 3: Spectral Smoothing ────────────────────────────────────────────────

def smooth_spectra(cube):
    """
    Applies Savitzky-Golay filter to smooth spectral signatures.

    WHY SMOOTH?
        Real sensor data has random noise — small random jumps
        in values from band to band. This noise doesn't represent
        the actual material; it's measurement error.
        Smoothing reduces this noise while keeping the real
        absorption features (the meaningful dips in the spectrum).

    WHAT IS SAVITZKY-GOLAY?
        It fits a small polynomial to a sliding window of points
        and uses it to smooth the data. It's better than simple
        averaging because it preserves the shape of peaks and dips.

    Input : cube — shape (rows, cols, bands)
    Output: cube — shape (rows, cols, bands), smoothed
    """
    rows, cols, bands = cube.shape

    # Need at least 11 bands for the default window
    # If fewer bands exist, reduce the window size
    window = min(11, bands if bands % 2 == 1 else bands - 1)
    poly_order = min(3, window - 1)

    if window <= poly_order:
        print("[Preprocessor] Too few bands to smooth — skipping")
        return cube

    # Flatten: shape (pixels, bands)
    flat = cube.reshape(-1, bands)

    # Apply filter along axis=1 (spectral axis, across bands)
    smoothed = savgol_filter(flat, window_length=window, polyorder=poly_order, axis=1)

    print(f"[Preprocessor] Smoothing applied (window={window}, poly={poly_order})")
    return smoothed.reshape(rows, cols, bands).astype(np.float32)


# ── Step 4: PCA Dimensionality Reduction ──────────────────────────────────────

def apply_pca(cube, n_components=10):
    """
    Reduces hundreds of spectral bands down to a small number
    of 'principal components' (PCA features).

    WHY USE PCA?
        A hyperspectral image might have 200 bands, but many of
        them carry the same information (they're correlated).
        PCA finds the directions of maximum variation in the data
        and represents the data along those directions.

        Example: Instead of 200 bands, we might capture 95% of
        the information using just 10 PCA components. This makes
        classification faster and often more accurate (because
        we've removed noise / redundant info).

    HOW PCA WORKS (simple explanation):
        Imagine your data as a cloud of points in 200-dimensional
        space. PCA finds the 10 directions along which the cloud
        is most 'spread out'. It then projects all points onto
        those 10 directions.
        → 200 numbers per pixel become 10 numbers per pixel
        → Much simpler, but still captures the key patterns

    WHY IS THIS IMPORTANT FOR VIVA?
        - Reduces computation time
        - Removes correlated/redundant band info
        - Often improves classification accuracy
        - Makes Random Forest training faster

    Input:
        cube         — shape (rows, cols, bands)
        n_components — how many PCA components to keep (default: 10)

    Output:
        pca_cube         — shape (rows, cols, n_components)
        pca_object       — fitted sklearn PCA object (for later inverse transform)
        variance_ratio   — array showing % variance each component explains
    """
    rows, cols, bands = cube.shape

    # Cap components at min(n_bands, n_pixels)
    n_components = min(n_components, bands, rows * cols)

    # Flatten to 2D: (pixels, bands)
    flat = cube.reshape(-1, bands)

    print(f"[Preprocessor] Running PCA: {bands} bands → {n_components} components")

    # Fit PCA and transform the data
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(flat)   # shape: (pixels, n_components)

    # How much total variance do we keep?
    total_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"[Preprocessor] PCA keeps {total_variance:.1f}% of total variance")

    # Reshape back to 3D: (rows, cols, n_components)
    pca_cube = transformed.reshape(rows, cols, n_components).astype(np.float32)

    return pca_cube, pca, pca.explained_variance_ratio_


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(cube, wavelengths, n_pca_components=10):
    """
    Runs all preprocessing steps in the correct order.

    This is the main function called by the UI.
    It takes a raw HyperspectralImage and returns cleaned,
    PCA-reduced data ready for classification.

    ORDER OF STEPS:
        1. Normalize       (scale values to 0–1)
        2. Remove bad bands (drop noisy wavelength ranges)
        3. Smooth spectra  (reduce sensor noise)
        4. PCA             (reduce bands → components)

    Input:
        hyperspectral_image — HyperspectralImage object from loader.py
        n_pca_components    — number of PCA components to keep

    Output: dictionary with keys:
        'pca_cube'        — np.array (rows, cols, n_pca_components)
        'clean_cube'      — np.array (rows, cols, bands_after_bad_removal)
        'wavelengths'     — np.array (bands_after_bad_removal,)
        'pca_object'      — fitted sklearn PCA
        'variance_ratio'  — array of per-component variance explained
    """
    print("\n[Preprocessor] Starting preprocessing pipeline...")

    cube = cube.copy()
    wavelengths = wavelengths.copy()

    # Step 1: Normalize
    print("[Preprocessor] Step 1/4 — Normalizing...")
    cube = normalize(cube)

    # Step 2: Remove bad bands (only if wavelengths are real nm values)
    if wavelengths.max() > 100:    # if wavelengths look like nm (not band indices)
        print("[Preprocessor] Step 2/4 — Removing bad bands...")
        cube, wavelengths = remove_bad_bands(cube, wavelengths)
    else:
        print("[Preprocessor] Step 2/4 — Skipping bad band removal (no wavelength info)")

    # Step 3: Smooth
    print("[Preprocessor] Step 3/4 — Smoothing spectra...")
    cube = smooth_spectra(cube)

    # Step 4: PCA
    print("[Preprocessor] Step 4/4 — Applying PCA...")
    pca_cube, pca_object, variance_ratio = apply_pca(cube, n_pca_components)

    print("[Preprocessor] Pipeline complete.\n")

    return {
        'pca_cube':      pca_cube,         # used by classifier
        'clean_cube':    cube,             # used by spectral plot
        'wavelengths':   wavelengths,      # used by spectral plot
        'pca_object':    pca_object,
        'variance_ratio': variance_ratio,
    }
