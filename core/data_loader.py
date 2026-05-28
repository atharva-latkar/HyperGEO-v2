"""
============================================================
  Module: core/loader.py
  Layer:  Data I/O Layer
============================================================

PURPOSE:
    Loads hyperspectral image files from disk into memory.
    Converts the raw file into a 3D NumPy array (the "data cube")
    that the rest of the project works with.

WHAT IS A HYPERSPECTRAL DATA CUBE?
    A normal image has 3 channels (R, G, B).
    A hyperspectral image has hundreds of channels (called bands).
    We store it as a 3D array:

        cube[row, column, band_index]

    Shape example: (145, 145, 200)
    → 145 rows × 145 columns × 200 spectral bands

INPUT:
    File path (string) to an ENVI or MATLAB .mat file

OUTPUT:
    A HyperspectralImage object containing:
        - cube       : numpy array of shape (rows, cols, bands)
        - wavelengths: numpy array of wavelength values in nm
        - filename   : original file name

SUPPORTED FORMATS:
    - ENVI (.hdr header + .img/.raw data file) — via spectral library
    - MATLAB .mat files — via scipy (for benchmark datasets like Indian Pines)

HOW IT CONNECTS TO OTHER MODULES:
    main_window.py → calls loader.load_file(path)
                  → returns HyperspectralImage
                  → passed to preprocessor.py, visualizer.py, classifier.py
"""

import numpy as np
import os


class HyperspectralImage:
    """
    A simple container that holds the loaded hyperspectral data.

    Think of it like a structured box that stores:
      - the pixel data (3D cube)
      - the wavelength values for each band
      - the original file name
    """

    def __init__(self, cube, wavelengths, filename=""):
        """
        Parameters:
            cube        : np.ndarray of shape (rows, cols, bands)
            wavelengths : np.ndarray of shape (bands,), values in nm
            filename    : str, name of the source file
        """
        self.cube = cube.astype(np.float32)   # use float32 to save memory
        self.wavelengths = wavelengths
        self.filename = filename

        # Convenient properties
        self.n_rows = cube.shape[0]
        self.n_cols = cube.shape[1]
        self.n_bands = cube.shape[2]

    def get_pixel_spectrum(self, row, col):
        """
        Returns the full spectral signature of a single pixel.

        A spectral signature is just a 1D array of reflectance values
        — one value per band. It's like the 'fingerprint' of that pixel.

        Input : row (int), col (int)
        Output: numpy array of shape (n_bands,)
        """
        return self.cube[row, col, :]

    def get_band_image(self, band_index):
        """
        Returns a single band as a 2D grayscale image.

        Input : band_index (int)
        Output: numpy array of shape (rows, cols)
        """
        return self.cube[:, :, band_index]

    def get_rgb_preview(self, r_band=None, g_band=None, b_band=None):
        """
        Creates a false-color RGB image by mapping 3 bands to R, G, B channels.

        Because hyperspectral images have hundreds of bands,
        we pick 3 representative ones to create a viewable image.

        Input : optional band indices for R, G, B (auto-selected if not given)
        Output: numpy array of shape (rows, cols, 3), dtype uint8
        """
        n = self.n_bands
        # Auto-select bands spread across the spectrum if not specified
        r = r_band if r_band is not None else int(n * 0.60)
        g = g_band if g_band is not None else int(n * 0.35)
        b = b_band if b_band is not None else int(n * 0.10)

        # Stack the 3 chosen bands into an RGB image
        rgb = np.stack([
            self.cube[:, :, r],
            self.cube[:, :, g],
            self.cube[:, :, b]
        ], axis=2)

        # Stretch contrast: map 2nd–98th percentile to full 0–255 range
        # This makes the image look better visually
        for channel in range(3):
            low, high = np.percentile(rgb[:, :, channel], [2, 98])
            if high > low:
                rgb[:, :, channel] = np.clip(
                    (rgb[:, :, channel] - low) / (high - low), 0, 1
                )

        return (rgb * 255).astype(np.uint8)


# ── File loading functions ────────────────────────────────────────────────────

def load_envi(filepath):
    """
    Loads an ENVI format hyperspectral file.

    ENVI format consists of two files:
      - A .hdr (header) file: contains metadata (rows, cols, bands, wavelengths)
      - A binary data file (.img or same name without .hdr): contains pixel values

    We use the 'spectral' Python library to read ENVI files.

    Input : filepath (str) — path to the .hdr file
    Output: HyperspectralImage object
    """
    try:
        import spectral
    except ImportError:
        raise ImportError(
            "The 'spectral' library is not installed.\n"
            "Fix: pip install spectral"
        )

    print(f"[Loader] Opening ENVI file: {os.path.basename(filepath)}")

    # Open the image (lazy — doesn't load all data yet)
    img = spectral.open_image(filepath)

    # Load all data into memory as a NumPy array
    # img.load() returns a MemoryMappedArray — we convert to plain ndarray
    cube = np.array(img.load(), dtype=np.float32)

    # Try to read wavelength values from the header
    if hasattr(img, 'bands') and img.bands.centers:
        wavelengths = np.array(img.bands.centers, dtype=np.float32)
    else:
        # If no wavelengths in header, create dummy values 400–2500 nm
        wavelengths = np.linspace(400, 2500, cube.shape[2], dtype=np.float32)

    print(f"[Loader] Loaded cube shape: {cube.shape}")
    return HyperspectralImage(cube, wavelengths, os.path.basename(filepath))


def load_mat(filepath):
    """
    Loads a MATLAB .mat file — used for benchmark datasets like:
      - Indian Pines (145×145, 200 bands)
      - Pavia University (610×340, 103 bands)
      - Salinas (512×217, 204 bands)

    These are standard academic datasets used to test hyperspectral methods.

    The .mat file contains a named variable holding the 3D data cube.
    We search for the first 3D variable and use it.

    Input : filepath (str) — path to the .mat file
    Output: HyperspectralImage object
    """
    import scipy.io as sio

    print(f"[Loader] Opening .mat file: {os.path.basename(filepath)}")

    # Known benchmark dataset variable names → typical wavelength ranges
    KNOWN_DATASETS = {
        'indian_pines':           np.linspace(400, 2500, 200),
        'indian_pines_corrected': np.linspace(400, 2500, 200),
        'paviaU':                 np.linspace(430,  860, 103),
        'pavia':                  np.linspace(430,  860, 102),
        'salinas':                np.linspace(400, 2500, 204),
        'salinas_corrected':      np.linspace(400, 2500, 204),
        'KSC':                    np.linspace(400, 2500, 176),
    }

    mat = sio.loadmat(filepath)

    cube = None
    wavelengths = None

    # First, check for known benchmark variable names
    for var_name, default_wl in KNOWN_DATASETS.items():
        if var_name in mat:
            cube = mat[var_name].astype(np.float32)
            wavelengths = default_wl.astype(np.float32)
            print(f"[Loader] Recognized benchmark dataset: {var_name}")
            break

    # If not a known dataset, find the first 3D array in the file
    if cube is None:
        for key, value in mat.items():
            if key.startswith('_'):
                continue   # skip MATLAB metadata keys
            if isinstance(value, np.ndarray) and value.ndim == 3:
                cube = value.astype(np.float32)
                wavelengths = np.linspace(400, 2500, cube.shape[2])
                print(f"[Loader] Found 3D array in variable: '{key}'")
                break

    if cube is None:
        raise ValueError(
            "No 3D array found in the .mat file.\n"
            "Expected a variable with shape (rows, cols, bands)."
        )

    # Some files store as (bands, rows, cols) — fix to (rows, cols, bands)
    # We detect this: if the first dimension is much smaller than the others
    if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
        cube = np.transpose(cube, (1, 2, 0))
        print(f"[Loader] Transposed cube from bands-first to rows-first")

    print(f"[Loader] Final cube shape: {cube.shape}")
    return HyperspectralImage(
        cube,
        wavelengths[:cube.shape[2]].astype(np.float32),
        os.path.basename(filepath)
    )


def load_file(filepath):
    """
    Main loader function — detects format and calls the right loader.

    This is the function called by the UI. It figures out what kind
    of file was selected and routes it to the correct loader.

    Input : filepath (str)
    Output: HyperspectralImage object

    Raises: ValueError if the format is not supported
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.hdr':
        return load_envi(filepath)

    elif ext == '.mat':
        return load_mat(filepath)

    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'\n"
            f"Supported formats: .hdr (ENVI), .mat (MATLAB)"
        )


def get_file_filter():
    """Returns the file dialog filter string for Qt file picker."""
    return (
        "Hyperspectral Files (*.hdr *.mat);;"
        "ENVI Files (*.hdr);;"
        "MATLAB Files (*.mat);;"
        "All Files (*)"
    )
