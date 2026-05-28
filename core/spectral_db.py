"""
Spectral Reference Database
Built-in mineral/rock spectral signatures derived from USGS Spectral Library.
Wavelengths in nanometers, reflectance values normalized 0-1.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MineralSpectrum:
    name: str
    category: str          # e.g. 'Clay', 'Carbonate', 'Iron Oxide', 'Silicate'
    wavelengths: np.ndarray
    reflectance: np.ndarray
    description: str = ""
    color: str = "#888888"  # Display color for mineral map


# Key wavelength points (nm) for our built-in reference spectra
_WL = np.array([
    400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
    1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450,
    1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950,
    2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500
], dtype=np.float32)


# ── Built-in mineral library ──────────────────────────────────────────────────

BUILTIN_MINERALS = [

    MineralSpectrum(
        name="Kaolinite",
        category="Clay Mineral",
        wavelengths=_WL,
        reflectance=np.array([
            0.10, 0.13, 0.18, 0.25, 0.28, 0.30, 0.35, 0.50, 0.55, 0.58, 0.60, 0.58,
            0.60, 0.62, 0.63, 0.62, 0.61, 0.60, 0.58, 0.40, 0.30, 0.28,
            0.48, 0.56, 0.60, 0.62, 0.61, 0.60, 0.55, 0.50, 0.35, 0.30,
            0.52, 0.58, 0.60, 0.55, 0.40, 0.55, 0.58, 0.55, 0.50, 0.40, 0.35
        ], dtype=np.float32),
        description="Common clay mineral; strong Al-OH absorptions at 1400 & 2200 nm",
        color="#F4D03F"
    ),

    MineralSpectrum(
        name="Montmorillonite",
        category="Clay Mineral",
        wavelengths=_WL,
        reflectance=np.array([
            0.08, 0.10, 0.15, 0.20, 0.23, 0.25, 0.30, 0.42, 0.46, 0.48, 0.50, 0.49,
            0.50, 0.52, 0.53, 0.52, 0.50, 0.49, 0.47, 0.30, 0.22, 0.20,
            0.40, 0.48, 0.52, 0.54, 0.53, 0.50, 0.44, 0.38, 0.25, 0.22,
            0.44, 0.50, 0.52, 0.46, 0.32, 0.44, 0.46, 0.42, 0.38, 0.30, 0.25
        ], dtype=np.float32),
        description="Smectite clay; broad absorptions at 1400, 1900, 2200 nm",
        color="#E8C84A"
    ),

    MineralSpectrum(
        name="Calcite",
        category="Carbonate",
        wavelengths=_WL,
        reflectance=np.array([
            0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.87, 0.88, 0.87,
            0.88, 0.89, 0.89, 0.88, 0.87, 0.86, 0.85, 0.60, 0.50, 0.48,
            0.80, 0.85, 0.87, 0.88, 0.87, 0.86, 0.82, 0.76, 0.55, 0.50,
            0.82, 0.85, 0.86, 0.82, 0.78, 0.60, 0.40, 0.50, 0.55, 0.45, 0.40
        ], dtype=np.float32),
        description="Limestone/marble; diagnostic CO3 absorption at 2340 nm",
        color="#ECF0F1"
    ),

    MineralSpectrum(
        name="Dolomite",
        category="Carbonate",
        wavelengths=_WL,
        reflectance=np.array([
            0.55, 0.60, 0.65, 0.70, 0.73, 0.75, 0.77, 0.80, 0.82, 0.83, 0.84, 0.83,
            0.84, 0.85, 0.85, 0.84, 0.83, 0.82, 0.80, 0.58, 0.48, 0.46,
            0.76, 0.80, 0.82, 0.84, 0.83, 0.81, 0.76, 0.70, 0.50, 0.46,
            0.78, 0.82, 0.83, 0.78, 0.70, 0.55, 0.36, 0.48, 0.52, 0.42, 0.36
        ], dtype=np.float32),
        description="Ca-Mg carbonate; similar to calcite but shifted CO3 feature ~2320 nm",
        color="#D5D8DC"
    ),

    MineralSpectrum(
        name="Goethite",
        category="Iron Oxide",
        wavelengths=_WL,
        reflectance=np.array([
            0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.18, 0.22, 0.25, 0.22, 0.20,
            0.22, 0.24, 0.25, 0.25, 0.26, 0.27, 0.26, 0.18, 0.15, 0.14,
            0.24, 0.26, 0.27, 0.28, 0.28, 0.27, 0.24, 0.20, 0.15, 0.13,
            0.22, 0.24, 0.25, 0.23, 0.21, 0.20, 0.19, 0.18, 0.16, 0.14, 0.12
        ], dtype=np.float32),
        description="Iron oxyhydroxide; broad Fe absorption ~900 nm",
        color="#E67E22"
    ),

    MineralSpectrum(
        name="Hematite",
        category="Iron Oxide",
        wavelengths=_WL,
        reflectance=np.array([
            0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.11, 0.15, 0.18, 0.20, 0.19, 0.17,
            0.19, 0.21, 0.22, 0.22, 0.23, 0.24, 0.23, 0.16, 0.13, 0.12,
            0.21, 0.23, 0.24, 0.25, 0.25, 0.24, 0.21, 0.18, 0.13, 0.11,
            0.20, 0.22, 0.23, 0.21, 0.19, 0.18, 0.17, 0.16, 0.14, 0.12, 0.10
        ], dtype=np.float32),
        description="Fe2O3; reddish iron oxide with absorptions near 530 & 900 nm",
        color="#C0392B"
    ),

    MineralSpectrum(
        name="Quartz",
        category="Silicate",
        wavelengths=_WL,
        reflectance=np.array([
            0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.87, 0.88, 0.88, 0.87,
            0.88, 0.89, 0.89, 0.89, 0.89, 0.88, 0.87, 0.72, 0.65, 0.62,
            0.85, 0.87, 0.88, 0.89, 0.88, 0.87, 0.84, 0.80, 0.68, 0.62,
            0.84, 0.87, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.80, 0.75, 0.70
        ], dtype=np.float32),
        description="SiO2; high reflectance, relatively featureless spectrum",
        color="#AED6F1"
    ),

    MineralSpectrum(
        name="Alunite",
        category="Sulfate",
        wavelengths=_WL,
        reflectance=np.array([
            0.20, 0.25, 0.32, 0.40, 0.43, 0.44, 0.46, 0.54, 0.58, 0.60, 0.61, 0.60,
            0.61, 0.62, 0.63, 0.62, 0.61, 0.60, 0.58, 0.38, 0.30, 0.28,
            0.48, 0.52, 0.54, 0.55, 0.54, 0.52, 0.48, 0.44, 0.32, 0.28,
            0.46, 0.50, 0.52, 0.46, 0.38, 0.46, 0.48, 0.44, 0.40, 0.32, 0.28
        ], dtype=np.float32),
        description="K-Al sulfate; Al-OH features at 1480, 2165 nm",
        color="#A9CCE3"
    ),

    MineralSpectrum(
        name="Jarosite",
        category="Sulfate",
        wavelengths=_WL,
        reflectance=np.array([
            0.03, 0.04, 0.06, 0.10, 0.14, 0.16, 0.18, 0.24, 0.28, 0.30, 0.29, 0.27,
            0.28, 0.30, 0.31, 0.31, 0.31, 0.30, 0.29, 0.20, 0.17, 0.16,
            0.27, 0.29, 0.30, 0.31, 0.30, 0.28, 0.25, 0.22, 0.17, 0.15,
            0.25, 0.27, 0.28, 0.26, 0.24, 0.22, 0.21, 0.20, 0.18, 0.15, 0.13
        ], dtype=np.float32),
        description="K-Fe sulfate; oxidized zones, yellowish-brown",
        color="#D4AC0D"
    ),

    MineralSpectrum(
        name="Muscovite",
        category="Mica",
        wavelengths=_WL,
        reflectance=np.array([
            0.15, 0.18, 0.24, 0.32, 0.36, 0.38, 0.40, 0.50, 0.54, 0.56, 0.57, 0.56,
            0.57, 0.58, 0.59, 0.58, 0.58, 0.57, 0.55, 0.36, 0.28, 0.26,
            0.46, 0.52, 0.55, 0.56, 0.55, 0.52, 0.46, 0.40, 0.28, 0.25,
            0.44, 0.48, 0.50, 0.44, 0.36, 0.44, 0.46, 0.42, 0.38, 0.30, 0.25
        ], dtype=np.float32),
        description="White mica; Al-OH features at 2200, 2350 nm",
        color="#D7BDE2"
    ),

    MineralSpectrum(
        name="Chlorite",
        category="Phyllosilicate",
        wavelengths=_WL,
        reflectance=np.array([
            0.04, 0.06, 0.10, 0.15, 0.14, 0.12, 0.14, 0.22, 0.27, 0.29, 0.28, 0.26,
            0.27, 0.29, 0.30, 0.29, 0.29, 0.28, 0.27, 0.18, 0.14, 0.13,
            0.24, 0.27, 0.28, 0.29, 0.28, 0.27, 0.24, 0.20, 0.15, 0.13,
            0.23, 0.25, 0.26, 0.24, 0.22, 0.21, 0.20, 0.18, 0.16, 0.13, 0.11
        ], dtype=np.float32),
        description="Mg-Fe sheet silicate; Fe-OH at 2250 nm, Mg-OH at 2335 nm",
        color="#27AE60"
    ),

    MineralSpectrum(
        name="Gypsum",
        category="Sulfate",
        wavelengths=_WL,
        reflectance=np.array([
            0.55, 0.60, 0.65, 0.70, 0.73, 0.75, 0.77, 0.80, 0.82, 0.83, 0.83, 0.82,
            0.83, 0.84, 0.84, 0.83, 0.82, 0.80, 0.78, 0.56, 0.46, 0.43,
            0.74, 0.78, 0.80, 0.82, 0.81, 0.78, 0.72, 0.66, 0.48, 0.43,
            0.74, 0.77, 0.78, 0.74, 0.70, 0.68, 0.66, 0.64, 0.60, 0.52, 0.46
        ], dtype=np.float32),
        description="CaSO4·2H2O; water features at 1450, 1950 nm",
        color="#FDFEFE"
    ),

    MineralSpectrum(
        name="Olivine",
        category="Silicate",
        wavelengths=_WL,
        reflectance=np.array([
            0.05, 0.07, 0.10, 0.15, 0.16, 0.15, 0.17, 0.26, 0.30, 0.28, 0.22, 0.18,
            0.20, 0.23, 0.25, 0.25, 0.26, 0.27, 0.26, 0.18, 0.14, 0.12,
            0.23, 0.25, 0.26, 0.27, 0.26, 0.25, 0.22, 0.18, 0.13, 0.11,
            0.21, 0.23, 0.24, 0.22, 0.20, 0.19, 0.18, 0.16, 0.14, 0.11, 0.09
        ], dtype=np.float32),
        description="Fe-Mg silicate; broad Fe2+ absorption centred ~1050 nm",
        color="#2ECC71"
    ),

    MineralSpectrum(
        name="Vegetation",
        category="Organic",
        wavelengths=_WL,
        reflectance=np.array([
            0.03, 0.04, 0.05, 0.10, 0.06, 0.04, 0.05, 0.45, 0.50, 0.52, 0.52, 0.51,
            0.50, 0.48, 0.45, 0.43, 0.40, 0.38, 0.35, 0.25, 0.18, 0.15,
            0.30, 0.34, 0.36, 0.37, 0.36, 0.35, 0.30, 0.25, 0.16, 0.13,
            0.28, 0.30, 0.28, 0.24, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08
        ], dtype=np.float32),
        description="Green vegetation; strong red-edge, chlorophyll absorptions",
        color="#1E8449"
    ),

    MineralSpectrum(
        name="Dry Soil",
        category="Soil",
        wavelengths=_WL,
        reflectance=np.array([
            0.08, 0.10, 0.13, 0.17, 0.20, 0.22, 0.24, 0.28, 0.31, 0.33, 0.34, 0.33,
            0.34, 0.35, 0.36, 0.36, 0.36, 0.35, 0.34, 0.24, 0.19, 0.17,
            0.31, 0.34, 0.36, 0.37, 0.36, 0.35, 0.32, 0.28, 0.20, 0.17,
            0.30, 0.33, 0.34, 0.32, 0.30, 0.29, 0.28, 0.26, 0.23, 0.19, 0.16
        ], dtype=np.float32),
        description="Bare dry soil; featureless, gently rising spectrum",
        color="#A04000"
    ),
]


class SpectralDatabase:
    """In-memory spectral reference library with lookup and matching utilities."""

    def __init__(self):
        self.minerals = {m.name: m for m in BUILTIN_MINERALS}

    def get_names(self) -> List[str]:
        return list(self.minerals.keys())

    def get_categories(self) -> List[str]:
        return sorted(set(m.category for m in self.minerals.values()))

    def get_by_category(self, category: str) -> List[MineralSpectrum]:
        return [m for m in self.minerals.values() if m.category == category]

    def get(self, name: str) -> Optional[MineralSpectrum]:
        return self.minerals.get(name)

    def interpolate_to(self, mineral_name: str, target_wavelengths: np.ndarray) -> np.ndarray:
        """Interpolate a reference spectrum to match the image wavelength grid."""
        mineral = self.minerals[mineral_name]
        return np.interp(target_wavelengths, mineral.wavelengths, mineral.reflectance).astype(np.float32)

    def build_reference_matrix(self, target_wavelengths: np.ndarray) -> tuple:
        """
        Build a (n_minerals × n_bands) reference matrix.
        Returns (matrix, names, colors).
        """
        names = self.get_names()
        matrix = np.stack([
            self.interpolate_to(n, target_wavelengths) for n in names
        ], axis=0)
        colors = [self.minerals[n].color for n in names]
        return matrix, names, colors

    def spectral_angle(self, spectrum: np.ndarray, reference: np.ndarray) -> float:
        """Compute spectral angle (radians) between two vectors."""
        num = np.dot(spectrum, reference)
        denom = np.linalg.norm(spectrum) * np.linalg.norm(reference)
        if denom == 0:
            return np.pi / 2
        cos_theta = np.clip(num / denom, -1, 1)
        return float(np.arccos(cos_theta))

    def identify_spectrum(self, spectrum: np.ndarray, wavelengths: np.ndarray,
                          top_n=3) -> List[dict]:
        """
        Match a single spectrum against the library using SAM.
        Returns top_n matches sorted by similarity.
        """
        results = []
        for name in self.get_names():
            ref = self.interpolate_to(name, wavelengths)
            angle = self.spectral_angle(spectrum, ref)
            similarity = 1 - (angle / (np.pi / 2))  # 0–1
            results.append({
                'name': name,
                'category': self.minerals[name].category,
                'angle_deg': np.degrees(angle),
                'similarity': similarity,
                'color': self.minerals[name].color,
            })
        results.sort(key=lambda x: x['angle_deg'])
        return results[:top_n]


# Singleton instance
spectral_db = SpectralDatabase()
