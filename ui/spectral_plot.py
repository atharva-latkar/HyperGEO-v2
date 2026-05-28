"""
Spectral Profile Plot Widget
Shows the spectrum of a clicked pixel alongside reference spectra.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SpectralPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._wavelengths = None
        self._pixel_spectrum = None
        self._reference_spectra = {}   # name -> reflectance array
        self._matches = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header = QHBoxLayout()
        self.pixel_label = QLabel("Click a pixel to view its spectrum")
        self.pixel_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        header.addWidget(self.pixel_label)
        header.addStretch()

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedWidth(60)
        self.clear_btn.clicked.connect(self.clear)
        self.clear_btn.setStyleSheet("""
            QPushButton { background: #333; color: #ccc; border: 1px solid #555;
                          border-radius: 3px; padding: 2px 6px; font-size: 11px; }
            QPushButton:hover { background: #444; }
        """)
        header.addWidget(self.clear_btn)
        layout.addLayout(header)

        # Matplotlib figure
        self.figure = Figure(figsize=(5, 3), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #1e1e1e;")
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self._style_axes()

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='#888888', labelsize=8)
        ax.spines[:].set_color('#333333')
        ax.set_xlabel('Wavelength (nm)', color='#888888', fontsize=9)
        ax.set_ylabel('Reflectance', color='#888888', fontsize=9)
        ax.set_title('Spectral Profile', color='#cccccc', fontsize=10, pad=6)
        self.figure.tight_layout(pad=1.5)

    def set_pixel_spectrum(self, row: int, col: int,
                           wavelengths: np.ndarray, spectrum: np.ndarray,
                           matches: list = None):
        """Update plot with new pixel spectrum and optional reference matches."""
        self._wavelengths = wavelengths
        self._pixel_spectrum = spectrum
        self._matches = matches or []
        self.pixel_label.setText(f"Pixel ({row}, {col})")
        self._redraw()

    def add_reference(self, name: str, wavelengths: np.ndarray, reflectance: np.ndarray):
        self._reference_spectra[name] = (wavelengths, reflectance)
        self._redraw()

    def clear(self):
        self._pixel_spectrum = None
        self._matches = []
        self.pixel_label.setText("Click a pixel to view its spectrum")
        self._redraw()

    def _redraw(self):
        self.ax.cla()
        self._style_axes()

        MATCH_COLORS = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF', '#C77DFF']

        # Plot top matches from library
        for i, match in enumerate(self._matches[:3]):
            from core.spectral_db import spectral_db
            name = match['name']
            mineral = spectral_db.get(name)
            if mineral is not None and self._wavelengths is not None:
                ref_interp = spectral_db.interpolate_to(name, self._wavelengths)
                self.ax.plot(self._wavelengths, ref_interp,
                             color=mineral.color, alpha=0.55, linewidth=1.2,
                             linestyle='--',
                             label=f"{name} ({match['similarity']:.0%})")

        # Plot pixel spectrum on top
        if self._pixel_spectrum is not None and self._wavelengths is not None:
            self.ax.plot(self._wavelengths, self._pixel_spectrum,
                         color='#00CFFF', linewidth=1.8, label='Pixel spectrum', zorder=5)

        if self.ax.lines:
            self.ax.legend(fontsize=8, facecolor='#2a2a2a',
                           edgecolor='#444', labelcolor='#cccccc',
                           loc='upper right')

        self.canvas.draw()
