"""
Band Viewer Widget — displays a single band or RGB composite.
Supports click-to-select pixel for spectral profiling.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QSlider, QComboBox, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm


class BandViewerWidget(QWidget):
    """Displays band images with click-to-profile functionality."""

    pixel_clicked = pyqtSignal(int, int)  # row, col

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cube = None
        self._wavelengths = None
        self._current_mode = 'RGB'
        self._current_band = 0
        self._result = None          # ClassificationResult
        self._show_overlay = False
        self._click_pos = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        mode_label = QLabel("View:")
        mode_label.setStyleSheet("color: #aaa; font-size: 11px;")
        ctrl.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['RGB Composite', 'Single Band', 'Mineral Map'])
        self.mode_combo.setStyleSheet(self._combo_style())
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        ctrl.addWidget(self.mode_combo)

        self.band_label = QLabel("Band: 0")
        self.band_label.setStyleSheet("color: #aaa; font-size: 11px;")
        ctrl.addWidget(self.band_label)

        self.band_slider = QSlider(Qt.Orientation.Horizontal)
        self.band_slider.setMinimum(0)
        self.band_slider.setMaximum(0)
        self.band_slider.setEnabled(False)
        self.band_slider.valueChanged.connect(self._on_band_changed)
        self.band_slider.setStyleSheet(self._slider_style())
        ctrl.addWidget(self.band_slider, 1)

        layout.addLayout(ctrl)

        # Image canvas
        self.figure = Figure(figsize=(5, 4), facecolor='#121212')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #121212;")
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        layout.addWidget(self.canvas, 1)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#121212')
        self.ax.axis('off')
        self.figure.tight_layout(pad=0)

        self.info_label = QLabel("No data loaded")
        self.info_label.setStyleSheet("color: #666; font-size: 10px; padding: 2px;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

    def load_data(self, cube: np.ndarray, wavelengths: np.ndarray):
        self._cube = cube
        self._wavelengths = wavelengths
        n_bands = cube.shape[2]
        self.band_slider.setMaximum(n_bands - 1)
        self.band_slider.setValue(0)
        self.band_slider.setEnabled(True)
        rows, cols = cube.shape[:2]
        self.info_label.setText(
            f"{rows}×{cols} px | {n_bands} bands | "
            f"{wavelengths[0]:.0f}–{wavelengths[-1]:.0f} nm"
            if wavelengths is not None and len(wavelengths) > 0
            else f"{rows}×{cols} px | {n_bands} bands"
        )
        self._render()

    def set_result(self, result):
        self._result = result
        self.mode_combo.setCurrentText('Mineral Map')

    def _on_mode_changed(self, text):
        self._current_mode = text
        single_band = text == 'Single Band'
        self.band_slider.setEnabled(single_band and self._cube is not None)
        self._render()

    def _on_band_changed(self, val):
        self._current_band = val
        wl = ""
        if self._wavelengths is not None and val < len(self._wavelengths):
            wl = f" ({self._wavelengths[val]:.0f} nm)"
        self.band_label.setText(f"Band: {val}{wl}")
        if self._current_mode == 'Single Band':
            self._render()

    def _on_canvas_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if self._cube is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        rows, cols = self._cube.shape[:2]
        if 0 <= row < rows and 0 <= col < cols:
            self._click_pos = (row, col)
            self._render()
            self.pixel_clicked.emit(row, col)

    def _render(self):
        if self._cube is None:
            return
        self.ax.cla()
        self.ax.axis('off')
        self.ax.set_facecolor('#121212')

        mode = self._current_mode

        if mode == 'RGB Composite':
            from core.data_loader import HyperspectralImage
            d = HyperspectralImage(
                self._cube,
                self._wavelengths if self._wavelengths is not None else np.arange(self._cube.shape[2])
            )
            img = d.get_rgb_preview()
            self.ax.imshow(img, aspect='auto')

        elif mode == 'Single Band':
            band_img = self._cube[:, :, self._current_band]
            self.ax.imshow(band_img, cmap='gray', aspect='auto')

        elif mode == 'Mineral Map':
            if self._result is not None:
                color_map = self._result.color_map
                if color_map is not None:
                    self.ax.imshow(color_map, aspect='auto')
            else:
                self.ax.text(0.5, 0.5, 'Run classification first',
                             transform=self.ax.transAxes,
                             ha='center', va='center', color='#666', fontsize=12)

        # Draw click marker
        if self._click_pos is not None:
            r, c = self._click_pos
            self.ax.plot(c, r, '+', color='#00FFFF', markersize=14,
                         markeredgewidth=1.5, zorder=10)

        self.figure.tight_layout(pad=0)
        self.canvas.draw()

    @staticmethod
    def _combo_style():
        return """
            QComboBox {
                background: #2a2a2a; color: #ccc; border: 1px solid #444;
                border-radius: 3px; padding: 2px 6px; font-size: 11px; min-width: 120px;
            }
            QComboBox QAbstractItemView { background: #2a2a2a; color: #ccc; }
        """

    @staticmethod
    def _slider_style():
        return """
            QSlider::groove:horizontal { background: #333; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal {
                background: #0a84ff; border-radius: 6px; width: 12px; height: 12px;
                margin: -4px 0;
            }
        """
