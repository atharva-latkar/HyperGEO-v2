"""
Main Application Window
"""

import os
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QPushButton, QLabel, QFileDialog, QProgressBar, QGroupBox,
    QComboBox, QTextEdit, QScrollArea, QFrame, QSizePolicy,
    QApplication, QStatusBar, QToolBar, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QAction, QColor

from ui.band_viewer import BandViewerWidget
from ui.spectral_plot import SpectralPlotWidget
from core.data_loader import load_file, get_file_filter
from core.preprocessor import run_pipeline
from core.classifier import Classifier
from core.spectral_db import spectral_db
from utils.export import export_full_report


# ── Worker threads ────────────────────────────────────────────────────────────

class LoadWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            data = load_file(self.filepath)
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))


class PreprocessWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, cube, wavelengths):
        super().__init__()
        self.cube = cube
        self.wavelengths = wavelengths

    def run(self):
        try:
            result = run_pipeline(
                self.cube,
                self.wavelengths
            )

            cube = result['clean_cube']
            wl = result['wavelengths']

            self.finished.emit(cube, wl)

        except Exception as e:
            self.error.emit(str(e))


class ClassifyWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, cube, wavelengths, method, classifier):
        super().__init__()
        self.cube = cube
        self.wavelengths = wavelengths
        self.method = method
        self.classifier = classifier

    def run(self):
        try:
            result = self.classifier.classify(
                self.cube, self.wavelengths,
                method=self.method,
                progress_callback=self.progress.emit
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    DARK_BG = "#0f0f0f"
    PANEL_BG = "#1a1a1a"
    ACCENT = "#0a84ff"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HyperGeo — Hyperspectral Geological Analysis")
        self.resize(1400, 820)
        self.setMinimumSize(900, 600)

        self._data = None
        self._processed_cube = None
        self._processed_wl = None
        self._result = None
        self._classifier = Classifier()
        self._threads = []

        self._apply_stylesheet()
        self._build_ui()
        self._build_menu()
        self._build_statusbar()

    # ── Stylesheet ────────────────────────────────────────────────────────────

    def _apply_stylesheet(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {self.DARK_BG};
                color: #cccccc;
                font-family: 'Segoe UI', 'SF Pro Display', Arial, sans-serif;
                font-size: 12px;
            }}
            QGroupBox {{
                border: 1px solid #2a2a2a;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: bold;
                color: #888888;
                font-size: 11px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px; top: -2px;
                padding: 0 4px;
            }}
            QPushButton {{
                background-color: #252525;
                color: #cccccc;
                border: 1px solid #383838;
                border-radius: 5px;
                padding: 6px 16px;
                font-size: 12px;
            }}
            QPushButton:hover {{ background-color: #2e2e2e; border-color: #555; }}
            QPushButton:pressed {{ background-color: #1a1a1a; }}
            QPushButton:disabled {{ color: #555; border-color: #2a2a2a; }}
            QPushButton#accent_btn {{
                background-color: {self.ACCENT};
                color: white;
                border: none;
                font-weight: bold;
            }}
            QPushButton#accent_btn:hover {{ background-color: #1a94ff; }}
            QPushButton#accent_btn:disabled {{ background-color: #1a3a5c; color: #555; }}
            QProgressBar {{
                border: none;
                border-radius: 3px;
                background: #222;
                height: 6px;
                text-align: center;
                color: transparent;
            }}
            QProgressBar::chunk {{
                background: {self.ACCENT};
                border-radius: 3px;
            }}
            QComboBox {{
                background: #222;
                color: #ccc;
                border: 1px solid #383838;
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QComboBox QAbstractItemView {{
                background: #222;
                color: #ccc;
                selection-background-color: #2a2a6a;
            }}
            QTextEdit {{
                background: #111;
                color: #999;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
            }}
            QLabel#title_label {{
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
            QLabel#subtitle_label {{
                color: #555555;
                font-size: 11px;
            }}
            QSplitter::handle {{
                background: #1e1e1e;
                width: 2px; height: 2px;
            }}
            QScrollBar:vertical {{
                background: #111; width: 8px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: #333; border-radius: 4px; min-height: 30px;
            }}
            QStatusBar {{ background: #111; color: #555; font-size: 11px; }}
            QMenuBar {{ background: #111; color: #aaa; }}
            QMenuBar::item:selected {{ background: #222; }}
            QMenu {{ background: #1a1a1a; color: #ccc; border: 1px solid #333; }}
            QMenu::item:selected {{ background: #2a2a6a; }}
        """)

    # ── UI Build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top banner ────────────────────────────────────────────────────────
        banner = QWidget()
        banner.setFixedHeight(52)
        banner.setStyleSheet(f"background: #111111; border-bottom: 1px solid #222;")
        bl = QHBoxLayout(banner)
        bl.setContentsMargins(18, 0, 18, 0)

        title = QLabel("⬡ HyperGeo")
        title.setObjectName("title_label")
        bl.addWidget(title)

        sub = QLabel("Hyperspectral Geological Analysis")
        sub.setObjectName("subtitle_label")
        bl.addWidget(sub)
        bl.addStretch()

        # Quick action buttons in banner
        self.load_btn = QPushButton("⊕  Load Data")
        self.load_btn.setObjectName("accent_btn")
        self.load_btn.setFixedHeight(32)
        self.load_btn.clicked.connect(self.open_file)
        bl.addWidget(self.load_btn)

        self.export_btn = QPushButton("↓  Export")
        self.export_btn.setFixedHeight(32)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        bl.addWidget(self.export_btn)

        root.addWidget(banner)

        # ── Main splitter ─────────────────────────────────────────────────────
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(2)
        root.addWidget(main_splitter, 1)

        # Left panel — controls
        left_panel = QWidget()
        left_panel.setFixedWidth(250)
        left_panel.setStyleSheet(f"background: {self.PANEL_BG}; border-right: 1px solid #222;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)
        self._build_left_panel(left_layout)
        main_splitter.addWidget(left_panel)

        # Center — viewer
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self.band_viewer = BandViewerWidget()
        self.band_viewer.pixel_clicked.connect(self._on_pixel_clicked)
        center_layout.addWidget(self.band_viewer, 1)

        main_splitter.addWidget(center_widget)

        # Right panel — spectral profile + legend
        right_panel = QWidget()
        right_panel.setFixedWidth(320)
        right_panel.setStyleSheet(f"background: {self.PANEL_BG}; border-left: 1px solid #222;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        self._build_right_panel(right_layout)
        main_splitter.addWidget(right_panel)

        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)

    def _build_left_panel(self, layout):
        # ── File info ─────────────────────────────────────────────────────────
        file_box = QGroupBox("Data File")
        fb_layout = QVBoxLayout(file_box)
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: #666; font-size: 10px;")
        fb_layout.addWidget(self.file_label)
        self.data_info = QLabel("")
        self.data_info.setStyleSheet("color: #888; font-size: 10px;")
        self.data_info.setWordWrap(True)
        fb_layout.addWidget(self.data_info)
        layout.addWidget(file_box)

        # ── Preprocessing ─────────────────────────────────────────────────────
        pre_box = QGroupBox("Preprocessing")
        pre_layout = QVBoxLayout(pre_box)
        self.preprocess_btn = QPushButton("Run Preprocessing")
        self.preprocess_btn.setEnabled(False)
        self.preprocess_btn.clicked.connect(self.run_preprocessing)
        pre_layout.addWidget(self.preprocess_btn)
        self.pre_status = QLabel("")
        self.pre_status.setStyleSheet("color: #888; font-size: 10px;")
        pre_layout.addWidget(self.pre_status)
        layout.addWidget(pre_box)

        # ── Classification ────────────────────────────────────────────────────
        cls_box = QGroupBox("Classification")
        cls_layout = QVBoxLayout(cls_box)

        method_lbl = QLabel("Method:")
        method_lbl.setStyleSheet("color: #888; font-size: 11px;")
        cls_layout.addWidget(method_lbl)

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Random Forest"
        ])
        cls_layout.addWidget(self.method_combo)

        self.classify_btn = QPushButton("Run Classification")
        self.classify_btn.setObjectName("accent_btn")
        self.classify_btn.setEnabled(False)
        self.classify_btn.clicked.connect(self.run_classification)
        cls_layout.addWidget(self.classify_btn)

        layout.addWidget(cls_box)

        # ── Progress ──────────────────────────────────────────────────────────
        prog_box = QGroupBox("Progress")
        prog_layout = QVBoxLayout(prog_box)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        prog_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("Idle")
        self.progress_label.setStyleSheet("color: #666; font-size: 10px;")
        prog_layout.addWidget(self.progress_label)
        layout.addWidget(prog_box)

        # ── Log ───────────────────────────────────────────────────────────────
        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(120)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_box)

        layout.addStretch()

    def _build_right_panel(self, layout):
        # Spectral plot
        spec_box = QGroupBox("Spectral Profile")
        spec_layout = QVBoxLayout(spec_box)
        self.spectral_plot = SpectralPlotWidget()
        spec_layout.addWidget(self.spectral_plot)
        layout.addWidget(spec_box, 1)

        # Legend / matches
        self.match_box = QGroupBox("Top Mineral Matches")
        match_layout = QVBoxLayout(self.match_box)
        self.match_labels = []
        for i in range(3):
            lbl = QLabel(f"—")
            lbl.setStyleSheet("color: #666; font-size: 11px; padding: 2px;")
            lbl.setWordWrap(True)
            match_layout.addWidget(lbl)
            self.match_labels.append(lbl)
        layout.addWidget(self.match_box)

        # Mineral legend (shown after classification)
        self.legend_box = QGroupBox("Mineral Legend")
        self.legend_layout = QVBoxLayout(self.legend_box)
        self.legend_scroll = QScrollArea()
        self.legend_scroll.setWidgetResizable(True)
        self.legend_scroll.setMaximumHeight(200)
        self.legend_scroll.setStyleSheet("border: none; background: transparent;")
        self.legend_inner = QWidget()
        self.legend_inner_layout = QVBoxLayout(self.legend_inner)
        self.legend_inner_layout.setSpacing(2)
        self.legend_scroll.setWidget(self.legend_inner)
        self.legend_layout.addWidget(self.legend_scroll)
        layout.addWidget(self.legend_box)

    def _build_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Hyperspectral File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        export_action = QAction("Export Results...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About HyperGeo", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — Load a hyperspectral file to begin")

    # ── Slot helpers ──────────────────────────────────────────────────────────

    def _log(self, msg):
        self.log_text.append(msg)
        self.log_text.ensureCursorVisible()

    def _set_progress(self, val, msg=""):
        self.progress_bar.setValue(val)
        self.progress_label.setText(msg)
        self.status_bar.showMessage(msg)
        QApplication.processEvents()

    # ── File loading ──────────────────────────────────────────────────────────

    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Hyperspectral File", "",
            get_file_filter()
        )
        if not filepath:
            return

        self._log(f"Loading: {os.path.basename(filepath)}")
        self.preprocess_btn.setEnabled(False)
        self.classify_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self._set_progress(0, "Loading file...")

        self._load_thread = QThread()
        self._load_worker = LoadWorker(filepath)
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.progress.connect(self._set_progress)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.error.connect(self._on_error)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.error.connect(self._load_thread.quit)
        self._load_thread.start()

    def _on_load_finished(self, data):
        self._data = data
        self._processed_cube = data.cube
        self._processed_wl = data.wavelengths
        self.file_label.setText(os.path.basename(data.filename))
        self.data_info.setText(
            f"Shape: {data.cube.shape}\n"
            f"Bands: {data.n_bands}"
        )
        self.band_viewer.load_data(data.cube, data.wavelengths)
        self.preprocess_btn.setEnabled(True)
        self.classify_btn.setEnabled(True)
        self._log(f"✓ Loaded {data.n_rows}×{data.n_cols}×{data.n_bands}")
        self._set_progress(100, "File loaded")

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def run_preprocessing(self):
        if self._data is None:
            return
        self._log("Starting preprocessing pipeline...")
        self._set_progress(0, "Preprocessing...")
        self.preprocess_btn.setEnabled(False)

        self._pre_thread = QThread()
        self._pre_worker = PreprocessWorker(self._data.cube, self._data.wavelengths)
        self._pre_worker.moveToThread(self._pre_thread)
        self._pre_thread.started.connect(self._pre_worker.run)
        self._pre_worker.progress.connect(self._set_progress)
        self._pre_worker.finished.connect(self._on_preprocess_finished)
        self._pre_worker.error.connect(self._on_error)
        self._pre_worker.finished.connect(self._pre_thread.quit)
        self._pre_worker.error.connect(self._pre_thread.quit)
        self._pre_thread.start()

    def _on_preprocess_finished(self, cube, wl):
        self._processed_cube = cube
        self._processed_wl = wl
        self.band_viewer.load_data(cube, wl)
        self.pre_status.setText(f"✓ Bands: {cube.shape[2]}, WL: {wl[0]:.0f}–{wl[-1]:.0f} nm")
        self.preprocess_btn.setEnabled(True)
        self.classify_btn.setEnabled(True)
        self._log(f"✓ Preprocessing done — {cube.shape[2]} bands remain")
        self._set_progress(100, "Preprocessing complete")

    # ── Classification ────────────────────────────────────────────────────────

    def run_classification(self):
        if self._processed_cube is None:
            return

        method = "rf"
        self._log(f"Running {method.upper()} classification...")
        self._set_progress(0, "Classifying...")
        self.classify_btn.setEnabled(False)

        self._cls_thread = QThread()
        self._cls_worker = ClassifyWorker(
            self._processed_cube, self._processed_wl,
            method, self._classifier
        )
        self._cls_worker.moveToThread(self._cls_thread)
        self._cls_thread.started.connect(self._cls_worker.run)
        self._cls_worker.progress.connect(self._set_progress)
        self._cls_worker.finished.connect(self._on_classify_finished)
        self._cls_worker.error.connect(self._on_error)
        self._cls_worker.finished.connect(self._cls_thread.quit)
        self._cls_worker.error.connect(self._cls_thread.quit)
        self._cls_thread.start()

    def _on_classify_finished(self, result):
        self._result = result
        self.band_viewer.set_result(result)
        self.export_btn.setEnabled(True)
        self.classify_btn.setEnabled(True)
        self._build_legend(result)
        self._log(f"✓ Classification done [{result.method}]")
        self._set_progress(100, f"Classification complete [{result.method}]")

    def _build_legend(self, result):
        # Clear existing
        while self.legend_inner_layout.count():
            item = self.legend_inner_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Count pixels per class
        rows, cols = result.label_map.shape
        total = rows * cols
        for idx, (name, color) in enumerate(zip(result.label_names, result.label_colors)):
            count = int(np.sum(result.label_map == idx))
            pct = count / total * 100
            if pct < 0.1:
                continue
            row_widget = QWidget()
            row_widget.setStyleSheet("background: transparent;")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            swatch = QLabel("  ")
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background-color: {color}; border-radius: 2px;"
            )
            row_layout.addWidget(swatch)

            lbl = QLabel(f"{name}  {pct:.1f}%")
            lbl.setStyleSheet("color: #aaa; font-size: 10px;")
            row_layout.addWidget(lbl, 1)

            self.legend_inner_layout.addWidget(row_widget)

        self.legend_inner_layout.addStretch()

    # ── Pixel click ───────────────────────────────────────────────────────────

    def _on_pixel_clicked(self, row, col):
        if self._processed_cube is None or self._processed_wl is None:
            return
        spectrum = self._processed_cube[row, col, :]
        matches = spectral_db.identify_spectrum(spectrum, self._processed_wl, top_n=3)
        self.spectral_plot.set_pixel_spectrum(row, col, self._processed_wl, spectrum, matches)

        # Update match labels
        for i, lbl in enumerate(self.match_labels):
            if i < len(matches):
                m = matches[i]
                lbl.setText(
                    f"#{i+1} {m['name']}  —  {m['similarity']:.0%}  "
                    f"({m['category']})"
                )
                lbl.setStyleSheet(
                    f"color: {spectral_db.get(m['name']).color}; "
                    f"font-size: 11px; padding: 2px;"
                )
            else:
                lbl.setText("—")

        self._log(f"Pixel ({row},{col}) → best match: {matches[0]['name']} ({matches[0]['similarity']:.0%})")

    # ── Export ────────────────────────────────────────────────────────────────

    def export_results(self):
        if self._result is None:
            QMessageBox.warning(self, "No Results", "Run classification first.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Export Results To")
        if not directory:
            return
        png_path, csv_path = export_full_report(self._result, directory, "hypergeo")
        self._log(f"✓ Exported: {os.path.basename(png_path)}, {os.path.basename(csv_path)}")
        QMessageBox.information(self, "Export Complete",
                                f"Saved to:\n{png_path}\n{csv_path}")

    # ── Error handler ─────────────────────────────────────────────────────────

    def _on_error(self, msg):
        self._log(f"✗ Error: {msg}")
        self._set_progress(0, "Error")
        self.preprocess_btn.setEnabled(self._data is not None)
        self.classify_btn.setEnabled(self._data is not None)
        QMessageBox.critical(self, "Error", msg)

    # ── About ─────────────────────────────────────────────────────────────────

    def _show_about(self):
        QMessageBox.about(self, "About HyperGeo",
            "<b>HyperGeo</b> v1.0<br><br>"
            "Hyperspectral Geological Analysis Desktop Application<br><br>"
            "Supports: ENVI, HDF5, GeoTIFF<br>"
            "Methods: Random Forest, SVM, SAM<br>"
            "Reference: USGS Spectral Library (built-in)<br><br>"
            "Built with PyQt6 · scikit-learn · NumPy"
        )
