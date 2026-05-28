"""
============================================================
  HyperGeo v2 — Hyperspectral Image Analysis Tool
  Final Year Project | Python + PyQt6 + scikit-learn
============================================================

PURPOSE:
    This is the entry point of the application.
    It creates the Qt application, launches the main window,
    and hands control over to the Qt event loop.

HOW IT CONNECTS:
    main.py  →  MainWindow (ui/main_window.py)
                    ↓
               calls core/, utils/ modules as needed
"""

import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    # QApplication is required by PyQt6 — it manages the GUI event loop
    app = QApplication(sys.argv)
    app.setApplicationName("HyperGeo")

    # Create and show the main window
    window = MainWindow()
    window.show()

    # Start the event loop — the app runs until the window is closed
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
