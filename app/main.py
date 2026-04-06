from __future__ import annotations

import os
import sys

from PyQt5.QtWidgets import QApplication

from ui import MainWindow


def _resolve_base_dir() -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    base_dir = _resolve_base_dir()
    classical_file = os.path.join(base_dir, "classical_scattering.py")
    quantum_file = os.path.join(base_dir, "quantum_scattering.py")

    app = QApplication(sys.argv)
    window = MainWindow(classical_file=classical_file, quantum_file=quantum_file)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
