from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import QApplication

from raschet_app.constants import APP_DISPLAY_NAME, APP_NAME
from raschet_app.ui.main_window import RaschetMainWindow


def resource_path(*parts: str) -> str:
    return str(Path(__file__).resolve().parent.joinpath(*parts))


def main() -> None:
    QGuiApplication.setApplicationDisplayName(APP_DISPLAY_NAME)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    icon_path = resource_path("assets", "app_icon.png")
    if Path(icon_path).exists():
        app.setWindowIcon(QIcon(icon_path))
    window = RaschetMainWindow(icon_path=icon_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
