from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QStandardPaths

from .constants import APP_NAME


class JsonConfig:
    def __init__(self) -> None:
        try:
            base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
        except AttributeError:
            base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        self.config_dir = Path(base) if base else Path.home() / f".{APP_NAME}"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.config_dir / "config.json"
        self.data = self.load()

    def load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
