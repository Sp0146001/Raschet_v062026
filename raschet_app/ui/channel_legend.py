from __future__ import annotations

from typing import List

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from raschet_app.models import ChannelData


class ChannelLegendWidget(QGroupBox):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Каналы / легенда", parent)
        self.rows: List[dict] = []

        outer = QVBoxLayout(self)
        btns = QHBoxLayout()
        self.select_all_btn = QPushButton("Все")
        self.clear_all_btn = QPushButton("Ни одного")
        self.select_all_btn.clicked.connect(self.select_all)
        self.clear_all_btn.clicked.connect(self.clear_all)
        btns.addWidget(self.select_all_btn)
        btns.addWidget(self.clear_all_btn)
        btns.addStretch(1)
        outer.addLayout(btns)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.content = QWidget()
        self.grid = QGridLayout(self.content)
        self.grid.setContentsMargins(6, 6, 6, 6)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(6)
        self.scroll.setWidget(self.content)
        outer.addWidget(self.scroll)

    def set_channels(self, channels: List[ChannelData], saved_names: List[str] | None = None) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.rows.clear()

        saved_names = saved_names or []
        for idx, channel in enumerate(channels):
            row = idx
            check = QCheckBox()
            check.setChecked(channel.visible)
            check.stateChanged.connect(lambda _state, self=self: self.changed.emit())

            color = QLabel()
            color.setFixedSize(16, 16)
            color.setStyleSheet(f"background:{channel.color}; border:1px solid #777; border-radius:3px;")
            color.setToolTip("Цвет канала на графике")

            name_edit = QLineEdit()
            if idx < len(saved_names) and saved_names[idx]:
                channel.display_name = saved_names[idx]
            name_edit.setText(channel.display_name)
            name_edit.editingFinished.connect(self.changed.emit)

            self.grid.addWidget(check, row, 0)
            self.grid.addWidget(color, row, 1)
            self.grid.addWidget(name_edit, row, 2)
            self.rows.append({"checkbox": check, "color": color, "edit": name_edit})

        self.setTitle(f"Каналы / легенда ({len(channels)})")

    def sync_to_channels(self, channels: List[ChannelData]) -> List[ChannelData]:
        for channel, row in zip(channels, self.rows):
            channel.visible = row["checkbox"].isChecked()
            channel.display_name = row["edit"].text().strip() or channel.display_name
        return channels

    def visible_channels(self, channels: List[ChannelData]) -> List[ChannelData]:
        self.sync_to_channels(channels)
        return [channel for channel in channels if channel.visible]

    def collect_names(self) -> List[str]:
        return [row["edit"].text().strip() for row in self.rows]

    def select_all(self) -> None:
        for row in self.rows:
            row["checkbox"].setChecked(True)
        self.changed.emit()

    def clear_all(self) -> None:
        for row in self.rows:
            row["checkbox"].setChecked(False)
        self.changed.emit()
