from __future__ import annotations

from typing import Dict

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)


class SegmentDetailsDialog(QDialog):
    def __init__(self, current_data: Dict[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Детали участка")
        self.setModal(True)
        self.resize(520, 260)

        root = QVBoxLayout(self)
        form = QFormLayout()

        self.segment_name_edit = QLineEdit(current_data.get("segment_name", ""))
        self.gas_edit = QLineEdit(current_data.get("gas_name", ""))
        self.concentration_edit = QLineEdit(current_data.get("concentration_ppm", ""))
        self.temperature_edit = QLineEdit(current_data.get("temperature_c", ""))
        self.light_combo = QComboBox()
        self.light_combo.addItems(["", "OFF", "ON"])
        self.light_combo.setCurrentText(current_data.get("light_mode", ""))
        self.comment_edit = QLineEdit(current_data.get("comment", ""))

        form.addRow("Название сегмента:", self.segment_name_edit)
        form.addRow("Газ:", self.gas_edit)
        form.addRow("Концентрация, ppm:", self.concentration_edit)
        form.addRow("Температура, °C:", self.temperature_edit)
        form.addRow("Свет:", self.light_combo)
        form.addRow("Комментарий:", self.comment_edit)
        root.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def get_data(self) -> Dict[str, str]:
        return {
            "segment_name": self.segment_name_edit.text().strip(),
            "gas_name": self.gas_edit.text().strip(),
            "concentration_ppm": self.concentration_edit.text().strip(),
            "temperature_c": self.temperature_edit.text().strip(),
            "light_mode": self.light_combo.currentText().strip(),
            "comment": self.comment_edit.text().strip(),
        }
