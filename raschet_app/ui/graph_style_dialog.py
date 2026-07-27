from __future__ import annotations

from typing import Dict

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from raschet_app.constants import BACKGROUND_OPTIONS, EXPORT_PRESET_OPTIONS, EXPORT_PRESET_OVERRIDES


class GraphStyleDialog(QDialog):
    styleApplied = Signal(dict)

    def __init__(self, current_style: Dict[str, object], x_title: str, y_title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Оформление графика")
        self.setModal(True)
        self.resize(520, 420)

        root = QVBoxLayout(self)

        form_box = QGroupBox("Оформление")
        form_layout = QFormLayout(form_box)

        self.x_title_edit = QLineEdit(x_title)
        self.y_title_edit = QLineEdit(y_title)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 28)
        self.font_size_spin.setValue(int(current_style.get("font_size", 10)))

        self.legend_font_size_spin = QSpinBox()
        self.legend_font_size_spin.setRange(8, 32)
        self.legend_font_size_spin.setValue(int(current_style.get("legend_font_size", 11)))

        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 8.0)
        self.line_width_spin.setSingleStep(0.5)
        self.line_width_spin.setDecimals(1)
        self.line_width_spin.setValue(float(current_style.get("line_width", 1.5)))

        self.point_size_spin = QDoubleSpinBox()
        self.point_size_spin.setRange(1.0, 20.0)
        self.point_size_spin.setSingleStep(0.5)
        self.point_size_spin.setDecimals(1)
        self.point_size_spin.setValue(float(current_style.get("point_size", 4.0)))

        self.background_combo = QComboBox()
        self.background_combo.addItems(BACKGROUND_OPTIONS)
        self.background_combo.setCurrentText(str(current_style.get("background_mode", "Белый")))

        self.grid_check = QCheckBox("Показывать сетку")
        self.grid_check.setChecked(bool(current_style.get("show_grid", True)))

        self.monochrome_check = QCheckBox("Чёрно-белый стиль")
        self.monochrome_check.setChecked(bool(current_style.get("monochrome", False)))

        self.export_scale_spin = QDoubleSpinBox()
        self.export_scale_spin.setRange(1.0, 6.0)
        self.export_scale_spin.setSingleStep(0.5)
        self.export_scale_spin.setDecimals(1)
        self.export_scale_spin.setValue(float(current_style.get("export_scale", 2.0)))

        form_layout.addRow("Подпись оси X:", self.x_title_edit)
        form_layout.addRow("Подпись оси Y:", self.y_title_edit)
        form_layout.addRow("Размер шрифта, pt:", self.font_size_spin)
        form_layout.addRow("Шрифт легенды, pt:", self.legend_font_size_spin)
        form_layout.addRow("Толщина линий:", self.line_width_spin)
        form_layout.addRow("Размер точек, px:", self.point_size_spin)
        form_layout.addRow("Фон графика:", self.background_combo)
        form_layout.addRow("Качество PNG (масштаб):", self.export_scale_spin)
        form_layout.addRow(self.grid_check)
        form_layout.addRow(self.monochrome_check)
        root.addWidget(form_box)

        preset_box = QGroupBox("Шаблон экспорта")
        preset_layout = QGridLayout(preset_box)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(EXPORT_PRESET_OPTIONS)
        self.preset_combo.setCurrentText(str(current_style.get("export_preset", "Обычный PNG")))
        self.apply_preset_btn = QPushButton("Применить шаблон к параметрам")
        self.apply_preset_btn.clicked.connect(self.apply_preset)
        preset_layout.addWidget(QLabel("Выберите шаблон:"), 0, 0)
        preset_layout.addWidget(self.preset_combo, 0, 1)
        preset_layout.addWidget(self.apply_preset_btn, 1, 0, 1, 2)
        root.addWidget(preset_box)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.apply_btn = buttons.addButton("Применить", QDialogButtonBox.ButtonRole.ApplyRole)
        self.apply_btn.clicked.connect(self._emit_style)
        buttons.accepted.connect(self._accept_with_emit)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def apply_preset(self) -> None:
        preset_name = self.preset_combo.currentText()
        overrides = EXPORT_PRESET_OVERRIDES.get(preset_name, {})
        if "line_width" in overrides:
            self.line_width_spin.setValue(float(overrides["line_width"]))
        if "background_mode" in overrides:
            self.background_combo.setCurrentText(str(overrides["background_mode"]))
        if "monochrome" in overrides:
            self.monochrome_check.setChecked(bool(overrides["monochrome"]))
        if "export_scale" in overrides:
            self.export_scale_spin.setValue(float(overrides["export_scale"]))
        if "legend_font_size" in overrides:
            self.legend_font_size_spin.setValue(int(overrides["legend_font_size"]))

    def get_style(self) -> dict:
        return {
            "x_axis_name": self.x_title_edit.text().strip() or "Time",
            "y_axis_name": self.y_title_edit.text().strip() or "Resistance",
            "font_size": int(self.font_size_spin.value()),
            "legend_font_size": int(self.legend_font_size_spin.value()),
            "legend_position": "Справа",
            "line_width": float(self.line_width_spin.value()),
            "point_size": float(self.point_size_spin.value()),
            "background_mode": self.background_combo.currentText(),
            "show_grid": self.grid_check.isChecked(),
            "monochrome": self.monochrome_check.isChecked(),
            "export_scale": float(self.export_scale_spin.value()),
            "export_preset": self.preset_combo.currentText(),
        }

    def _emit_style(self) -> None:
        self.styleApplied.emit(self.get_style())

    def _accept_with_emit(self) -> None:
        self._emit_style()
        self.accept()
