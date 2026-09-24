from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QDoubleValidator, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from raschet_app.config import JsonConfig
from raschet_app.constants import (
    APP_DISPLAY_NAME,
    APP_VERSION,
    CALC_MODES,
    DEFAULT_GRAPH_STYLE,
    EXPORT_PLOT_FILTER,
    EXPORT_PRESET_OVERRIDES,
    EXPORT_TABLE_FILTER,
)
from raschet_app.models import ChannelData, ParsedFile
from raschet_app.services.calculator import build_slice_dataframe, build_xline_dataframe, calculate_results, normalize_interval, slice_channel
from raschet_app.services.db import ProjectDatabase
from raschet_app.services.exporters import export_dataframe, export_plot
from raschet_app.services.parser import SmartTxtParser
from raschet_app.services.features import compute_channel_features
from raschet_app.services.pca_analysis import SCALE_MODES, run_pca
from raschet_app.services.preprocess import OPERATIONS, apply_preprocess_pipeline, profile_to_json
from raschet_app.ui.channel_legend import ChannelLegendWidget
from raschet_app.ui.graph_style_dialog import GraphStyleDialog
from raschet_app.ui.plot_widget import FastPlotWidget
from raschet_app.ui.segment_details_dialog import SegmentDetailsDialog
from raschet_app.ui.table_utils import TableUtils


class RaschetMainWindow(QMainWindow):
    def __init__(self, icon_path: str | None = None) -> None:
        super().__init__()
        self.config = JsonConfig()
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.resize(1600, 950)
        if icon_path and Path(icon_path).exists():
            self.setWindowIcon(QIcon(icon_path))

        self.db = ProjectDatabase()
        self.project_path: Optional[Path] = None
        self.current_source_file_id: Optional[int] = None
        self.current_segment_id: Optional[int] = None
        self.current_preprocessed_run_id: Optional[int] = None
        self.current_preprocessed_segment_id: Optional[int] = None
        self.current_preprocessed_reference_id: Optional[int] = None
        self.current_preprocessed_profile_payload: Optional[Dict[str, object]] = None
        self.current_preprocessed_run_channels: List[ChannelData] = []
        self.current_features_wide_df = pd.DataFrame()
        self.current_features_long_df = pd.DataFrame()
        self.current_feature_source_kind: str = ""
        self.current_feature_segment_id: Optional[int] = None
        self.current_feature_run_id: Optional[int] = None
        self.parsed: Optional[ParsedFile] = None
        self.current_interval: Optional[Tuple[float, float]] = None
        self.results_df = pd.DataFrame()
        self.xline_df = pd.DataFrame()
        self.full_slice_df = pd.DataFrame()
        self.slice_preview_df = pd.DataFrame()
        self.graph_style: Dict[str, object] = dict(DEFAULT_GRAPH_STYLE)
        self.segment_form_data: Dict[str, str] = {
            "segment_name": "",
            "gas_name": "",
            "concentration_ppm": "",
            "temperature_c": "",
            "light_mode": "",
            "is_reference": "0",
            "comment": "",
        }
        self.table_file_labels: List[QLabel] = []

        self._build_ui()
        self._create_menu()
        self._restore_config_to_ui()
        self._update_ui_state()

    # ------------------------------------------------------------- build UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.main_sections = QTabWidget()
        self.project_page = self._build_project_page()
        self.measurement_page = self._build_measurement_page()
        self.segments_page = self._build_segments_page()
        self.preprocessing_page = self._build_preprocessing_page()
        self.analytics_page = self._build_analytics_page()
        self.models_page = self._build_placeholder_page(
            "Модели",
            "Этот раздел зарезервирован под будущие ML-модели и прогнозирование по обученной базе сегментов.",
        )
        self.main_sections.addTab(self.project_page, "Проект")
        self.main_sections.addTab(self.measurement_page, "Измерения и расчёт")
        self.main_sections.addTab(self.segments_page, "Сегменты")
        self.main_sections.addTab(self.preprocessing_page, "Предобработка")
        self.main_sections.addTab(self.analytics_page, "Аналитика")
        self.main_sections.addTab(self.models_page, "Модели")

        root.addWidget(self.main_sections, stretch=1)
        self._build_status_bar()

    def _build_project_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        top = QHBoxLayout()
        project_box = QGroupBox("Проект SQLite")
        project_layout = QVBoxLayout(project_box)
        self.new_project_btn = QPushButton("Новый проект")
        self.new_project_btn.clicked.connect(self.create_project)
        self.open_project_btn = QPushButton("Открыть проект")
        self.open_project_btn.clicked.connect(self.open_project)
        self.export_project_btn = QPushButton("Экспорт проекта (.sqlite)")
        self.export_project_btn.clicked.connect(self.export_project_copy)
        self.export_tables_btn = QPushButton("Экспорт таблиц БД")
        self.export_tables_btn.clicked.connect(self.export_database_tables)
        self.import_tables_xlsx_btn = QPushButton("Импорт таблиц из XLSX")
        self.import_tables_xlsx_btn.clicked.connect(self.import_database_tables_xlsx)
        self.import_tables_csv_btn = QPushButton("Импорт таблиц из папки CSV")
        self.import_tables_csv_btn.clicked.connect(self.import_database_tables_csv)
        self.project_path_label = QLabel("Проект не открыт")
        self.project_path_label.setWordWrap(True)
        self.project_stats_label = QLabel("Файлов: — | Сегментов: —")
        self.project_stats_label.setStyleSheet("color:#555;")
        for w in [self.new_project_btn, self.open_project_btn, self.export_project_btn, self.export_tables_btn, self.import_tables_xlsx_btn, self.import_tables_csv_btn, self.project_path_label, self.project_stats_label]:
            project_layout.addWidget(w)
        top.addWidget(project_box)

        src_box = QGroupBox("Импортированные исходные файлы")
        src_layout = QVBoxLayout(src_box)
        self.project_files_table = QTableWidget()
        TableUtils.setup_table(self.project_files_table)
        src_btns = QHBoxLayout()
        self.refresh_project_btn = QPushButton("Обновить списки")
        self.refresh_project_btn.clicked.connect(self.refresh_project_views)
        self.load_source_btn = QPushButton("Показать выбранный файл")
        self.load_source_btn.clicked.connect(self.load_selected_source_file)
        src_btns.addWidget(self.refresh_project_btn)
        src_btns.addWidget(self.load_source_btn)
        src_btns.addStretch(1)
        src_layout.addWidget(self.project_files_table)
        src_layout.addLayout(src_btns)
        top.addWidget(src_box, stretch=1)

        layout.addLayout(top)
        return page

    def _build_measurement_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        main_splitter = self._build_main_splitter()
        layout.addLayout(self._build_top_controls())
        layout.addWidget(self._build_segment_label_group())
        layout.addWidget(main_splitter, stretch=1)
        return page

    def _build_segments_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        info = QLabel("Сегменты, сохранённые в БД проекта. Отсюда их можно открыть на графике, удалить или экспортировать таблицы.")
        info.setStyleSheet("color:#555;")
        layout.addWidget(info)

        self.segments_table = QTableWidget()
        TableUtils.setup_table(self.segments_table)
        layout.addWidget(self.segments_table)

        btns = QHBoxLayout()
        self.refresh_segments_btn = QPushButton("Обновить сегменты")
        self.refresh_segments_btn.clicked.connect(self.refresh_project_views)
        self.load_segment_btn = QPushButton("Показать сегмент на графике")
        self.load_segment_btn.clicked.connect(self.load_selected_segment)
        self.delete_segment_btn = QPushButton("Удалить сегмент")
        self.delete_segment_btn.clicked.connect(self.delete_selected_segment)
        btns.addWidget(self.refresh_segments_btn)
        btns.addWidget(self.load_segment_btn)
        btns.addWidget(self.delete_segment_btn)
        btns.addStretch(1)
        layout.addLayout(btns)
        return page

    def _build_preprocessing_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        top = QHBoxLayout()

        left_box = QGroupBox("Выбор сегмента для предобработки")
        left_layout = QVBoxLayout(left_box)
        self.pre_segments_table = QTableWidget()
        TableUtils.setup_table(self.pre_segments_table)
        self.pre_segments_table.setMaximumHeight(180)
        left_layout.addWidget(QLabel("Сегменты из БД проекта:"))
        left_layout.addWidget(self.pre_segments_table)

        preprocess_hint = QLabel(
            "Профили и ручной выбор референса убраны. На следующем этапе здесь появится конструктор алгоритма обработки."
        )
        preprocess_hint.setWordWrap(True)
        preprocess_hint.setStyleSheet("color:#555;")
        left_layout.addWidget(preprocess_hint)

        ops_box = QGroupBox("Конструктор алгоритма предобработки")
        ops_layout = QVBoxLayout(ops_box)

        constructor_layout = QHBoxLayout()
        operation_buttons_box = QGroupBox("Операции")
        operation_buttons_layout = QVBoxLayout(operation_buttons_box)
        self.pre_algorithm_ops: List[str] = []
        self.pre_operation_buttons: Dict[str, QPushButton] = {}
        for op_key, op_label in OPERATIONS:
            btn = QPushButton(op_label)
            btn.clicked.connect(lambda _checked=False, key=op_key: self._add_preprocess_operation(key))
            self.pre_operation_buttons[op_key] = btn
            operation_buttons_layout.addWidget(btn)
        operation_buttons_layout.addStretch(1)
        constructor_layout.addWidget(operation_buttons_box, 1)

        algorithm_order_box = QGroupBox("Порядок обработки")
        algorithm_order_layout = QVBoxLayout(algorithm_order_box)
        self.pre_algorithm_chain_label = QLabel("R")
        self.pre_algorithm_chain_label.setWordWrap(True)
        self.pre_algorithm_chain_label.setStyleSheet("font-weight:600; color:#333;")
        algorithm_order_layout.addWidget(self.pre_algorithm_chain_label)

        order_actions = QHBoxLayout()
        self.pre_remove_last_op_btn = QPushButton("Убрать последнюю")
        self.pre_remove_last_op_btn.clicked.connect(self._remove_last_preprocess_operation)
        self.pre_clear_algorithm_btn = QPushButton("Очистить")
        self.pre_clear_algorithm_btn.clicked.connect(self._clear_preprocess_algorithm)
        order_actions.addWidget(self.pre_remove_last_op_btn)
        order_actions.addWidget(self.pre_clear_algorithm_btn)
        algorithm_order_layout.addLayout(order_actions)
        algorithm_order_layout.addStretch(1)
        constructor_layout.addWidget(algorithm_order_box, 1)

        ops_layout.addLayout(constructor_layout)
        ops_layout.addWidget(QLabel("Алгоритм:"))
        self.pre_algorithm_edit = QLineEdit("R")
        self.pre_algorithm_edit.setReadOnly(True)
        self.pre_algorithm_edit.setToolTip("Пока поле заполняется автоматически. Ручной ввод будет подключён на этапе парсера алгоритма.")
        ops_layout.addWidget(self.pre_algorithm_edit)
        self._refresh_preprocess_algorithm_display()
        left_layout.addWidget(ops_box)

        run_btns = QHBoxLayout()
        self.pre_apply_btn = QPushButton("Применить к выбранному сегменту")
        self.pre_apply_btn.clicked.connect(self.apply_preprocess_to_selected_segment)
        self.pre_save_run_btn = QPushButton("Сохранить результат в БД")
        self.pre_save_run_btn.clicked.connect(self.save_current_preprocess_run)
        run_btns.addWidget(self.pre_apply_btn)
        run_btns.addWidget(self.pre_save_run_btn)
        left_layout.addLayout(run_btns)

        feature_btns = QHBoxLayout()
        self.pre_compute_features_btn = QPushButton("Рассчитать признаки")
        self.pre_compute_features_btn.clicked.connect(self.compute_features_for_current_preprocess)
        self.pre_save_features_btn = QPushButton("Сохранить признаки в БД")
        self.pre_save_features_btn.clicked.connect(self.save_current_feature_set)
        feature_btns.addWidget(self.pre_compute_features_btn)
        feature_btns.addWidget(self.pre_save_features_btn)
        left_layout.addLayout(feature_btns)
        top.addWidget(left_box, 1)

        right_box = QGroupBox("Предпросмотр предобработки")
        right_layout = QVBoxLayout(right_box)
        self.preprocess_summary_label = QLabel("Сначала выберите сегмент из БД и операции предобработки.")
        self.preprocess_summary_label.setWordWrap(True)
        self.preprocess_summary_label.setStyleSheet("color:#555;")
        self.preprocess_plot_widget = FastPlotWidget()
        self.preprocess_plot_widget.set_selection_mode(False)
        right_layout.addWidget(self.preprocess_summary_label)
        right_layout.addWidget(self.preprocess_plot_widget, stretch=1)
        top.addWidget(right_box, 2)

        layout.addLayout(top)

        runs_box = QGroupBox("Сохранённые результаты предобработки")
        runs_layout = QVBoxLayout(runs_box)
        self.pre_runs_table = QTableWidget()
        TableUtils.setup_table(self.pre_runs_table)
        runs_layout.addWidget(self.pre_runs_table)
        run_actions = QHBoxLayout()
        self.pre_refresh_btn = QPushButton("Обновить списки")
        self.pre_refresh_btn.clicked.connect(self.refresh_preprocessing_views)
        self.pre_open_run_btn = QPushButton("Открыть результат на графике")
        self.pre_open_run_btn.clicked.connect(self.load_selected_preprocessed_run)
        run_actions.addWidget(self.pre_refresh_btn)
        run_actions.addWidget(self.pre_open_run_btn)
        run_actions.addStretch(1)
        runs_layout.addLayout(run_actions)
        layout.addWidget(runs_box)

        feature_box = QGroupBox("Признаки текущего результата")
        feature_layout = QVBoxLayout(feature_box)
        self.current_features_table = QTableWidget()
        TableUtils.setup_table(self.current_features_table)
        feature_layout.addWidget(self.current_features_table)
        layout.addWidget(feature_box)
        return page

    def _build_analytics_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        info = QLabel("На этом этапе доступны наборы признаков и PCA-анализ по рассчитанным признакам. LDA будет следующим этапом.")
        info.setStyleSheet("color:#555;")
        info.setWordWrap(True)
        layout.addWidget(info)

        filter_box = QGroupBox("Выборка для PCA")
        filter_layout = QGridLayout(filter_box)
        self.pca_source_kind_combo = QComboBox()
        self.pca_source_kind_combo.addItems(["Все источники", "segment", "preprocessed_run"])
        self.pca_gas_filter_edit = QLineEdit()
        self.pca_temp_filter_edit = QLineEdit()
        self.pca_light_filter_combo = QComboBox()
        self.pca_light_filter_combo.addItems(["Все", "", "OFF", "ON"])
        self.pca_scaling_combo = QComboBox()
        self.pca_scaling_combo.addItems(list(SCALE_MODES.keys()))
        self.pca_refresh_btn = QPushButton("Обновить выборку")
        self.pca_refresh_btn.clicked.connect(self.refresh_feature_views)
        self.pca_run_btn = QPushButton("Рассчитать PCA")
        self.pca_run_btn.clicked.connect(self.run_pca_on_filtered_feature_sets)
        filter_layout.addWidget(QLabel("Источник признаков:"), 0, 0)
        filter_layout.addWidget(self.pca_source_kind_combo, 0, 1)
        filter_layout.addWidget(QLabel("Газ содержит:"), 0, 2)
        filter_layout.addWidget(self.pca_gas_filter_edit, 0, 3)
        filter_layout.addWidget(QLabel("Температура содержит:"), 1, 0)
        filter_layout.addWidget(self.pca_temp_filter_edit, 1, 1)
        filter_layout.addWidget(QLabel("Свет:"), 1, 2)
        filter_layout.addWidget(self.pca_light_filter_combo, 1, 3)
        filter_layout.addWidget(QLabel("Масштабирование:"), 2, 0)
        filter_layout.addWidget(self.pca_scaling_combo, 2, 1)
        filter_layout.addWidget(self.pca_refresh_btn, 2, 2)
        filter_layout.addWidget(self.pca_run_btn, 2, 3)
        layout.addWidget(filter_box)

        self.features_sets_table = QTableWidget()
        TableUtils.setup_table(self.features_sets_table)
        layout.addWidget(self.features_sets_table)

        actions = QHBoxLayout()
        self.features_refresh_btn = QPushButton("Обновить наборы признаков")
        self.features_refresh_btn.clicked.connect(self.refresh_feature_views)
        self.features_load_btn = QPushButton("Показать признаки")
        self.features_load_btn.clicked.connect(self.load_selected_feature_set)
        actions.addWidget(self.features_refresh_btn)
        actions.addWidget(self.features_load_btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.pca_plot_widget = pg.PlotWidget()
        self.pca_plot_widget.setBackground("w")
        self.pca_plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.pca_plot_widget.getPlotItem().setLabel("bottom", "PC1")
        self.pca_plot_widget.getPlotItem().setLabel("left", "PC2")
        layout.addWidget(self.pca_plot_widget, stretch=1)

        self.analytics_tabs = QTabWidget()
        self.features_values_table = QTableWidget()
        TableUtils.setup_table(self.features_values_table)
        self.pca_variance_table = QTableWidget()
        TableUtils.setup_table(self.pca_variance_table)
        self.pca_scores_table = QTableWidget()
        TableUtils.setup_table(self.pca_scores_table)
        self.pca_loadings_table = QTableWidget()
        TableUtils.setup_table(self.pca_loadings_table)

        for title, table in [
            ("Значения признаков", self.features_values_table),
            ("PCA: дисперсия", self.pca_variance_table),
            ("PCA: scores", self.pca_scores_table),
            ("PCA: loadings", self.pca_loadings_table),
        ]:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_layout.addWidget(table)
            self.analytics_tabs.addTab(tab, title)
        layout.addWidget(self.analytics_tabs)
        return page

    def _build_placeholder_page(self, title: str, text: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        box = QGroupBox(title)
        box_layout = QVBoxLayout(box)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("font-size:11pt; color:#444;")
        box_layout.addWidget(label)
        layout.addWidget(box)
        layout.addStretch(1)
        return page

    def _build_segment_label_group(self) -> QWidget:
        box = QGroupBox("Сегмент для БД")
        layout = QGridLayout(box)
        self.segment_summary_label = QLabel("Детали участка не заполнены.")
        self.segment_summary_label.setWordWrap(True)
        self.segment_summary_label.setStyleSheet("color:#444;")
        self.segment_details_btn = QPushButton("Детали участка")
        self.segment_details_btn.clicked.connect(self.open_segment_details_dialog)
        self.save_segment_btn = QPushButton("Сохранить выделенный сегмент в БД")
        self.save_segment_btn.clicked.connect(self.save_current_segment_to_db)
        self.clear_segment_form_btn = QPushButton("Очистить детали")
        self.clear_segment_form_btn.clicked.connect(self.clear_segment_form)
        layout.addWidget(self.segment_summary_label, 0, 0, 1, 4)
        layout.addWidget(self.segment_details_btn, 1, 0, 1, 1)
        layout.addWidget(self.save_segment_btn, 1, 1, 1, 2)
        layout.addWidget(self.clear_segment_form_btn, 1, 3, 1, 1)
        return box

    def _build_top_controls(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(8)

        # File
        file_box = QGroupBox("Файл")
        file_layout = QVBoxLayout(file_box)
        self.load_btn = QPushButton("Загрузить TXT")
        self.load_btn.clicked.connect(self.load_txt_file)
        self.plot_all_btn = QPushButton("Построить графики")
        self.plot_all_btn.clicked.connect(self.plot_all_graphs)
        self.save_plot_btn = QPushButton("Сохранить график")
        self.save_plot_btn.clicked.connect(self.export_plot_dialog)
        self.graph_style_btn = QPushButton("Оформление графика")
        self.graph_style_btn.clicked.connect(self.open_graph_style_dialog)
        self.point_mode_check = QCheckBox("Точечное представление")
        self.point_mode_check.setChecked(False)
        self.point_mode_check.stateChanged.connect(self.on_view_changed)
        self.show_grid_check = QCheckBox("Сетка")
        self.show_grid_check.setChecked(True)
        self.show_grid_check.stateChanged.connect(self.on_view_changed)
        self.auto_range_btn = QPushButton("Автомасштаб")
        self.auto_range_btn.clicked.connect(self.plot_widget.auto_range)
        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(self.plot_all_btn)
        file_layout.addWidget(self.save_plot_btn)
        file_layout.addWidget(self.graph_style_btn)
        file_layout.addWidget(self.point_mode_check)
        file_layout.addWidget(self.show_grid_check)
        file_layout.addWidget(self.auto_range_btn)
        layout.addWidget(file_box)

        # Interval
        interval_box = QGroupBox("Участок")
        interval_layout = QGridLayout(interval_box)
        validator = QDoubleValidator(-1e15, 1e15, 6)
        self.start_edit = QLineEdit()
        self.end_edit = QLineEdit()
        self.start_edit.setValidator(validator)
        self.end_edit.setValidator(validator)
        self.selection_mode_check = QCheckBox("Режим выбора участка мышью")
        self.selection_mode_check.setChecked(True)
        self.selection_mode_check.stateChanged.connect(self.on_selection_mode_changed)
        self.plot_selected_btn = QPushButton("Построить участок")
        self.plot_selected_btn.clicked.connect(self.plot_selected_interval)
        self.reset_interval_btn = QPushButton("Сбросить участок")
        self.reset_interval_btn.clicked.connect(self.reset_interval)
        self.export_slice_btn = QPushButton("Сохранить участок")
        self.export_slice_btn.clicked.connect(self.export_selected_slice)
        interval_layout.addWidget(QLabel("Начало, s:"), 0, 0)
        interval_layout.addWidget(self.start_edit, 0, 1)
        interval_layout.addWidget(QLabel("Конец, s:"), 1, 0)
        interval_layout.addWidget(self.end_edit, 1, 1)
        interval_layout.addWidget(self.selection_mode_check, 2, 0, 1, 2)
        interval_layout.addWidget(self.plot_selected_btn, 3, 0, 1, 2)
        interval_layout.addWidget(self.reset_interval_btn, 4, 0, 1, 2)
        interval_layout.addWidget(self.export_slice_btn, 5, 0, 1, 2)
        layout.addWidget(interval_box)

        # Calc
        calc_box = QGroupBox("Расчёт")
        calc_layout = QGridLayout(calc_box)
        self.variant_combo = QComboBox()
        self.variant_combo.addItems(list(CALC_MODES.keys()))
        self.avg_points_spin = QSpinBox()
        self.avg_points_spin.setRange(1, 200)
        self.avg_points_spin.setValue(1)
        self.calculate_btn = QPushButton("Выполнить расчёт")
        self.calculate_btn.clicked.connect(self.calculate_results_action)
        self.export_results_btn = QPushButton("Сохранить результаты")
        self.export_results_btn.clicked.connect(self.export_results)
        self.copy_results_btn = QPushButton("Копировать в Excel")
        self.copy_results_btn.clicked.connect(lambda: self.copy_table(self.results_table))
        calc_layout.addWidget(QLabel("Вариант:"), 0, 0)
        calc_layout.addWidget(self.variant_combo, 0, 1)
        calc_layout.addWidget(QLabel("Усреднение N:"), 1, 0)
        calc_layout.addWidget(self.avg_points_spin, 1, 1)
        calc_layout.addWidget(self.calculate_btn, 2, 0, 1, 2)
        calc_layout.addWidget(self.export_results_btn, 3, 0, 1, 2)
        calc_layout.addWidget(self.copy_results_btn, 4, 0, 1, 2)
        layout.addWidget(calc_box)

        # X-line
        xline_box = QGroupBox("Точка на графике")
        xline_layout = QGridLayout(xline_box)
        self.xline_edit = QLineEdit()
        self.xline_edit.setValidator(QDoubleValidator(-1e15, 1e15, 6))
        self.build_xline_btn = QPushButton("Поставить линию")
        self.build_xline_btn.clicked.connect(self.build_xline)
        self.clear_xline_btn = QPushButton("Убрать линию")
        self.clear_xline_btn.clicked.connect(self.clear_xline)
        self.copy_xline_btn = QPushButton("Копировать Rᵢ")
        self.copy_xline_btn.clicked.connect(lambda: self.copy_table(self.xline_table))
        self.current_xline_label = QLabel("Rᵢ: —")
        self.current_xline_label.setStyleSheet("font-weight:600;")
        xline_layout.addWidget(QLabel("Время tᵢ, s:"), 0, 0)
        xline_layout.addWidget(self.xline_edit, 0, 1)
        xline_layout.addWidget(self.build_xline_btn, 1, 0, 1, 2)
        xline_layout.addWidget(self.clear_xline_btn, 2, 0, 1, 2)
        xline_layout.addWidget(self.copy_xline_btn, 3, 0, 1, 2)
        xline_layout.addWidget(self.current_xline_label, 4, 0, 1, 2)
        layout.addWidget(xline_box)

        # View
        view_box = QGroupBox("Вид / оси")
        view_layout = QGridLayout(view_box)
        self.yscale_combo = QComboBox()
        self.yscale_combo.addItems(["log", "linear"])
        self.yscale_combo.currentTextChanged.connect(self.on_view_changed)
        self.x_axis_name_edit = QLineEdit("Time")
        self.y_axis_name_edit = QLineEdit("Resistance")
        self.x_axis_name_edit.editingFinished.connect(self.on_view_changed)
        self.y_axis_name_edit.editingFinished.connect(self.on_view_changed)
        view_layout.addWidget(QLabel("Ось Y:"), 0, 0)
        view_layout.addWidget(self.yscale_combo, 0, 1)
        view_layout.addWidget(QLabel("Подпись X:"), 1, 0)
        view_layout.addWidget(self.x_axis_name_edit, 1, 1)
        view_layout.addWidget(QLabel("Подпись Y:"), 2, 0)
        view_layout.addWidget(self.y_axis_name_edit, 2, 1)
        layout.addWidget(view_box)

        self.file_info_box = QGroupBox("Загруженный файл")
        file_info_layout = QVBoxLayout(self.file_info_box)
        self.file_name_label = QLabel("Файл не загружен")
        self.file_name_label.setWordWrap(True)
        self.file_meta_label = QLabel("—")
        self.file_meta_label.setWordWrap(True)
        self.file_meta_label.setStyleSheet("color:#555;")
        self.open_source_from_current_btn = QPushButton("Показать исходный файл")
        self.open_source_from_current_btn.clicked.connect(self.open_current_source_from_measurement_view)
        file_info_layout.addWidget(self.file_name_label)
        file_info_layout.addWidget(self.file_meta_label)
        file_info_layout.addWidget(self.open_source_from_current_btn)
        layout.addWidget(self.file_info_box)

        layout.addStretch(1)
        return layout

    def _build_main_splitter(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Vertical)

        top = QSplitter(Qt.Orientation.Horizontal)
        self.plot_widget = FastPlotWidget()
        self.plot_widget.intervalChanged.connect(self.on_plot_interval_changed)
        self.plot_widget.cursorChanged.connect(self.on_cursor_changed)
        self.plot_widget.xLineMoved.connect(self.on_xline_moved)
        self.channel_legend = ChannelLegendWidget()
        self.channel_legend.changed.connect(self.on_channels_changed)
        top.addWidget(self.plot_widget)
        top.addWidget(self.channel_legend)
        top.setStretchFactor(0, 5)
        top.setStretchFactor(1, 1)
        splitter.addWidget(top)

        tabs_host = QWidget()
        tabs_layout = QVBoxLayout(tabs_host)
        tabs_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()

        self.results_table = self._make_table_tab(self.tabs, "Результаты расчёта", self.export_results)
        self.xline_table = self._make_table_tab(self.tabs, "Значения в точке tᵢ", self.export_xline_table)
        self.slice_table = self._make_table_tab(self.tabs, "Выделенный участок", self.export_selected_slice, index_visible=False, hint="Предпросмотр первых 300 строк выделенного участка.")
        tabs_layout.addWidget(self.tabs)
        splitter.addWidget(tabs_host)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        return splitter

    def _make_table_tab(
        self,
        tabs: QTabWidget,
        title: str,
        export_slot,
        index_visible: bool = True,
        hint: str | None = None,
    ) -> QTableWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        file_label = QLabel("Файл: —")
        file_label.setStyleSheet("font-weight:600;")
        layout.addWidget(file_label)
        self.table_file_labels.append(file_label)
        if hint:
            label = QLabel(hint)
            label.setStyleSheet("color:#666;")
            layout.addWidget(label)
        table = QTableWidget()
        table.setProperty("index_visible", index_visible)
        TableUtils.setup_table(table)
        btns = QHBoxLayout()
        copy_btn = QPushButton("Копировать в буфер")
        copy_btn.clicked.connect(lambda: self.copy_table(table))
        export_btn = QPushButton("Экспорт таблицы")
        export_btn.clicked.connect(export_slot)
        btns.addWidget(copy_btn)
        btns.addWidget(export_btn)
        btns.addStretch(1)
        layout.addWidget(table)
        layout.addLayout(btns)
        tabs.addTab(page, title)
        return table

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        self.status_file = QLabel("Файл: —")
        self.status_points = QLabel("Точек: —")
        self.status_cursor = QLabel("x: — ; y: —")
        status.addWidget(self.status_file, 1)
        status.addPermanentWidget(self.status_points)
        status.addPermanentWidget(self.status_cursor)

    def _create_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("Файл")
        new_project_action = QAction("Новый проект", self)
        new_project_action.triggered.connect(self.create_project)
        open_project_action = QAction("Открыть проект", self)
        open_project_action.triggered.connect(self.open_project)
        open_action = QAction("Загрузить TXT в проект", self)
        open_action.triggered.connect(self.load_txt_file)
        save_plot_action = QAction("Сохранить график", self)
        save_plot_action.triggered.connect(self.export_plot_dialog)
        style_action = QAction("Оформление графика", self)
        style_action.triggered.connect(self.open_graph_style_dialog)
        export_db_action = QAction("Экспорт проекта (.sqlite)", self)
        export_db_action.triggered.connect(self.export_project_copy)
        export_tables_action = QAction("Экспорт таблиц БД", self)
        export_tables_action.triggered.connect(self.export_database_tables)
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(new_project_action)
        file_menu.addAction(open_project_action)
        file_menu.addSeparator()
        file_menu.addAction(open_action)
        file_menu.addAction(save_plot_action)
        file_menu.addAction(style_action)
        file_menu.addSeparator()
        file_menu.addAction(export_db_action)
        file_menu.addAction(export_tables_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

    # ------------------------------------------------------------- config
    def _restore_config_to_ui(self) -> None:
        if self.config.get("window_size"):
            width, height = self.config.get("window_size")
            self.resize(width, height)
        last_variant = self.config.get("last_variant")
        if last_variant in CALC_MODES:
            self.variant_combo.setCurrentText(last_variant)
        last_scale = self.config.get("y_scale", "log")
        if last_scale in ("log", "linear"):
            self.yscale_combo.setCurrentText(last_scale)
        self.avg_points_spin.setValue(int(self.config.get("avg_points", 1)))
        self.graph_style = dict(DEFAULT_GRAPH_STYLE)
        self.graph_style["font_size"] = int(self.config.get("graph_font_size", self.graph_style["font_size"]))
        self.graph_style["legend_font_size"] = int(self.config.get("legend_font_size", self.graph_style["legend_font_size"]))
        self.graph_style["legend_position"] = "Справа"
        self.graph_style["line_width"] = float(self.config.get("line_width", self.graph_style["line_width"]))
        self.graph_style["point_size"] = float(self.config.get("point_size", self.graph_style["point_size"]))
        self.graph_style["point_mode"] = bool(self.config.get("point_mode", self.graph_style["point_mode"]))
        self.graph_style["background_mode"] = self.config.get("background_mode", self.graph_style["background_mode"])
        self.graph_style["show_grid"] = bool(self.config.get("show_grid", self.graph_style["show_grid"]))
        self.graph_style["monochrome"] = bool(self.config.get("monochrome", self.graph_style["monochrome"]))
        self.graph_style["export_scale"] = float(self.config.get("export_scale", self.graph_style["export_scale"]))
        self.graph_style["export_preset"] = self.config.get("export_preset", self.graph_style["export_preset"])
        self.point_mode_check.setChecked(bool(self.graph_style.get("point_mode", False)))
        self.show_grid_check.setChecked(bool(self.graph_style.get("show_grid", True)))
        self.selection_mode_check.setChecked(bool(self.config.get("selection_mode", True)))
        self.x_axis_name_edit.setText(self.config.get("x_axis_name", "Time"))
        self.y_axis_name_edit.setText(self.config.get("y_axis_name", "Resistance"))
        self.on_view_changed()
        self.on_selection_mode_changed()

    def _save_config_from_ui(self) -> None:
        self.config.set("window_size", [self.width(), self.height()])
        self.config.set("last_variant", self.variant_combo.currentText())
        self.config.set("y_scale", self.yscale_combo.currentText())
        self.config.set("avg_points", self.avg_points_spin.value())
        self.config.set("show_grid", bool(self.graph_style.get("show_grid", self.show_grid_check.isChecked())))
        self.config.set("selection_mode", self.selection_mode_check.isChecked())
        self.config.set("x_axis_name", self.x_axis_name_edit.text().strip() or "Time")
        self.config.set("y_axis_name", self.y_axis_name_edit.text().strip() or "Resistance")
        self.config.set("graph_font_size", int(self.graph_style.get("font_size", 10)))
        self.config.set("legend_font_size", int(self.graph_style.get("legend_font_size", 11)))
        self.config.set("legend_position", self.graph_style.get("legend_position", "Справа"))
        self.config.set("line_width", float(self.graph_style.get("line_width", 1.5)))
        self.config.set("point_size", float(self.graph_style.get("point_size", 4.0)))
        self.config.set("point_mode", bool(self.graph_style.get("point_mode", False)))
        self.config.set("background_mode", self.graph_style.get("background_mode", "Белый"))
        self.config.set("monochrome", bool(self.graph_style.get("monochrome", False)))
        self.config.set("export_scale", float(self.graph_style.get("export_scale", 2.0)))
        self.config.set("export_preset", self.graph_style.get("export_preset", "Обычный PNG"))
        self.config.set("last_dir", self.config.get("last_dir", str(Path.home())))
        if self.parsed:
            self.config.set("channel_names", self.channel_legend.collect_names())
        self.config.save()

    # ------------------------------------------------------------- helpers
    def _set_status(self, text: str) -> None:
        self.statusBar().showMessage(text, 5000)

    def _update_ui_state(self) -> None:
        project_open = self.db.is_open()
        has_data = self.parsed is not None
        has_interval = self._get_interval_values() is not None
        self.load_btn.setEnabled(project_open)
        self.plot_all_btn.setEnabled(has_data)
        self.save_plot_btn.setEnabled(has_data)
        self.graph_style_btn.setEnabled(True)
        self.plot_selected_btn.setEnabled(has_data)
        self.reset_interval_btn.setEnabled(has_data)
        self.export_slice_btn.setEnabled(has_data)
        self.calculate_btn.setEnabled(has_data)
        self.export_results_btn.setEnabled(not self.results_df.empty)
        self.copy_results_btn.setEnabled(not self.results_df.empty)
        self.build_xline_btn.setEnabled(has_data)
        self.clear_xline_btn.setEnabled(self.plot_widget.current_xline() is not None)
        self.copy_xline_btn.setEnabled(not self.xline_df.empty)
        self.save_segment_btn.setEnabled(project_open and has_data and has_interval)
        self.export_project_btn.setEnabled(project_open)
        self.export_tables_btn.setEnabled(project_open)
        self.import_tables_xlsx_btn.setEnabled(project_open)
        self.import_tables_csv_btn.setEnabled(project_open)
        self.refresh_project_btn.setEnabled(project_open)
        self.load_source_btn.setEnabled(project_open)
        self.refresh_segments_btn.setEnabled(project_open)
        self.load_segment_btn.setEnabled(project_open)
        self.delete_segment_btn.setEnabled(project_open)
        self.open_source_from_current_btn.setEnabled(project_open and self.current_source_file_id is not None)
        self.pre_refresh_btn.setEnabled(project_open)
        self.pre_apply_btn.setEnabled(project_open)
        self.pre_save_run_btn.setEnabled(project_open and bool(self.current_preprocessed_run_channels))
        self.pre_compute_features_btn.setEnabled(project_open)
        self.pre_save_features_btn.setEnabled(project_open and not self.current_features_wide_df.empty)
        self.pre_open_run_btn.setEnabled(project_open)
        self.features_refresh_btn.setEnabled(project_open)
        self.features_load_btn.setEnabled(project_open)
        self.pca_refresh_btn.setEnabled(project_open)
        self.pca_run_btn.setEnabled(project_open)

    def _all_channels_synced(self) -> List[ChannelData]:
        if not self.parsed:
            return []
        return self.channel_legend.sync_to_channels(self.parsed.channels)

    def _visible_channels(self) -> List[ChannelData]:
        if not self.parsed:
            return []
        return self.channel_legend.visible_channels(self.parsed.channels)

    def _get_interval_values(self) -> Optional[Tuple[float, float]]:
        start_text = self.start_edit.text().strip().replace(",", ".")
        end_text = self.end_edit.text().strip().replace(",", ".")
        if not start_text or not end_text:
            return None
        return normalize_interval(start_text, end_text)

    def _sync_file_info(self) -> None:
        if not self.parsed:
            self.file_name_label.setText("Файл не загружен")
            self.file_meta_label.setText("—")
            self.status_file.setText("Файл: —")
            self.status_points.setText("Точек: —")
            for label in self.table_file_labels:
                label.setText("Файл: —")
            return
        meta_preview = "; ".join(self.parsed.metadata_lines[:3]) if self.parsed.metadata_lines else "Метаданные не найдены"
        if self.parsed.notes:
            meta_preview += " | " + " | ".join(self.parsed.notes)
        self.file_name_label.setText(f"Файл: {self.parsed.file_name}")
        self.file_meta_label.setText(meta_preview)
        self.status_file.setText(f"Файл: {self.parsed.file_name}")
        self.status_points.setText(f"Строк: {self.parsed.raw_row_count} | Столбцов: {self.parsed.column_count}")
        for label in self.table_file_labels:
            label.setText(f"Файл: {self.parsed.file_name}")

    def _current_file_stem(self, fallback: str) -> str:
        if self.parsed and self.parsed.file_path:
            return self.parsed.file_path.stem
        return fallback

    def _ensure_project_open(self) -> bool:
        if self.db.is_open():
            return True
        QMessageBox.information(self, "Проект", "Сначала создайте или откройте проект SQLite во вкладке «Проект». ")
        self.main_sections.setCurrentWidget(self.project_page)
        return False

    def _update_project_info(self) -> None:
        info = self.db.project_info()
        if info is None:
            self.project_path_label.setText("Проект не открыт")
            self.project_stats_label.setText("Файлов: — | Сегментов: —")
            TableUtils.populate_from_dataframe(self.project_files_table, pd.DataFrame(), index_visible=False)
            TableUtils.populate_from_dataframe(self.segments_table, pd.DataFrame(), index_visible=False)
            return
        self.project_path = info.path
        self.project_path_label.setText(f"Проект: {info.path}")
        self.project_stats_label.setText(f"Файлов: {info.source_files_count} | Сегментов: {info.segments_count}")

    def refresh_project_views(self) -> None:
        self._update_project_info()
        if not self.db.is_open():
            self._update_ui_state()
            return
        files_df = self.db.list_source_files()
        segments_df = self.db.list_segments()
        if not files_df.empty:
            files_df = files_df.rename(
                columns={
                    "id": "ID",
                    "file_name": "Файл",
                    "file_path": "Источник",
                    "row_count": "Строк",
                    "channel_count": "Каналов",
                }
            )[["ID", "Файл", "Источник", "Строк", "Каналов"]]
        if not segments_df.empty:
            segments_df = segments_df.copy()
            segments_df["is_reference"] = segments_df["is_reference"].fillna(0).astype(int).map({1: "Да", 0: "Нет"})
            segments_df = segments_df.rename(
                columns={
                    "id": "ID",
                    "segment_name": "Сегмент",
                    "source_file_name": "Источник",
                    "t_start": "Начало, s",
                    "t_end": "Конец, s",
                    "points_count": "Точек",
                    "gas_name": "Газ",
                    "concentration_ppm": "Концентрация, ppm",
                    "temperature_c": "Температура, °C",
                    "light_mode": "Свет",
                    "is_reference": "Референс",
                    "comment": "Комментарий",
                }
            )[["ID", "Сегмент", "Источник", "Начало, s", "Конец, s", "Точек", "Газ", "Концентрация, ppm", "Температура, °C", "Свет", "Референс", "Комментарий"]]
        TableUtils.populate_from_dataframe(self.project_files_table, files_df, index_visible=False)
        TableUtils.populate_from_dataframe(self.segments_table, segments_df, index_visible=False)
        self.refresh_preprocessing_views(segments_df)
        self._update_ui_state()

    def _collect_current_segment_channels(self, interval: Tuple[float, float]) -> List[ChannelData]:
        channels: List[ChannelData] = []
        for ch in self._visible_channels():
            time, resistance = slice_channel(ch, interval)
            if len(time) == 0:
                continue
            channels.append(
                ChannelData(
                    index=ch.index,
                    original_name=ch.original_name,
                    display_name=ch.display_name,
                    time=time,
                    resistance=resistance,
                    color=ch.color,
                )
            )
        return channels

    def _collect_all_current_channels(self) -> List[ChannelData]:
        if not self.parsed:
            return []
        channels: List[ChannelData] = []
        for ch in self._visible_channels():
            channels.append(
                ChannelData(
                    index=ch.index,
                    original_name=ch.original_name,
                    display_name=ch.display_name,
                    time=ch.time.copy(),
                    resistance=ch.resistance.copy(),
                    color=ch.color,
                )
            )
        return channels

    def _default_segment_name(self, interval: Tuple[float, float]) -> str:
        stem = self._current_file_stem("segment")
        return f"{stem}_{interval[0]:.1f}_{interval[1]:.1f}"

    def _update_segment_summary(self) -> None:
        details = []
        if self.segment_form_data.get("segment_name"):
            details.append(f"Сегмент: {self.segment_form_data['segment_name']}")
        if self.segment_form_data.get("gas_name"):
            details.append(f"Газ: {self.segment_form_data['gas_name']}")
        if self.segment_form_data.get("concentration_ppm"):
            details.append(f"Концентрация: {self.segment_form_data['concentration_ppm']} ppm")
        if self.segment_form_data.get("temperature_c"):
            details.append(f"Температура: {self.segment_form_data['temperature_c']} °C")
        if self.segment_form_data.get("light_mode"):
            details.append(f"Свет: {self.segment_form_data['light_mode']}")
        if str(self.segment_form_data.get("is_reference", "0")) in {"1", "true", "True", "Да", "да"}:
            details.append("Референс: Да")
        if self.segment_form_data.get("comment"):
            details.append(f"Комментарий: {self.segment_form_data['comment']}")
        self.segment_summary_label.setText(" | ".join(details) if details else "Детали участка не заполнены.")

    def _refresh_plot(self) -> None:
        if not self.parsed:
            return
        interval = self.current_interval if self.current_interval is not None else None
        style_payload = self._current_graph_style_payload()
        self.plot_widget.set_log_mode(self.yscale_combo.currentText() == "log")
        self.plot_widget.apply_graph_style(style_payload)
        self.plot_widget.redraw(self._visible_channels(), interval=interval)

    def _set_interval_fields(self, start: float, end: float) -> None:
        start, end = sorted((float(start), float(end)))
        self.start_edit.setText(f"{start:.4f}")
        self.end_edit.setText(f"{end:.4f}")

    def _update_slice_preview(self) -> None:
        interval = self._get_interval_values()
        channels = self._visible_channels()
        if not interval or not channels:
            self.full_slice_df = pd.DataFrame()
            self.slice_preview_df = pd.DataFrame()
        else:
            self.full_slice_df = build_slice_dataframe(channels, interval)
            self.slice_preview_df = self.full_slice_df.head(300).copy() if not self.full_slice_df.empty else pd.DataFrame()
        TableUtils.populate_from_dataframe(self.slice_table, self.slice_preview_df, index_visible=False)

    def _current_graph_style_payload(self) -> Dict[str, object]:
        payload = dict(self.graph_style)
        payload["x_axis_name"] = self.x_axis_name_edit.text().strip() or "Time"
        payload["y_axis_name"] = self.y_axis_name_edit.text().strip() or "Resistance"
        payload["show_grid"] = self.show_grid_check.isChecked()
        payload["point_mode"] = self.point_mode_check.isChecked()
        return payload

    def _apply_graph_style_from_dialog(self, style: Dict[str, object]) -> None:
        self.graph_style.update(style)
        self.x_axis_name_edit.setText(str(style.get("x_axis_name", self.x_axis_name_edit.text())))
        self.y_axis_name_edit.setText(str(style.get("y_axis_name", self.y_axis_name_edit.text())))
        self.show_grid_check.setChecked(bool(style.get("show_grid", self.show_grid_check.isChecked())))
        if self.parsed:
            self._refresh_plot()
        else:
            self.plot_widget.apply_graph_style(self._current_graph_style_payload())

    def _export_style_for_current_preset(self) -> Dict[str, object]:
        style = dict(self._current_graph_style_payload())
        preset_name = str(style.get("export_preset", "Обычный PNG"))
        overrides = EXPORT_PRESET_OVERRIDES.get(preset_name, {})
        style.update(overrides)
        style["export_preset"] = preset_name
        return style

    def _export_dataframe_dialog(self, df: pd.DataFrame, title: str, default_name: str, index: bool) -> None:
        if df is None or df.empty:
            QMessageBox.information(self, title, "Нет данных для экспорта.")
            return
        last_dir = self.config.get("last_dir", str(Path.home()))
        file_path, _ = QFileDialog.getSaveFileName(self, title, str(Path(last_dir) / default_name), EXPORT_TABLE_FILTER)
        if not file_path:
            return
        path = Path(file_path)
        if path.suffix.lower() not in {".txt", ".csv", ".xlsx"}:
            path = path.with_suffix(".txt")
        self.config.set("last_dir", str(path.parent))
        try:
            export_dataframe(df, str(path), index=index)
            QMessageBox.information(self, title, f"Файл сохранён:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, title, f"Не удалось сохранить файл:\n{exc}")

    # ------------------------------------------------------------- slots
    def on_view_changed(self) -> None:
        self.graph_style["show_grid"] = self.show_grid_check.isChecked()
        self.graph_style["point_mode"] = self.point_mode_check.isChecked()
        self.graph_style["x_axis_name"] = self.x_axis_name_edit.text().strip() or "Time"
        self.graph_style["y_axis_name"] = self.y_axis_name_edit.text().strip() or "Resistance"
        self.plot_widget.set_log_mode(self.yscale_combo.currentText() == "log")
        if self.parsed:
            self._refresh_plot()
        else:
            self.plot_widget.apply_graph_style(self._current_graph_style_payload())

    def on_selection_mode_changed(self) -> None:
        self.plot_widget.set_selection_mode(self.selection_mode_check.isChecked())

    def on_channels_changed(self) -> None:
        self._refresh_plot()
        self._update_slice_preview()
        x_line = self.plot_widget.current_xline()
        if x_line is not None:
            self.on_xline_moved(x_line)

    def on_cursor_changed(self, x: float, y: float) -> None:
        self.status_cursor.setText(f"x: {x:.4f} s ; y: {y:.4g} Ohm")

    def on_plot_interval_changed(self, start: float, end: float) -> None:
        self._set_interval_fields(start, end)
        self._update_slice_preview()
        if not self.segment_form_data.get("segment_name"):
            self.segment_form_data["segment_name"] = self._default_segment_name((start, end))
        self._update_segment_summary()
        self._update_ui_state()

    def on_xline_moved(self, x_value: float) -> None:
        self.xline_edit.setText(f"{x_value:.4f}")
        channels = self._visible_channels()
        if channels:
            self.xline_df = build_xline_dataframe(channels, x_value)
        else:
            self.xline_df = pd.DataFrame()
        TableUtils.populate_from_dataframe(self.xline_table, self.xline_df, index_visible=False)
        self.current_xline_label.setText(f"Rᵢ @ t={x_value:.4f} s")
        self._update_ui_state()

    def _operation_label(self, op_key: str) -> str:
        return dict(OPERATIONS).get(op_key, op_key)

    def _preprocess_algorithm_text(self) -> str:
        if not self.pre_algorithm_ops:
            return "R"
        return " -> ".join(self._operation_label(op) for op in self.pre_algorithm_ops)

    def _refresh_preprocess_algorithm_display(self) -> None:
        algorithm = self._preprocess_algorithm_text()
        self.pre_algorithm_chain_label.setText(algorithm)
        self.pre_algorithm_edit.setText(algorithm)
        self.pre_remove_last_op_btn.setEnabled(bool(self.pre_algorithm_ops))
        self.pre_clear_algorithm_btn.setEnabled(bool(self.pre_algorithm_ops))

    def _add_preprocess_operation(self, op_key: str) -> None:
        self.pre_algorithm_ops.append(op_key)
        self._refresh_preprocess_algorithm_display()

    def _remove_last_preprocess_operation(self) -> None:
        if self.pre_algorithm_ops:
            self.pre_algorithm_ops.pop()
        self._refresh_preprocess_algorithm_display()

    def _clear_preprocess_algorithm(self) -> None:
        self.pre_algorithm_ops.clear()
        self._refresh_preprocess_algorithm_display()

    def _build_preprocess_profile_payload(self) -> Dict[str, object]:
        operations = list(self.pre_algorithm_ops)
        algorithm = self._preprocess_algorithm_text()
        return {
            "profile_name": algorithm,
            "algorithm": algorithm,
            "operations": operations,
            "reference_segment_id": None,
        }

    def refresh_preprocessing_views(self, segments_df: Optional[pd.DataFrame] = None) -> None:
        if isinstance(segments_df, bool):
            segments_df = None
        if not self.db.is_open():
            TableUtils.populate_from_dataframe(self.pre_segments_table, pd.DataFrame(), index_visible=False)
            TableUtils.populate_from_dataframe(self.pre_runs_table, pd.DataFrame(), index_visible=False)
            TableUtils.populate_from_dataframe(self.current_features_table, pd.DataFrame(), index_visible=False)
            self.preprocess_summary_label.setText("Сначала откройте проект и выберите сегмент для предобработки.")
            self.current_preprocessed_run_id = None
            self.current_preprocessed_run_channels = []
            self.current_features_wide_df = pd.DataFrame()
            self.current_features_long_df = pd.DataFrame()
            self.refresh_feature_views()
            self._update_ui_state()
            return

        if segments_df is None:
            segments_df = self.db.list_segments()
            if not segments_df.empty:
                segments_df = segments_df.copy()
                if "is_reference" in segments_df.columns:
                    segments_df["is_reference"] = segments_df["is_reference"].fillna(0).astype(int).map({1: "Да", 0: "Нет"})
                segments_df = segments_df.rename(
                    columns={
                        "id": "ID",
                        "segment_name": "Сегмент",
                        "source_file_name": "Источник",
                        "t_start": "Начало, s",
                        "t_end": "Конец, s",
                        "gas_name": "Газ",
                        "temperature_c": "Температура, °C",
                        "is_reference": "Референс",
                    }
                )
                columns = ["ID", "Сегмент", "Источник", "Начало, s", "Конец, s", "Газ", "Температура, °C"]
                if "Референс" in segments_df.columns:
                    columns.append("Референс")
                segments_df = segments_df[columns]
        TableUtils.populate_from_dataframe(self.pre_segments_table, segments_df, index_visible=False)

        runs_df = self.db.list_preprocessed_runs()
        if not runs_df.empty:
            runs_df = runs_df.copy()
            runs_df["reference_segment_id"] = runs_df["reference_segment_id"].fillna("")
            runs_df = runs_df.rename(
                columns={
                    "id": "ID запуска",
                    "run_name": "Алгоритм / название",
                    "segment_name": "Сегмент",
                    "reference_segment_id": "ID референса",
                    "created_at": "Создан",
                }
            )[["ID запуска", "Алгоритм / название", "Сегмент", "ID референса", "Создан"]]
        TableUtils.populate_from_dataframe(self.pre_runs_table, runs_df, index_visible=False)
        self.refresh_feature_views()
        self._update_ui_state()

    def refresh_feature_views(self) -> None:
        if not self.db.is_open():
            TableUtils.populate_from_dataframe(self.features_sets_table, pd.DataFrame(), index_visible=False)
            TableUtils.populate_from_dataframe(self.features_values_table, pd.DataFrame(), index_visible=False)
            TableUtils.populate_from_dataframe(self.pca_variance_table, pd.DataFrame(), index_visible=False)
            TableUtils.populate_from_dataframe(self.pca_scores_table, pd.DataFrame(), index_visible=False)
            TableUtils.populate_from_dataframe(self.pca_loadings_table, pd.DataFrame(), index_visible=False)
            self.pca_plot_widget.clear()
            return
        sets_df_raw = self.db.list_feature_sets()
        filtered = sets_df_raw.copy()
        source_kind = self.pca_source_kind_combo.currentText() if hasattr(self, 'pca_source_kind_combo') else "Все источники"
        if source_kind == "segment":
            filtered = filtered[filtered["source_kind"] == "segment"]
        elif source_kind == "preprocessed_run":
            filtered = filtered[filtered["source_kind"] == "preprocessed_run"]
        gas_filter = self.pca_gas_filter_edit.text().strip().lower() if hasattr(self, 'pca_gas_filter_edit') else ""
        if gas_filter:
            filtered = filtered[filtered["gas_name"].fillna("").astype(str).str.lower().str.contains(gas_filter, na=False)]
        temp_filter = self.pca_temp_filter_edit.text().strip().lower() if hasattr(self, 'pca_temp_filter_edit') else ""
        if temp_filter:
            filtered = filtered[filtered["temperature_c"].fillna("").astype(str).str.lower().str.contains(temp_filter, na=False)]
        light_filter = self.pca_light_filter_combo.currentText() if hasattr(self, 'pca_light_filter_combo') else "Все"
        if light_filter != "Все":
            filtered = filtered[filtered["light_mode"].fillna("") == light_filter]

        display_df = filtered.copy()
        if not display_df.empty:
            display_df = display_df.rename(
                columns={
                    "id": "ID набора",
                    "feature_set_name": "Набор признаков",
                    "segment_name": "Сегмент",
                    "run_name": "Результат предобработки",
                    "profile_name": "Профиль",
                    "source_kind": "Источник",
                    "gas_name": "Газ",
                    "temperature_c": "Температура, °C",
                    "light_mode": "Свет",
                    "created_at": "Создан",
                    "note": "Комментарий",
                }
            )[["ID набора", "Набор признаков", "Сегмент", "Результат предобработки", "Профиль", "Газ", "Температура, °C", "Свет", "Источник", "Создан", "Комментарий"]]
        TableUtils.populate_from_dataframe(self.features_sets_table, display_df, index_visible=False)
        if not display_df.empty and self.features_sets_table.currentRow() < 0:
            self.features_sets_table.selectRow(0)

    def compute_features_for_current_preprocess(self) -> None:
        channels = self.current_preprocessed_run_channels
        source_kind = "preprocessed_run"
        segment_id = getattr(self, "current_preprocessed_segment_id", None)
        run_id = self.current_preprocessed_run_id if isinstance(self.current_preprocessed_run_id, int) and self.current_preprocessed_run_id > 0 else None

        if not channels:
            segment_id = self._selected_preprocess_segment_id()
            if segment_id is None:
                QMessageBox.information(self, "Признаки", "Сначала выполните предобработку или выберите сегмент.")
                return
            try:
                parsed = self.db.load_segment(segment_id)
                channels = parsed.channels
                source_kind = "segment"
                run_id = None
            except Exception as exc:
                QMessageBox.critical(self, "Признаки", f"Не удалось загрузить сегмент для расчёта признаков:\n{exc}")
                return

        wide_df, long_df = compute_channel_features(channels)
        self.current_features_wide_df = wide_df
        self.current_features_long_df = long_df
        self.current_feature_source_kind = source_kind
        self.current_feature_segment_id = segment_id
        self.current_feature_run_id = run_id
        TableUtils.populate_from_dataframe(self.current_features_table, wide_df, index_visible=False)
        self.preprocess_summary_label.setText(
            f"Признаки рассчитаны: {len(wide_df)} канал(ов), источник = {'предобработанный результат' if source_kind == 'preprocessed_run' else 'сегмент'}"
        )
        self._update_ui_state()

    def save_current_feature_set(self) -> None:
        if not self._ensure_project_open():
            return
        if self.current_features_wide_df.empty or self.current_features_long_df.empty:
            QMessageBox.information(self, "Признаки", "Сначала рассчитайте признаки.")
            return
        try:
            payload = self.current_preprocessed_profile_payload or self._build_preprocess_profile_payload()
            algorithm = str(payload.get("algorithm") or payload.get("profile_name") or "R").strip() or "R"
            feature_set_name = f"{algorithm}_{self.current_feature_source_kind}_{self.current_feature_segment_id or 'unknown'}"
            feature_set_id = self.db.save_feature_set(
                feature_set_name=feature_set_name,
                features_long=self.current_features_long_df,
                segment_id=self.current_feature_segment_id,
                run_id=self.current_feature_run_id,
                profile_id=None,
                source_kind=self.current_feature_source_kind,
                note="Рассчитано в Raschet",
            )
            self.refresh_feature_views()
            self._set_status(f"Набор признаков сохранён: ID={feature_set_id}")
        except Exception as exc:
            QMessageBox.critical(self, "Признаки", f"Не удалось сохранить набор признаков:\n{exc}")

    def load_selected_feature_set(self) -> None:
        if not self._ensure_project_open():
            return
        row = self.features_sets_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Признаки", "Выберите набор признаков.")
            return
        item = self.features_sets_table.item(row, 0)
        if item is None:
            return
        try:
            feature_set_id = int(item.text())
            values_df = self.db.load_feature_set_values(feature_set_id)
        except Exception as exc:
            QMessageBox.critical(self, "Признаки", f"Не удалось открыть набор признаков:\n{exc}")
            return
        TableUtils.populate_from_dataframe(self.features_values_table, values_df, index_visible=False)
        self.main_sections.setCurrentWidget(self.analytics_page)

    def run_pca_on_filtered_feature_sets(self) -> None:
        if not self._ensure_project_open():
            return
        display_row_count = self.features_sets_table.rowCount()
        if display_row_count < 2:
            QMessageBox.information(self, "PCA", "Для PCA нужно минимум два набора признаков в текущей выборке.")
            return
        feature_set_ids: List[int] = []
        for row in range(display_row_count):
            item = self.features_sets_table.item(row, 0)
            if item is None:
                continue
            try:
                feature_set_ids.append(int(item.text()))
            except Exception:
                continue
        try:
            meta_df, feature_df = self.db.load_feature_matrix(feature_set_ids)
            if feature_df.empty:
                raise ValueError("Не удалось собрать матрицу признаков для PCA.")
            matrix = feature_df.drop(columns=["id"])
            scale_mode = SCALE_MODES[self.pca_scaling_combo.currentText()]
            result = run_pca(matrix, scale_mode)
        except Exception as exc:
            QMessageBox.critical(self, "PCA", f"Не удалось выполнить PCA:\n{exc}")
            return

        scores_df = pd.concat([meta_df[["id", "feature_set_name", "segment_name", "gas_name", "temperature_c", "light_mode"]].reset_index(drop=True), result["scores"].reset_index(drop=True)], axis=1)
        TableUtils.populate_from_dataframe(self.pca_variance_table, result["variance"], index_visible=False)
        TableUtils.populate_from_dataframe(self.pca_scores_table, scores_df, index_visible=False)
        TableUtils.populate_from_dataframe(self.pca_loadings_table, result["loadings"], index_visible=False)
        self._plot_pca_scores(scores_df)
        self.analytics_tabs.setCurrentIndex(1)
        self.main_sections.setCurrentWidget(self.analytics_page)
        self._set_status("PCA рассчитан по текущей выборке наборов признаков.")

    def _plot_pca_scores(self, scores_df: pd.DataFrame) -> None:
        self.pca_plot_widget.clear()
        plot_item = self.pca_plot_widget.getPlotItem()
        plot_item.setTitle("PCA scores")
        plot_item.setLabel("bottom", "PC1")
        plot_item.setLabel("left", "PC2")
        plot_item.showGrid(x=True, y=True, alpha=0.25)
        if scores_df.empty or "PC1" not in scores_df.columns:
            return
        pc2_col = "PC2" if "PC2" in scores_df.columns else None
        color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        labels = scores_df["gas_name"].fillna("Без метки")
        for idx, label in enumerate(sorted(labels.unique())):
            sub = scores_df[labels == label]
            x = sub["PC1"].to_numpy(dtype=float)
            y = sub[pc2_col].to_numpy(dtype=float) if pc2_col else [0.0] * len(sub)
            scatter = pg.ScatterPlotItem(x=x, y=y, pen=pg.mkPen(color_cycle[idx % len(color_cycle)], width=1.2), brush=pg.mkBrush(color_cycle[idx % len(color_cycle)]), size=8, name=str(label))
            plot_item.addItem(scatter)

    def _selected_preprocess_segment_id(self) -> Optional[int]:
        row = self.pre_segments_table.currentRow()
        if row < 0:
            return None
        item = self.pre_segments_table.item(row, 0)
        if item is None:
            return None
        try:
            return int(item.text())
        except Exception:
            return None

    def apply_preprocess_to_selected_segment(self) -> None:
        if not self._ensure_project_open():
            return
        segment_id = self._selected_preprocess_segment_id()
        if segment_id is None:
            QMessageBox.information(self, "Предобработка", "Выберите сегмент из БД проекта.")
            return
        try:
            parsed = self.db.load_segment(segment_id)
            payload = self._build_preprocess_profile_payload()
            operations = payload.get("operations", []) or []
            if any(op in {"x_div_ref", "ref_div_x"} for op in operations):
                QMessageBox.information(
                    self,
                    "Предобработка",
                    "Ручной выбор референса убран. Операции x/Xref и Xref/x будут подключены позже через автоматический референсный сегмент.",
                )
                return
            processed_channels = apply_preprocess_pipeline(parsed.channels, payload, reference_channels=None)
        except Exception as exc:
            QMessageBox.critical(self, "Предобработка", f"Не удалось выполнить предобработку:\n{exc}")
            return
        self.current_preprocessed_run_channels = processed_channels
        self.current_preprocessed_segment_id = segment_id
        self.current_preprocessed_reference_id = None
        self.current_preprocessed_profile_payload = payload
        self.current_preprocessed_run_id = -1
        self.preprocess_plot_widget.apply_graph_style(self._current_graph_style_payload())
        self.preprocess_plot_widget.set_log_mode(self.yscale_combo.currentText() == "log")
        self.preprocess_plot_widget.redraw(processed_channels, interval=None)
        self.preprocess_summary_label.setText(
            f"Сегмент ID={segment_id} | Операции: {', '.join(payload.get('operations', [])) or 'без предобработки'}"
        )
        self._update_ui_state()

    def save_current_preprocess_run(self) -> None:
        if not self._ensure_project_open():
            return
        channels = getattr(self, 'current_preprocessed_run_channels', None)
        segment_id = getattr(self, 'current_preprocessed_segment_id', None)
        payload = getattr(self, 'current_preprocessed_profile_payload', None)
        if not channels or segment_id is None or payload is None:
            QMessageBox.information(self, "Предобработка", "Сначала выполните предобработку выбранного сегмента.")
            return
        try:
            algorithm = str(payload.get("algorithm") or payload.get("profile_name") or "R").strip() or "R"
            run_name = f"{algorithm}_segment_{segment_id}"
            run_id = self.db.save_preprocessed_run(
                segment_id=segment_id,
                run_name=run_name,
                profile_json=profile_to_json(payload),
                channels=channels,
                profile_id=None,
                reference_segment_id=None,
            )
            self.current_preprocessed_run_id = run_id
            self.current_preprocessed_reference_id = None
            self.refresh_preprocessing_views()
            self._set_status(f"Результат предобработки сохранён: ID={run_id}")
        except Exception as exc:
            QMessageBox.critical(self, "Предобработка", f"Не удалось сохранить результат:\n{exc}")

    def load_selected_preprocessed_run(self) -> None:
        if not self._ensure_project_open():
            return
        row = self.pre_runs_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Предобработка", "Выберите сохранённый результат предобработки.")
            return
        item = self.pre_runs_table.item(row, 0)
        if item is None:
            return
        try:
            run_id = int(item.text())
            parsed = self.db.load_preprocessed_run(run_id)
        except Exception as exc:
            QMessageBox.critical(self, "Предобработка", f"Не удалось открыть результат:\n{exc}")
            return
        self.current_preprocessed_run_id = run_id
        self.current_preprocessed_run_channels = parsed.channels
        self.current_preprocessed_segment_id = None
        self.current_preprocessed_reference_id = None
        self.current_preprocessed_profile_payload = None
        self.current_segment_id = None
        self.current_source_file_id = None
        self.parsed = parsed
        self.current_interval = None
        self.start_edit.clear()
        self.end_edit.clear()
        self.xline_edit.clear()
        self.plot_widget.clear_region()
        self.plot_widget.clear_xline()
        self.channel_legend.set_channels(parsed.channels, saved_names=[])
        self._sync_file_info()
        self._clear_tables_for_new_loaded_data()
        self._refresh_plot()
        self.main_sections.setCurrentWidget(self.measurement_page)
        self._set_status(f"Открыт результат предобработки: {parsed.file_name}")
        self._update_ui_state()

    def open_graph_style_dialog(self) -> None:
        dialog = GraphStyleDialog(
            current_style=self._current_graph_style_payload(),
            x_title=self.x_axis_name_edit.text().strip() or "Time",
            y_title=self.y_axis_name_edit.text().strip() or "Resistance",
            parent=self,
        )
        dialog.styleApplied.connect(self._apply_graph_style_from_dialog)
        dialog.exec()

    def open_segment_details_dialog(self) -> None:
        interval = self._get_interval_values()
        if interval is not None and not self.segment_form_data.get("segment_name"):
            self.segment_form_data["segment_name"] = self._default_segment_name(interval)
        dialog = SegmentDetailsDialog(self.segment_form_data, parent=self)
        if dialog.exec():
            self.segment_form_data.update(dialog.get_data())
            if interval is not None and not self.segment_form_data.get("segment_name"):
                self.segment_form_data["segment_name"] = self._default_segment_name(interval)
            self._update_segment_summary()
            self._update_ui_state()

    def clear_segment_form(self) -> None:
        self.segment_form_data = {
            "segment_name": "",
            "gas_name": "",
            "concentration_ppm": "",
            "temperature_c": "",
            "light_mode": "",
            "is_reference": "0",
            "comment": "",
        }
        interval = self._get_interval_values()
        if interval is not None:
            self.segment_form_data["segment_name"] = self._default_segment_name(interval)
        self._update_segment_summary()

    # ------------------------------------------------------------- actions
    def create_project(self) -> None:
        last_dir = self.config.get("last_dir", str(Path.home()))
        file_path, _ = QFileDialog.getSaveFileName(self, "Создать проект SQLite", str(Path(last_dir) / "raschet_project.sqlite"), "SQLite project (*.sqlite)")
        if not file_path:
            return
        path = Path(file_path)
        if path.suffix.lower() != ".sqlite":
            path = path.with_suffix(".sqlite")
        try:
            self.db.create(str(path))
        except Exception as exc:
            QMessageBox.critical(self, "Проект", f"Не удалось создать проект:\n{exc}")
            return
        self.config.set("last_dir", str(path.parent))
        self.project_path = path
        self.refresh_project_views()
        self._set_status(f"Создан проект: {path.name}")
        self._update_ui_state()
        QMessageBox.information(self, "Проект", f"Проект создан:\n{path}")

    def open_project(self) -> None:
        last_dir = self.config.get("last_dir", str(Path.home()))
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть проект SQLite", last_dir, "SQLite project (*.sqlite)")
        if not file_path:
            return
        try:
            self.db.open(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Проект", f"Не удалось открыть проект:\n{exc}")
            return
        self.config.set("last_dir", str(Path(file_path).parent))
        self.project_path = Path(file_path)
        self.refresh_project_views()
        self._set_status(f"Открыт проект: {Path(file_path).name}")
        self._update_ui_state()
        QMessageBox.information(self, "Проект", f"Проект открыт:\n{file_path}")

    def export_project_copy(self) -> None:
        if not self._ensure_project_open():
            return
        last_dir = self.config.get("last_dir", str(Path.home()))
        default_name = self.project_path.name if self.project_path else "raschet_project.sqlite"
        file_path, _ = QFileDialog.getSaveFileName(self, "Экспорт проекта SQLite", str(Path(last_dir) / default_name), "SQLite project (*.sqlite)")
        if not file_path:
            return
        try:
            self.db.export_project_copy(file_path)
            QMessageBox.information(self, "Экспорт проекта", f"Копия проекта сохранена:\n{file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Экспорт проекта", f"Не удалось экспортировать проект:\n{exc}")

    def export_database_tables(self) -> None:
        if not self._ensure_project_open():
            return
        last_dir = self.config.get("last_dir", str(Path.home()))
        stem = self.project_path.stem if self.project_path else "raschet_project"
        file_path, _ = QFileDialog.getSaveFileName(self, "Экспорт таблиц БД", str(Path(last_dir) / f"{stem}_db_tables.xlsx"), "Excel (*.xlsx);;CSV package (*.csv)")
        if not file_path:
            return
        try:
            exported_to = self.db.export_tables(file_path)
            QMessageBox.information(self, "Экспорт таблиц БД", f"Таблицы экспортированы:\n{exported_to}")
        except Exception as exc:
            QMessageBox.critical(self, "Экспорт таблиц БД", f"Не удалось экспортировать таблицы:\n{exc}")

    def import_database_tables_xlsx(self) -> None:
        if not self._ensure_project_open():
            return
        last_dir = self.config.get("last_dir", str(Path.home()))
        file_path, _ = QFileDialog.getOpenFileName(self, "Импорт таблиц из XLSX", last_dir, "Excel (*.xlsx)")
        if not file_path:
            return
        try:
            self.db.import_tables(file_path)
            self.refresh_project_views()
            QMessageBox.information(self, "Импорт таблиц", f"Таблицы успешно импортированы из:\n{file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Импорт таблиц", f"Не удалось импортировать таблицы:\n{exc}")

    def import_database_tables_csv(self) -> None:
        if not self._ensure_project_open():
            return
        last_dir = self.config.get("last_dir", str(Path.home()))
        folder = QFileDialog.getExistingDirectory(self, "Импорт таблиц из папки CSV", last_dir)
        if not folder:
            return
        try:
            self.db.import_tables(folder)
            self.refresh_project_views()
            QMessageBox.information(self, "Импорт таблиц", f"Таблицы успешно импортированы из:\n{folder}")
        except Exception as exc:
            QMessageBox.critical(self, "Импорт таблиц", f"Не удалось импортировать таблицы:\n{exc}")

    def load_txt_file(self) -> None:
        if not self._ensure_project_open():
            return
        last_dir = self.config.get("last_dir", str(Path.home()))
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть TXT", last_dir, "Text files (*.txt)")
        if not file_path:
            return
        try:
            parsed = SmartTxtParser.parse(file_path)
            source_file_id = self.db.import_source_file(parsed)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка чтения файла", str(exc))
            return

        self.current_source_file_id = source_file_id
        self.current_segment_id = None
        self.current_preprocessed_run_id = None
        self.current_preprocessed_segment_id = None
        self.current_preprocessed_reference_id = None
        self.current_preprocessed_profile_payload = None
        self.current_preprocessed_run_channels = []
        self.parsed = parsed
        self.current_interval = None
        self.results_df = pd.DataFrame()
        self.xline_df = pd.DataFrame()
        self.full_slice_df = pd.DataFrame()
        self.slice_preview_df = pd.DataFrame()
        self.start_edit.clear()
        self.end_edit.clear()
        self.xline_edit.clear()
        self.plot_widget.clear_region()
        self.plot_widget.clear_xline()
        saved_names = self.config.get("channel_names", [])
        self.channel_legend.set_channels(parsed.channels, saved_names=saved_names)
        self._sync_file_info()
        self._update_tables()
        self._refresh_plot()
        self.refresh_project_views()
        self.config.set("last_dir", str(Path(file_path).parent))
        self.segment_form_data["segment_name"] = self._default_segment_name((parsed.channels[0].time[0], parsed.channels[0].time[-1]))
        self._update_segment_summary()
        self._set_status(f"Файл загружен в проект: {parsed.file_name}")
        self.main_sections.setCurrentWidget(self.measurement_page)
        self._update_ui_state()
        QMessageBox.information(self, "Файл загружен", f"Загружено каналов: {len(parsed.channels)}\nДанные импортированы в проект SQLite.")

    def load_selected_source_file(self) -> None:
        if not self._ensure_project_open():
            return
        row = self.project_files_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Файлы проекта", "Выберите строку с импортированным файлом.")
            return
        try:
            source_id_item = self.project_files_table.item(row, 0)
            if source_id_item is None:
                raise ValueError("Не удалось прочитать ID файла.")
            source_file_id = int(source_id_item.text())
            parsed = self.db.load_source_file(source_file_id)
        except Exception as exc:
            QMessageBox.critical(self, "Файлы проекта", f"Не удалось загрузить файл из БД:\n{exc}")
            return
        self._apply_parsed_to_measurement_view(parsed, source_file_id, None, f"Из проекта открыт файл: {parsed.file_name}")

    def _apply_parsed_to_measurement_view(self, parsed: ParsedFile, source_file_id: Optional[int], segment_id: Optional[int], status_message: str) -> None:
        self.current_source_file_id = source_file_id
        self.current_segment_id = segment_id
        self.current_preprocessed_run_id = None
        self.current_preprocessed_segment_id = None
        self.current_preprocessed_reference_id = None
        self.current_preprocessed_profile_payload = None
        self.current_preprocessed_run_channels = []
        self.parsed = parsed
        self.current_interval = None
        self.clear_segment_form()
        if parsed.channels:
            self.segment_form_data["segment_name"] = self._default_segment_name((parsed.channels[0].time[0], parsed.channels[0].time[-1]))
        self._update_segment_summary()
        self.start_edit.clear()
        self.end_edit.clear()
        self.xline_edit.clear()
        self.plot_widget.clear_region()
        self.plot_widget.clear_xline()
        self.channel_legend.set_channels(parsed.channels, saved_names=[])
        self._sync_file_info()
        self._clear_tables_for_new_loaded_data()
        self._refresh_plot()
        self.main_sections.setCurrentWidget(self.measurement_page)
        self._set_status(status_message)
        self._update_ui_state()

    def open_current_source_from_measurement_view(self) -> None:
        if not self._ensure_project_open() or self.current_source_file_id is None:
            QMessageBox.information(self, "Источник", "Для текущего отображения не найден исходный файл в проекте.")
            return
        try:
            parsed = self.db.load_source_file(self.current_source_file_id)
        except Exception as exc:
            QMessageBox.critical(self, "Источник", f"Не удалось открыть исходный файл:\n{exc}")
            return
        self._apply_parsed_to_measurement_view(parsed, self.current_source_file_id, None, f"Открыт исходный файл: {parsed.file_name}")

    def load_selected_segment(self) -> None:
        if not self._ensure_project_open():
            return
        row = self.segments_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Сегменты", "Выберите строку с сегментом.")
            return
        try:
            segment_id_item = self.segments_table.item(row, 0)
            if segment_id_item is None:
                raise ValueError("Не удалось прочитать ID сегмента.")
            segment_id = int(segment_id_item.text())
            parsed = self.db.load_segment(segment_id)
        except Exception as exc:
            QMessageBox.critical(self, "Сегменты", f"Не удалось открыть сегмент:\n{exc}")
            return
        self._apply_parsed_to_measurement_view(parsed, int(parsed.metadata.get("source_file_id", 0)) or None, segment_id, f"Открыт сегмент: {parsed.file_name}")
        self.segment_form_data.update(
            {
                "segment_name": parsed.file_name,
                "gas_name": parsed.metadata.get("gas", ""),
                "concentration_ppm": parsed.metadata.get("concentration_ppm", ""),
                "temperature_c": parsed.metadata.get("temperature_c", ""),
                "light_mode": parsed.metadata.get("light_mode", ""),
                "is_reference": parsed.metadata.get("is_reference", "0"),
                "comment": parsed.metadata.get("comment", ""),
            }
        )
        self._update_segment_summary()

    def delete_selected_segment(self) -> None:
        if not self._ensure_project_open():
            return
        row = self.segments_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Сегменты", "Выберите сегмент для удаления.")
            return
        segment_id_item = self.segments_table.item(row, 0)
        if segment_id_item is None:
            return
        segment_id = int(segment_id_item.text())
        confirm = QMessageBox.question(self, "Удаление сегмента", f"Удалить сегмент ID={segment_id}?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self.db.delete_segment(segment_id)
            self.refresh_project_views()
            self._set_status(f"Удалён сегмент ID={segment_id}")
        except Exception as exc:
            QMessageBox.critical(self, "Сегменты", f"Не удалось удалить сегмент:\n{exc}")

    def save_current_segment_to_db(self) -> None:
        if not self._ensure_project_open():
            return
        if not self.parsed:
            QMessageBox.warning(self, "Сегмент", "Сначала загрузите файл или сегмент на график.")
            return
        interval = self._get_interval_values()
        if not interval:
            QMessageBox.warning(self, "Сегмент", "Сначала задайте корректный интервал сегмента.")
            return
        channels = self._collect_current_segment_channels(interval)
        if not channels:
            QMessageBox.warning(self, "Сегмент", "В выделенном интервале нет данных по выбранным каналам.")
            return

        segment_name = self.segment_form_data.get("segment_name", "").strip() or self._default_segment_name(interval)
        labels = {
            "gas_name": self.segment_form_data.get("gas_name", ""),
            "concentration_ppm": self.segment_form_data.get("concentration_ppm", ""),
            "temperature_c": self.segment_form_data.get("temperature_c", ""),
            "humidity_pct": "",
            "light_mode": self.segment_form_data.get("light_mode", ""),
            "sample_group": "",
            "class_label": "",
            "comment": self.segment_form_data.get("comment", ""),
            "is_reference": self.segment_form_data.get("is_reference", "0"),
        }
        try:
            segment_id = self.db.save_segment(
                self.current_source_file_id,
                self.parsed.file_name,
                segment_name,
                interval,
                labels,
                channels,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Сегмент", f"Не удалось сохранить сегмент в БД:\n{exc}")
            return
        self.current_segment_id = segment_id
        self.refresh_project_views()
        self._set_status(f"Сегмент сохранён в БД: ID={segment_id}")
        QMessageBox.information(self, "Сегмент", f"Сегмент сохранён в БД проекта.\nID: {segment_id}")

    def _update_tables(self) -> None:
        TableUtils.populate_from_dataframe(self.results_table, self.results_df, index_visible=True)
        TableUtils.populate_from_dataframe(self.xline_table, self.xline_df, index_visible=False)
        TableUtils.populate_from_dataframe(self.slice_table, self.slice_preview_df, index_visible=False)

    def _clear_tables_for_new_loaded_data(self) -> None:
        self.results_df = pd.DataFrame()
        self.xline_df = pd.DataFrame()
        self.full_slice_df = pd.DataFrame()
        self.slice_preview_df = pd.DataFrame()
        self.current_xline_label.setText("Rᵢ: —")
        self._update_tables()

    def plot_all_graphs(self) -> None:
        if not self.parsed:
            return
        self.current_interval = None
        self.plot_widget.clear_region()
        self._refresh_plot()
        self._set_status("Построены графики всех видимых каналов.")

    def plot_selected_interval(self) -> None:
        if not self.parsed:
            return
        interval = self._get_interval_values()
        if not interval:
            QMessageBox.warning(self, "Участок", "Введите корректные границы интервала.")
            return
        self.current_interval = interval
        self.plot_widget.set_region(*interval, emit_signal=False)
        self._refresh_plot()
        self._update_slice_preview()
        if not self.segment_form_data.get("segment_name"):
            self.segment_form_data["segment_name"] = self._default_segment_name(interval)
            self._update_segment_summary()
        self.tabs.setCurrentIndex(2)
        self._set_status("Построен выделенный участок.")
        self._update_ui_state()

    def reset_interval(self) -> None:
        self.start_edit.clear()
        self.end_edit.clear()
        self.current_interval = None
        self.plot_widget.clear_region()
        self.full_slice_df = pd.DataFrame()
        self.slice_preview_df = pd.DataFrame()
        TableUtils.populate_from_dataframe(self.slice_table, self.slice_preview_df, index_visible=False)
        self._refresh_plot()
        self._set_status("Выделенный участок сброшен.")

    def calculate_results_action(self) -> None:
        if not self.parsed:
            QMessageBox.warning(self, "Расчёт", "Сначала загрузите файл.")
            return
        interval = self._get_interval_values()
        if not interval:
            QMessageBox.warning(self, "Расчёт", "Сначала задайте корректный интервал.")
            return
        channels = self._visible_channels()
        if not channels:
            QMessageBox.warning(self, "Расчёт", "Не выбрано ни одного канала.")
            return
        self.results_df = calculate_results(channels, interval, self.variant_combo.currentText(), self.avg_points_spin.value())
        TableUtils.populate_from_dataframe(self.results_table, self.results_df, index_visible=True)
        self.tabs.setCurrentIndex(0)
        self._update_ui_state()
        self._set_status("Расчёт выполнен.")

    def build_xline(self) -> None:
        if not self.parsed:
            QMessageBox.warning(self, "Точка на графике", "Сначала загрузите файл.")
            return
        text = self.xline_edit.text().strip().replace(",", ".")
        if not text:
            QMessageBox.warning(self, "Точка на графике", "Введите время tᵢ.")
            return
        try:
            x_value = float(text)
        except ValueError:
            QMessageBox.warning(self, "Точка на графике", "Некорректное значение времени.")
            return
        self.plot_widget.set_xline(x_value, emit_signal=True)
        self.tabs.setCurrentIndex(1)
        self._set_status("Построена bar-line; её можно перетаскивать мышью.")
        self._update_ui_state()

    def clear_xline(self) -> None:
        self.plot_widget.clear_xline()
        self.xline_df = pd.DataFrame()
        TableUtils.populate_from_dataframe(self.xline_table, self.xline_df, index_visible=False)
        self.current_xline_label.setText("Rᵢ: —")
        self._update_ui_state()

    def export_results(self) -> None:
        stem = self._current_file_stem("results")
        self._export_dataframe_dialog(self.results_df, "Экспорт результатов", f"{stem}_results.txt", index=True)

    def export_xline_table(self) -> None:
        stem = self._current_file_stem("ri_table")
        self._export_dataframe_dialog(self.xline_df, "Экспорт Rᵢ", f"{stem}_ri_table.txt", index=False)

    def export_selected_slice(self) -> None:
        self._update_slice_preview()
        stem = self._current_file_stem("selected_slice")
        self._export_dataframe_dialog(self.full_slice_df, "Экспорт участка", f"{stem}_selected_slice.txt", index=False)

    def export_plot_dialog(self) -> None:
        if not self.parsed:
            QMessageBox.information(self, "Сохранить график", "Сначала загрузите файл.")
            return
        last_dir = self.config.get("last_dir", str(Path.home()))
        stem = self._current_file_stem("plot")
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить график", str(Path(last_dir) / f"{stem}_plot.png"), EXPORT_PLOT_FILTER)
        if not file_path:
            return
        path = Path(file_path)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            path = path.with_suffix(".png")
        self.config.set("last_dir", str(path.parent))

        export_style = self._export_style_for_current_preset()
        try:
            export_plot(
                self.plot_widget.plot_widget,
                self._visible_channels(),
                str(path),
                scale_factor=float(export_style.get("export_scale", 2.0)),
                legend_position=str(export_style.get("legend_position", "Справа")),
                legend_font_size=int(export_style.get("legend_font_size", 11)),
                line_width=float(export_style.get("line_width", 1.5)),
                monochrome=bool(export_style.get("monochrome", False)),
                background_mode=str(export_style.get("background_mode", "Белый")),
                x_axis_name=str(export_style.get("x_axis_name", self.x_axis_name_edit.text().strip() or "Time")),
                y_axis_name=str(export_style.get("y_axis_name", self.y_axis_name_edit.text().strip() or "Resistance")),
                font_size=int(export_style.get("font_size", 10)),
                point_mode=bool(export_style.get("point_mode", False)),
                point_size=float(export_style.get("point_size", 4.0)),
                interval=self.current_interval,
                x_log=bool(getattr(self.plot_widget.bottom_axis, "logMode", False)),
                y_log=(self.yscale_combo.currentText() == "log"),
                show_grid=bool(export_style.get("show_grid", True)),
            )
            QMessageBox.information(
                self,
                "Сохранить график",
                f"График сохранён:\n{path}\n\nШаблон: {export_style.get('export_preset', 'Обычный PNG')}\n"
                f"Толщина линий: {export_style.get('line_width', 1.5)}\n"
                f"Размер точек: {export_style.get('point_size', 4.0)} px\n"
                f"Шрифт легенды: {export_style.get('legend_font_size', 11)} pt",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Сохранить график", f"Не удалось сохранить график:\n{exc}")

    def copy_table(self, table: QTableWidget) -> None:
        if TableUtils.copy_table_to_clipboard(self, table):
            self._set_status("Таблица скопирована в буфер обмена.")

    # ------------------------------------------------------------- close
    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_config_from_ui()
        self.db.close()
        super().closeEvent(event)
