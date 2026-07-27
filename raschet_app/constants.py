from __future__ import annotations

from typing import Dict

APP_NAME = "raschet-qt"
APP_DISPLAY_NAME = "raschet"
APP_VERSION = "2.1.0"

DEFAULT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
]

CALC_MODES: Dict[str, Dict[str, object]] = {
    "Отклик-Сопр.Растёт": {
        "fraction": 0.9,
        "fraction_label": "0.9",
        "s_mode": "grow",
        "time_rows": ("t0", "t(g)"),
    },
    "Отклик-Сопр.Падает": {
        "fraction": 0.9,
        "fraction_label": "0.9",
        "s_mode": "fall",
        "time_rows": ("t0", "t(g)"),
    },
    "Восстановл-Сопр.Падает": {
        "fraction": 0.1,
        "fraction_label": "0.1",
        "s_mode": "fall",
        "time_rows": ("t(g)", "t0"),
    },
    "Восстановл-Сопр.Растёт": {
        "fraction": 0.1,
        "fraction_label": "0.1",
        "s_mode": "grow",
        "time_rows": ("t(g)", "t0"),
    },
}

EXPORT_TABLE_FILTER = "Text (*.txt);;CSV (*.csv);;Excel (*.xlsx)"
EXPORT_PLOT_FILTER = "PNG (*.png);;JPEG (*.jpg *.jpeg)"

LEGEND_POSITION_OPTIONS = ["Справа", "Снизу", "Слева", "Сверху"]
BACKGROUND_OPTIONS = ["Белый", "Светло-серый", "Чёрный"]
EXPORT_PRESET_OPTIONS = [
    "Обычный PNG",
    "PNG высокого качества",
    "Ч/б стиль",
    "Толстые линии",
    "Крупная легенда",
]

DEFAULT_GRAPH_STYLE: Dict[str, object] = {
    "font_size": 10,
    "legend_font_size": 11,
    "legend_position": "Справа",
    "line_width": 1.5,
    "point_size": 4.0,
    "point_mode": False,
    "background_mode": "Белый",
    "show_grid": True,
    "monochrome": False,
    "export_scale": 2.0,
    "export_preset": "Обычный PNG",
}

EXPORT_PRESET_OVERRIDES: Dict[str, Dict[str, object]] = {
    "Обычный PNG": {
        "export_scale": 2.0,
        "monochrome": False,
        "line_width": 1.5,
        "legend_font_size": 11,
        "legend_position": "Справа",
        "background_mode": "Белый",
    },
    "PNG высокого качества": {
        "export_scale": 5.0,
        "monochrome": False,
        "legend_position": "Справа",
        "background_mode": "Белый",
    },
    "Ч/б стиль": {
        "monochrome": True,
        "background_mode": "Белый",
        "line_width": 2.5,
    },
    "Толстые линии": {
        "line_width": 4.5,
    },
    "Крупная легенда": {
        "legend_font_size": 20,
        "legend_position": "Справа",
    },
}
