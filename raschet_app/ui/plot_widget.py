from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QVBoxLayout, QWidget

from raschet_app.constants import DEFAULT_GRAPH_STYLE
from raschet_app.models import ChannelData


MONO_PEN_STYLES = [
    Qt.PenStyle.SolidLine,
    Qt.PenStyle.DashLine,
    Qt.PenStyle.DotLine,
    Qt.PenStyle.DashDotLine,
    Qt.PenStyle.DashDotDotLine,
]
MONO_COLORS = ["#111111", "#444444", "#666666", "#888888", "#aaaaaa"]
SUPERSCRIPT_MAP = str.maketrans({
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "-": "⁻",
})


def exponent_to_superscript(power: int) -> str:
    return str(power).translate(SUPERSCRIPT_MAP)


class ScaledAxisItem(pg.AxisItem):
    def __init__(self, orientation: str, axis_title: str, axis_unit: str, scale_divisor: float = 1.0) -> None:
        super().__init__(orientation=orientation)
        self.axis_title = axis_title
        self.axis_unit = axis_unit
        self.scale_divisor = float(scale_divisor) if scale_divisor else 1.0
        self.scale_power = 0
        self.font_size = 10
        self.text_color = "#000000"
        self.setTextPen(self.text_color)
        self.setPen(self.text_color)
        if hasattr(self, "enableAutoSIPrefix"):
            try:
                self.enableAutoSIPrefix(False)
            except Exception:
                pass
        if hasattr(self, "autoSIPrefix"):
            try:
                self.autoSIPrefix = False
            except Exception:
                pass
        self._refresh_label()

    def set_axis_title(self, title: str) -> None:
        self.axis_title = title
        self._refresh_label()

    def set_axis_unit(self, unit: str) -> None:
        self.axis_unit = unit
        self._refresh_label()

    def set_scale_divisor(self, divisor: float, power: int = 0) -> None:
        self.scale_divisor = float(divisor) if divisor else 1.0
        self.scale_power = int(power)
        self._refresh_label()

    def set_font_size(self, size: int) -> None:
        self.font_size = int(size)
        self.setStyle(tickFont=QFont("Arial", self.font_size))
        self._refresh_label()

    def set_text_color(self, color: str) -> None:
        self.text_color = color
        self.setTextPen(color)
        self.setPen(color)
        self._refresh_label()

    def _refresh_label(self) -> None:
        self.setLabel(
            self.axis_title,
            units=self.axis_unit,
            color=self.text_color,
            **{"font-size": f"{self.font_size}pt"},
        )

    def tickStrings(self, values, scale, spacing):  # pragma: no cover - GUI formatting
        strings = []
        divisor = self.scale_divisor if self.scale_divisor else 1.0
        power_suffix = exponent_to_superscript(self.scale_power) if self.scale_power else ""
        is_log_axis = bool(getattr(self, "logMode", False))

        for value in values:
            actual_value = (10 ** value) if is_log_axis else value
            scaled = actual_value / divisor
            if not math.isfinite(scaled) or scaled == 0:
                strings.append("" if not math.isfinite(scaled) else "0")
                continue

            if is_log_axis:
                exponent = int(math.floor(math.log10(abs(actual_value))))
                mantissa = actual_value / (10 ** exponent)
                if abs(mantissa - round(mantissa)) < 1e-8:
                    mantissa_text = str(int(round(mantissa)))
                else:
                    mantissa_text = f"{mantissa:.2f}".rstrip("0").rstrip(".")
                strings.append(f"{mantissa_text}×10{exponent_to_superscript(exponent)}")
                continue

            if abs(scaled) >= 1000:
                base = f"{scaled:.0f}"
            elif abs(scaled) >= 100:
                base = f"{scaled:.1f}"
            elif abs(scaled) >= 10:
                base = f"{scaled:.2f}"
            elif abs(scaled) >= 1:
                base = f"{scaled:.2f}"
            elif abs(scaled) >= 0.1:
                base = f"{scaled:.3f}"
            else:
                base = f"{scaled:.2e}"

            if self.scale_power:
                strings.append(f"{base}×10{power_suffix}")
            else:
                strings.append(base)
        return strings


class SelectionViewBox(pg.ViewBox):
    intervalSelecting = Signal(float, float)
    intervalSelected = Signal(float, float)
    mouseCoordinateChanged = Signal(float, float)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.selection_mode = False
        self._drag_start_x: Optional[float] = None
        self.setMouseMode(self.PanMode)
        self.setMenuEnabled(True)

    def set_selection_mode(self, enabled: bool) -> None:
        self.selection_mode = enabled

    def mouseDragEvent(self, ev, axis=None) -> None:  # pragma: no cover - GUI event
        if self.selection_mode and ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            current = self.mapSceneToView(ev.scenePos()).x()
            if ev.isStart():
                self._drag_start_x = current
                self.intervalSelecting.emit(current, current)
                return
            if self._drag_start_x is None:
                self._drag_start_x = current
            self.intervalSelecting.emit(min(self._drag_start_x, current), max(self._drag_start_x, current))
            if ev.isFinish():
                self.intervalSelected.emit(min(self._drag_start_x, current), max(self._drag_start_x, current))
                self._drag_start_x = None
            return
        super().mouseDragEvent(ev, axis=axis)

    def mouseMoveEvent(self, ev) -> None:  # pragma: no cover - GUI event
        point = self.mapSceneToView(ev.scenePos())
        self.mouseCoordinateChanged.emit(point.x(), point.y())
        super().mouseMoveEvent(ev)


class FastPlotWidget(QWidget):
    intervalChanged = Signal(float, float)
    cursorChanged = Signal(float, float)
    xLineMoved = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view_box = SelectionViewBox()
        self.bottom_axis = ScaledAxisItem("bottom", "Time", "s", scale_divisor=1.0)
        self.left_axis = ScaledAxisItem("left", "Resistance", "Ohm", scale_divisor=1.0)
        self.plot_widget = pg.PlotWidget(
            viewBox=self.view_box,
            axisItems={"bottom": self.bottom_axis, "left": self.left_axis},
        )
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setDownsampling(auto=True, mode="peak")
        self.plot_item.setClipToView(True)
        layout.addWidget(self.plot_widget)

        self.curves: List[pg.PlotDataItem] = []
        self._xline_label = pg.TextItem(anchor=(0, 1), color="#c0392b")
        self._xline_item: Optional[pg.InfiniteLine] = None
        self._region_item: Optional[pg.LinearRegionItem] = None
        self._region_updating = False
        self._xline_updating = False
        self._log_mode = True
        self._current_channels: List[ChannelData] = []
        self._current_interval: Tuple[float, float] | None = None
        self._x_axis_title = "Time"
        self._y_axis_title = "Resistance"
        self._x_scale_divisor = 1.0
        self._x_scale_power = 0
        self._font_size = int(DEFAULT_GRAPH_STYLE["font_size"])
        self._line_width = float(DEFAULT_GRAPH_STYLE["line_width"])
        self._point_size = float(DEFAULT_GRAPH_STYLE["point_size"])
        self._point_mode = bool(DEFAULT_GRAPH_STYLE["point_mode"])
        self._background_mode = str(DEFAULT_GRAPH_STYLE["background_mode"])
        self._show_grid = bool(DEFAULT_GRAPH_STYLE["show_grid"])
        self._monochrome = bool(DEFAULT_GRAPH_STYLE["monochrome"])
        self._data_x_bounds: Optional[Tuple[float, float]] = None
        self._data_y_bounds: Optional[Tuple[float, float]] = None

        self.view_box.mouseCoordinateChanged.connect(self.cursorChanged.emit)
        self.view_box.intervalSelecting.connect(self._preview_region)
        self.view_box.intervalSelected.connect(self._accept_region)
        self.view_box.sigRangeChanged.connect(self._on_view_range_changed)
        self._apply_visual_theme()

    # --------------------------------------------------------- public API
    def set_selection_mode(self, enabled: bool) -> None:
        self.view_box.set_selection_mode(enabled)

    def set_log_mode(self, enabled: bool) -> None:
        self._log_mode = enabled
        self.plot_item.setLogMode(x=False, y=enabled)

    def set_grid_visible(self, enabled: bool) -> None:
        self._show_grid = enabled
        self.plot_item.showGrid(x=enabled, y=enabled, alpha=0.3 if self._background_mode != "Чёрный" else 0.45)

    def set_axis_titles(self, x_title: str, y_title: str) -> None:
        self._x_axis_title = x_title or "Time"
        self._y_axis_title = y_title or "Resistance"
        self.bottom_axis.set_axis_title(self._x_axis_title)
        self.left_axis.set_axis_title(self._y_axis_title)

    def apply_graph_style(self, style: Dict[str, object]) -> None:
        self._font_size = int(style.get("font_size", self._font_size))
        self._line_width = float(style.get("line_width", self._line_width))
        self._point_size = float(style.get("point_size", self._point_size))
        self._point_mode = bool(style.get("point_mode", self._point_mode))
        self._background_mode = str(style.get("background_mode", self._background_mode))
        self._show_grid = bool(style.get("show_grid", self._show_grid))
        self._monochrome = bool(style.get("monochrome", self._monochrome))
        self.set_axis_titles(str(style.get("x_axis_name", self._x_axis_title)), str(style.get("y_axis_name", self._y_axis_title)))
        self._apply_visual_theme()

    def graph_style_state(self) -> Dict[str, object]:
        return {
            "x_axis_name": self._x_axis_title,
            "y_axis_name": self._y_axis_title,
            "font_size": self._font_size,
            "line_width": self._line_width,
            "point_size": self._point_size,
            "point_mode": self._point_mode,
            "background_mode": self._background_mode,
            "show_grid": self._show_grid,
            "monochrome": self._monochrome,
        }

    def auto_range(self) -> None:
        self.plot_item.enableAutoRange()
        self.plot_item.autoRange()

    def redraw(self, channels: Iterable[ChannelData], interval: Tuple[float, float] | None = None) -> None:
        self._current_channels = list(channels)
        self._current_interval = interval
        self._data_x_bounds = None
        self._data_y_bounds = None
        for curve in self.curves:
            self.plot_item.removeItem(curve)
        self.curves.clear()

        if not self._current_channels:
            self.plot_item.setTitle("", color=self._theme_text_color())
            return

        self.plot_item.setTitle("")

        x_min = None
        x_max = None
        y_min = None
        y_max = None

        for idx, channel in enumerate(self._current_channels):
            x = channel.time
            y = channel.resistance
            if interval is not None:
                mask = (x >= interval[0]) & (x <= interval[1])
                x = x[mask]
                y = y[mask]
            if len(x) == 0:
                continue
            if self._log_mode:
                mask_positive = y > 0
                x = x[mask_positive]
                y = y[mask_positive]
            if len(x) == 0:
                continue
            plot_kwargs = self._make_plot_kwargs(channel, idx)
            curve = self.plot_item.plot(x=x, y=y, name=channel.display_name, **plot_kwargs)
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
            self.curves.append(curve)

            x_for_bounds = x
            y_for_bounds = y
            if bool(getattr(self.bottom_axis, "logMode", False)):
                x_for_bounds = x[x > 0]
            if bool(getattr(self.left_axis, "logMode", False)):
                y_for_bounds = y[y > 0]
            if len(x_for_bounds) == 0 or len(y_for_bounds) == 0:
                continue

            current_x_min = float(math.log10(x_for_bounds.min())) if bool(getattr(self.bottom_axis, "logMode", False)) else float(x_for_bounds.min())
            current_x_max = float(math.log10(x_for_bounds.max())) if bool(getattr(self.bottom_axis, "logMode", False)) else float(x_for_bounds.max())
            current_y_min = float(math.log10(y_for_bounds.min())) if bool(getattr(self.left_axis, "logMode", False)) else float(y_for_bounds.min())
            current_y_max = float(math.log10(y_for_bounds.max())) if bool(getattr(self.left_axis, "logMode", False)) else float(y_for_bounds.max())
            x_min = current_x_min if x_min is None else min(x_min, current_x_min)
            x_max = current_x_max if x_max is None else max(x_max, current_x_max)
            y_min = current_y_min if y_min is None else min(y_min, current_y_min)
            y_max = current_y_max if y_max is None else max(y_max, current_y_max)

        if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
            self._data_x_bounds = (x_min, x_max)
            self._data_y_bounds = (y_min, y_max)
            self._apply_data_limits()
            self._update_axes_for_view_range((x_min, x_max), (y_min, y_max))

        if interval is not None:
            self.set_region(*interval, emit_signal=False)
        elif self._region_item is not None:
            self.plot_item.addItem(self._region_item)

        if self._xline_item is not None:
            self._xline_item.setPen(pg.mkPen("#c0392b", width=max(2.0, self._line_width)))
            self.plot_item.addItem(self._xline_item)
            self.plot_item.addItem(self._xline_label)
            self._refresh_xline_label(self._xline_item.value())

        self.auto_range()

    def clear_region(self) -> None:
        self._current_interval = None
        if self._region_item is not None:
            try:
                self._region_item.sigRegionChanged.disconnect(self._on_region_changed)
                self._region_item.sigRegionChangeFinished.disconnect(self._on_region_changed_finished)
            except Exception:
                pass
            self.plot_item.removeItem(self._region_item)
            self._region_item = None

    def set_region(self, start: float, end: float, emit_signal: bool = True) -> None:
        start, end = sorted((float(start), float(end)))
        if self._region_item is None:
            self._region_item = pg.LinearRegionItem(values=(start, end), movable=True)
            self._region_item.setBrush(pg.mkBrush(46, 204, 113, 50))
            self._region_item.setHoverBrush(pg.mkBrush(46, 204, 113, 70))
            self._region_item.setZValue(-10)
            self._region_item.sigRegionChanged.connect(self._on_region_changed)
            self._region_item.sigRegionChangeFinished.connect(self._on_region_changed_finished)
        else:
            self._region_updating = True
            self._region_item.setRegion((start, end))
            self._region_updating = False
        if self._region_item.scene() is None:
            self.plot_item.addItem(self._region_item)
        if emit_signal:
            self.intervalChanged.emit(start, end)

    def current_region(self) -> Optional[Tuple[float, float]]:
        if self._region_item is None:
            return None
        region = self._region_item.getRegion()
        return float(region[0]), float(region[1])

    def set_xline(self, x_value: float, emit_signal: bool = True) -> None:
        x_value = float(x_value)
        if self._xline_item is None:
            self._xline_item = pg.InfiniteLine(pos=x_value, angle=90, movable=True, pen=pg.mkPen("#c0392b", width=max(2.0, self._line_width)))
            self._xline_item.sigPositionChanged.connect(self._on_xline_moved)
            self._xline_item.setZValue(20)
            self.plot_item.addItem(self._xline_item)
            self.plot_item.addItem(self._xline_label)
        else:
            self._xline_updating = True
            self._xline_item.setValue(x_value)
            self._xline_updating = False
            self._xline_item.setPen(pg.mkPen("#c0392b", width=max(2.0, self._line_width)))
            if self._xline_item.scene() is None:
                self.plot_item.addItem(self._xline_item)
            if self._xline_label.scene() is None:
                self.plot_item.addItem(self._xline_label)
        self._refresh_xline_label(x_value)
        if emit_signal:
            self.xLineMoved.emit(x_value)

    def clear_xline(self) -> None:
        if self._xline_item is not None:
            try:
                self._xline_item.sigPositionChanged.disconnect(self._on_xline_moved)
            except Exception:
                pass
            self.plot_item.removeItem(self._xline_item)
            self._xline_item = None
        if self._xline_label.scene() is not None:
            self.plot_item.removeItem(self._xline_label)

    def current_xline(self) -> Optional[float]:
        return float(self._xline_item.value()) if self._xline_item is not None else None

    # --------------------------------------------------------- theme helpers
    def _apply_visual_theme(self) -> None:
        background = "#ffffff"
        foreground = "#111111"
        if self._background_mode == "Светло-серый":
            background = "#f2f4f7"
            foreground = "#1f2937"
        elif self._background_mode == "Чёрный":
            background = "#101214"
            foreground = "#f5f5f5"

        self.plot_widget.setBackground(background)
        self.bottom_axis.set_text_color(foreground)
        self.left_axis.set_text_color(foreground)
        self.bottom_axis.set_font_size(self._font_size)
        self.left_axis.set_font_size(self._font_size)
        self.bottom_axis.set_axis_title(self._x_axis_title)
        self.left_axis.set_axis_title(self._y_axis_title)
        self._xline_label.setColor("#ff6b6b" if self._background_mode == "Чёрный" else "#c0392b")
        self.set_grid_visible(self._show_grid)

    def _theme_text_color(self) -> str:
        return "#f5f5f5" if self._background_mode == "Чёрный" else "#111111"

    def _series_color_and_style(self, channel: ChannelData, index: int):
        if self._monochrome:
            color = MONO_COLORS[index % len(MONO_COLORS)]
            style = MONO_PEN_STYLES[index % len(MONO_PEN_STYLES)]
            return color, style
        return channel.color, Qt.PenStyle.SolidLine

    def _make_plot_kwargs(self, channel: ChannelData, index: int) -> Dict[str, object]:
        color, style = self._series_color_and_style(channel, index)
        if self._point_mode:
            return {
                "pen": None,
                "symbol": "o",
                "symbolSize": self._point_size,
                "symbolPen": pg.mkPen(color=color, width=max(1.0, self._point_size / 4), style=style),
                "symbolBrush": pg.mkBrush(color),
            }
        return {
            "pen": pg.mkPen(color=color, width=self._line_width, style=style),
        }

    # --------------------------------------------------------- internals
    def _apply_data_limits(self) -> None:
        if self._data_x_bounds is None or self._data_y_bounds is None:
            return

        x_min, x_max = self._data_x_bounds
        y_min, y_max = self._data_y_bounds

        x_span = max(x_max - x_min, 1e-12)
        y_span = max(y_max - y_min, 1e-12)

        x_pad = x_span * 0.08
        y_pad = y_span * 0.12

        self.view_box.setLimits(
            xMin=x_min - x_pad,
            xMax=x_max + x_pad,
            yMin=max(1e-12, y_min - y_pad) if self._log_mode else y_min - y_pad,
            yMax=y_max + y_pad,
            minXRange=max(x_span * 1e-6, 1e-12),
            minYRange=max(y_span * 1e-6, 1e-12),
        )

    def _update_axes_for_view_range(self, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> None:
        max_abs_x = max(abs(float(x_range[0])), abs(float(x_range[1])))
        max_abs_y = max(abs(float(y_range[0])), abs(float(y_range[1])))

        # Для времени в секундах сохраняем обычный вид чисел вплоть до 10^4.
        x_is_log = bool(getattr(self.bottom_axis, "logMode", False))
        y_is_log = bool(getattr(self.left_axis, "logMode", False))

        x_power = 0 if x_is_log else self._choose_power_of_ten(max_abs_x, plain_upper_exponent=4)
        y_power = 0 if y_is_log else self._choose_power_of_ten(max_abs_y, plain_upper_exponent=2)

        self._x_scale_power = x_power
        self._x_scale_divisor = 10.0 ** x_power if x_power else 1.0
        self.bottom_axis.set_scale_divisor(self._x_scale_divisor, x_power)
        self.bottom_axis.set_axis_title(self._x_axis_title)
        self.bottom_axis.set_axis_unit("s")

        y_divisor = 10.0 ** y_power if y_power else 1.0
        self.left_axis.set_scale_divisor(y_divisor, y_power)
        self.left_axis.set_axis_title(self._y_axis_title)
        self.left_axis.set_axis_unit("Ohm")

    @staticmethod
    def _choose_power_of_ten(max_abs_value: float, plain_upper_exponent: int = 2) -> int:
        if not math.isfinite(max_abs_value) or max_abs_value <= 0:
            return 0
        exponent = int(math.floor(math.log10(max_abs_value)))
        if -2 <= exponent <= plain_upper_exponent:
            return 0
        return exponent

    def _preview_region(self, start: float, end: float) -> None:
        self.set_region(start, end, emit_signal=False)

    def _accept_region(self, start: float, end: float) -> None:
        self.set_region(start, end, emit_signal=True)

    def _on_region_changed(self) -> None:
        if self._region_item is None or self._region_updating:
            return
        region = self._region_item.getRegion()
        self.intervalChanged.emit(float(region[0]), float(region[1]))

    def _on_region_changed_finished(self) -> None:
        if self._region_item is None:
            return
        region = self._region_item.getRegion()
        self.intervalChanged.emit(float(region[0]), float(region[1]))

    def _on_xline_moved(self) -> None:
        if self._xline_item is None:
            return
        value = float(self._xline_item.value())
        self._refresh_xline_label(value)
        if not self._xline_updating:
            self.xLineMoved.emit(value)

    def _refresh_xline_label(self, x_value: float) -> None:
        view_range = self.plot_item.viewRange()
        ymax = view_range[1][1]
        label_x = x_value / self._x_scale_divisor if self._x_scale_divisor else x_value
        if self._x_scale_power:
            text = f"tᵢ = {label_x:.4f}×10{exponent_to_superscript(self._x_scale_power)} s"
        else:
            text = f"tᵢ = {label_x:.4f} s"
        self._xline_label.setText(text)
        self._xline_label.setPos(x_value, ymax)

    def _on_view_range_changed(self, _, ranges) -> None:
        try:
            x_range = ranges[0]
            y_range = ranges[1]
        except Exception:
            return
        self._update_axes_for_view_range((float(x_range[0]), float(x_range[1])), (float(y_range[0]), float(y_range[1])))
        if self._xline_item is not None:
            self._refresh_xline_label(float(self._xline_item.value()))
