from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, QPoint, QRect, QRectF, QSize
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from raschet_app.models import ChannelData
from raschet_app.ui.plot_widget import ScaledAxisItem

MONO_COLORS = ["#111111", "#444444", "#666666", "#888888", "#aaaaaa"]
MONO_PEN_STYLES = [
    Qt.PenStyle.SolidLine,
    Qt.PenStyle.DashLine,
    Qt.PenStyle.DotLine,
    Qt.PenStyle.DashDotLine,
    Qt.PenStyle.DashDotDotLine,
]


def export_dataframe(df: pd.DataFrame, file_path: str, index: bool = True) -> None:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        df.to_csv(path, sep="\t", index=index, encoding="utf-8-sig")
    elif suffix == ".csv":
        df.to_csv(path, sep=";", index=index, encoding="utf-8-sig")
    elif suffix == ".xlsx":
        df.to_excel(path, index=index)
    else:
        raise ValueError(f"Неподдерживаемый формат: {suffix}")


def export_plot(
    plot_widget: QWidget,
    channels: Iterable[ChannelData],
    file_path: str,
    scale_factor: float = 2.0,
    legend_position: str = "Справа",
    legend_font_size: int = 11,
    line_width: float = 1.5,
    monochrome: bool = False,
    background_mode: str = "Белый",
    x_axis_name: str = "Time",
    y_axis_name: str = "Resistance",
    font_size: int = 10,
    point_mode: bool = False,
    point_size: float = 4.0,
    interval: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = True,
    show_grid: bool = True,
) -> None:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Неподдерживаемый формат изображения: {suffix}")

    visible_channels = list(channels)
    if not visible_channels:
        raise ValueError("Нет видимых каналов для сохранения рисунка.")

    source_size = plot_widget.size()
    if source_size.width() <= 0 or source_size.height() <= 0:
        source_size = QSize(1200, 700)

    bg_color, fg_color = _theme_colors(background_mode)
    export_plot_widget = _build_export_plot_widget(
        size=source_size,
        channels=visible_channels,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
        font_size=font_size,
        line_width=line_width,
        point_mode=point_mode,
        point_size=point_size,
        monochrome=monochrome,
        background_mode=background_mode,
        interval=interval,
        x_log=x_log,
        y_log=y_log,
        show_grid=show_grid,
    )
    QApplication.processEvents()

    margin = 24
    line_sample = 34
    row_height = max(24, legend_font_size + 12)
    legend_title = ""

    legend_font = QFont("Arial", legend_font_size)
    legend_title_font = QFont("Arial", max(legend_font_size + 1, 12))
    legend_title_font.setBold(True)
    fm = QFontMetrics(legend_font)
    fm_title = QFontMetrics(legend_title_font)

    max_text_width = max(fm.horizontalAdvance(ch.display_name) for ch in visible_channels)
    item_width = line_sample + 12 + max_text_width + 10
    title_height = 0

    legend_rect_size = _calculate_legend_rect(
        position=legend_position,
        channel_count=len(visible_channels),
        plot_size=source_size,
        item_width=item_width,
        row_height=row_height,
        title_height=title_height,
        margin=margin,
    )

    total_width, total_height, plot_rect, legend_rect = _compose_layout(
        position=legend_position,
        plot_size=source_size,
        legend_size=legend_rect_size,
        margin=margin,
    )

    image = QImage(int(total_width * scale_factor), int(total_height * scale_factor), QImage.Format.Format_RGB32)
    image.fill(bg_color)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.scale(scale_factor, scale_factor)

    export_plot_widget.render(painter, QRectF(plot_rect), export_plot_widget.rect())
    _draw_legend(
        painter=painter,
        rect=legend_rect,
        channels=visible_channels,
        title=legend_title,
        font=legend_font,
        title_font=legend_title_font,
        line_width=line_width,
        monochrome=monochrome,
        fg_color=fg_color,
        bg_color=bg_color,
        line_sample=line_sample,
        row_height=row_height,
        margin=12,
        point_mode=point_mode,
        point_size=point_size,
    )

    painter.end()
    image.save(str(path))
    export_plot_widget.deleteLater()


def _build_export_plot_widget(
    size: QSize,
    channels: list[ChannelData],
    x_axis_name: str,
    y_axis_name: str,
    font_size: int,
    line_width: float,
    point_mode: bool,
    point_size: float,
    monochrome: bool,
    background_mode: str,
    interval: Optional[Tuple[float, float]],
    x_log: bool,
    y_log: bool,
    show_grid: bool,
) -> pg.PlotWidget:
    bottom_axis = ScaledAxisItem("bottom", x_axis_name, "s", scale_divisor=1.0)
    left_axis = ScaledAxisItem("left", y_axis_name, "Ohm", scale_divisor=1.0)
    plot_widget = pg.PlotWidget(axisItems={"bottom": bottom_axis, "left": left_axis})
    plot_widget.resize(size)
    plot_widget.setFixedSize(size)
    try:
        plot_widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    except Exception:
        pass
    plot_item = plot_widget.getPlotItem()
    plot_item.setTitle("")
    plot_item.setDownsampling(auto=True, mode="peak")
    plot_item.setClipToView(True)
    plot_item.setLogMode(x=x_log, y=y_log)
    bg_color, fg_color = _theme_colors(background_mode)
    plot_widget.setBackground(bg_color)
    bottom_axis.set_text_color(fg_color.name())
    left_axis.set_text_color(fg_color.name())
    bottom_axis.set_font_size(font_size)
    left_axis.set_font_size(font_size)
    plot_item.showGrid(x=show_grid, y=show_grid, alpha=0.3 if background_mode != "Чёрный" else 0.45)

    for idx, channel in enumerate(channels):
        x = channel.time
        y = channel.resistance
        if interval is not None:
            mask = (x >= interval[0]) & (x <= interval[1])
            x = x[mask]
            y = y[mask]
        if len(x) == 0 or len(y) == 0:
            continue
        if x_log:
            mask = x > 0
            x = x[mask]
            y = y[mask]
        if y_log:
            mask = y > 0
            x = x[mask]
            y = y[mask]
        if len(x) == 0 or len(y) == 0:
            continue

        plot_item.plot(x=x, y=y, **_make_export_plot_kwargs(channel, idx, monochrome, line_width, point_mode, point_size))

    plot_item.enableAutoRange()
    plot_item.autoRange()
    plot_widget.show()
    QApplication.processEvents()
    return plot_widget


def _make_export_plot_kwargs(channel: ChannelData, index: int, monochrome: bool, line_width: float, point_mode: bool, point_size: float) -> Dict[str, object]:
    if monochrome:
        color = MONO_COLORS[index % len(MONO_COLORS)]
        style = MONO_PEN_STYLES[index % len(MONO_PEN_STYLES)]
    else:
        color = channel.color
        style = Qt.PenStyle.SolidLine

    if point_mode:
        return {
            "pen": None,
            "symbol": "o",
            "symbolSize": point_size,
            "symbolPen": pg.mkPen(color=color, width=max(1.0, point_size / 4), style=style),
            "symbolBrush": pg.mkBrush(color),
        }

    return {
        "pen": pg.mkPen(color=color, width=line_width, style=style),
    }


def _theme_colors(background_mode: str) -> Tuple[QColor, QColor]:
    if background_mode == "Светло-серый":
        return QColor("#f2f4f7"), QColor("#1f2937")
    if background_mode == "Чёрный":
        return QColor("#101214"), QColor("#f5f5f5")
    return QColor("#ffffff"), QColor("#111111")


def _calculate_legend_rect(
    position: str,
    channel_count: int,
    plot_size: QSize,
    item_width: int,
    row_height: int,
    title_height: int,
    margin: int,
) -> QSize:
    if position in {"Снизу", "Сверху"}:
        usable_width = max(plot_size.width() - 2 * margin, item_width)
        columns = max(1, min(channel_count, usable_width // max(item_width, 1)))
        rows = int(math.ceil(channel_count / columns))
        width = max(plot_size.width(), columns * item_width + 2 * margin)
        height = title_height + rows * row_height + 2 * margin
        return QSize(width, height)

    width = max(230, item_width + 2 * margin)
    height = title_height + channel_count * row_height + 2 * margin
    return QSize(width, height)


def _compose_layout(position: str, plot_size: QSize, legend_size: QSize, margin: int):
    if position == "Слева":
        total_width = legend_size.width() + plot_size.width() + 3 * margin
        total_height = max(plot_size.height(), legend_size.height()) + 2 * margin
        legend_rect = QRect(margin, margin, legend_size.width(), legend_size.height())
        plot_rect = QRect(legend_rect.right() + margin, margin, plot_size.width(), plot_size.height())
        return total_width, total_height, plot_rect, legend_rect

    if position == "Сверху":
        total_width = max(plot_size.width(), legend_size.width()) + 2 * margin
        total_height = legend_size.height() + plot_size.height() + 3 * margin
        legend_rect = QRect(margin, margin, legend_size.width(), legend_size.height())
        plot_rect = QRect(margin, legend_rect.bottom() + margin, plot_size.width(), plot_size.height())
        return total_width, total_height, plot_rect, legend_rect

    if position == "Снизу":
        total_width = max(plot_size.width(), legend_size.width()) + 2 * margin
        total_height = plot_size.height() + legend_size.height() + 3 * margin
        plot_rect = QRect(margin, margin, plot_size.width(), plot_size.height())
        legend_rect = QRect(margin, plot_rect.bottom() + margin, legend_size.width(), legend_size.height())
        return total_width, total_height, plot_rect, legend_rect

    total_width = plot_size.width() + legend_size.width() + 3 * margin
    total_height = max(plot_size.height(), legend_size.height()) + 2 * margin
    plot_rect = QRect(margin, margin, plot_size.width(), plot_size.height())
    legend_rect = QRect(plot_rect.right() + margin, margin, legend_size.width(), legend_size.height())
    return total_width, total_height, plot_rect, legend_rect


def _draw_legend(
    painter: QPainter,
    rect: QRect,
    channels: list[ChannelData],
    title: str,
    font: QFont,
    title_font: QFont,
    line_width: float,
    monochrome: bool,
    fg_color: QColor,
    bg_color: QColor,
    line_sample: int,
    row_height: int,
    margin: int,
    point_mode: bool,
    point_size: float,
) -> None:
    painter.save()
    painter.setBrush(bg_color)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setFont(font)
    max_text_width = max(QFontMetrics(font).horizontalAdvance(ch.display_name) for ch in channels)
    item_width = line_sample + 12 + max_text_width + 10

    if rect.width() > item_width * 2 + 2 * margin:
        columns = max(1, min(len(channels), (rect.width() - 2 * margin) // max(item_width, 1)))
    else:
        columns = 1
    rows = int(math.ceil(len(channels) / columns))

    start_y = rect.y() + margin
    for idx, channel in enumerate(channels):
        col = idx // rows
        row = idx % rows
        x = rect.x() + margin + col * item_width
        y = start_y + row * row_height

        color, pen_style = _legend_pen_style(channel, idx, monochrome)
        if point_mode:
            painter.setPen(QPen(color, max(1.0, point_size / 5), pen_style))
            painter.setBrush(color)
            diameter = max(4.0, point_size)
            painter.drawEllipse(QRectF(x + (line_sample - diameter) / 2, y + (row_height - diameter) / 2, diameter, diameter))
        else:
            painter.setPen(QPen(color, max(2.5, line_width + 0.5), pen_style))
            painter.drawLine(QPoint(x, y + row_height // 2), QPoint(x + line_sample, y + row_height // 2))
        painter.setPen(QPen(fg_color, 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawText(
            QRectF(x + line_sample + 12, y, item_width - line_sample - 12, row_height),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            channel.display_name,
        )

    painter.restore()


def _legend_pen_style(channel: ChannelData, index: int, monochrome: bool):
    if monochrome:
        color = QColor(MONO_COLORS[index % len(MONO_COLORS)])
        style = MONO_PEN_STYLES[index % len(MONO_PEN_STYLES)]
        return color, style
    return QColor(channel.color), Qt.PenStyle.SolidLine
