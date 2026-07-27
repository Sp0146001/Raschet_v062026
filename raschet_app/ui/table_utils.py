from __future__ import annotations

from typing import List

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QMessageBox, QTableWidget, QTableWidgetItem, QWidget


class TableUtils:
    @staticmethod
    def setup_table(table: QTableWidget) -> None:
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    @staticmethod
    def populate_from_dataframe(table: QTableWidget, df: pd.DataFrame, index_visible: bool = True) -> None:
        table.clear()
        if df is None or df.empty:
            table.setRowCount(0)
            table.setColumnCount(0)
            return

        columns = [str(c) for c in df.columns]
        row_labels = [str(i) for i in df.index] if index_visible else ["" for _ in range(len(df.index))]
        table.setRowCount(len(df.index))
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        if index_visible:
            table.setVerticalHeaderLabels(row_labels)
        else:
            table.verticalHeader().setVisible(False)

        for r in range(len(df.index)):
            for c, col_name in enumerate(df.columns):
                value = df.iloc[r, c]
                item = QTableWidgetItem("" if pd.isna(value) else str(value))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(r, c, item)

        table.resizeRowsToContents()
        table.resizeColumnsToContents()

    @staticmethod
    def table_to_tsv(table: QTableWidget, include_headers: bool = True, selection_only: bool = False) -> str:
        if table.rowCount() == 0 or table.columnCount() == 0:
            return ""

        if selection_only and table.selectedIndexes():
            selected = table.selectedIndexes()
            rows = sorted({index.row() for index in selected})
            cols = sorted({index.column() for index in selected})
        else:
            rows = list(range(table.rowCount()))
            cols = list(range(table.columnCount()))

        lines: List[str] = []
        if include_headers:
            header = [""]
            header.extend(table.horizontalHeaderItem(c).text() if table.horizontalHeaderItem(c) else f"C{c + 1}" for c in cols)
            lines.append("\t".join(header))

        for row in rows:
            parts = []
            if include_headers:
                vheader = table.verticalHeaderItem(row)
                parts.append(vheader.text() if vheader else f"R{row + 1}")
            for col in cols:
                item = table.item(row, col)
                parts.append(item.text() if item else "")
            lines.append("\t".join(parts))
        return "\n".join(lines)

    @staticmethod
    def copy_table_to_clipboard(parent: QWidget, table: QTableWidget, selection_only: bool = False) -> bool:
        text = TableUtils.table_to_tsv(table, include_headers=True, selection_only=selection_only)
        if not text:
            QMessageBox.information(parent, "Буфер обмена", "Нет данных для копирования.")
            return False
        QGuiApplication.clipboard().setText(text)
        return True
