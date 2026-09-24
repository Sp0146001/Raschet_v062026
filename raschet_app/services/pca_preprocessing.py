from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class PCAPreparationResult:
    matrix: pd.DataFrame
    report: pd.DataFrame


def _safe_column_name(value: object) -> str:
    text = str(value).strip()
    return text if text else "Не указано"


def _prepare_numeric_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    prepared = pd.DataFrame(index=df.index)
    report_rows: list[dict[str, object]] = []

    for column in df.columns:
        source = pd.to_numeric(df[column], errors="coerce")
        missing_before = int(source.isna().sum())
        if source.notna().sum() == 0:
            fill_value = 0.0
            status = "все значения были пропусками, заполнено 0"
        else:
            fill_value = float(source.median())
            status = "заполнено медианой" if missing_before else "без пропусков"
        prepared[column] = source.fillna(fill_value).astype(float)
        report_rows.append(
            {
                "Колонка": column,
                "Тип": "числовая",
                "Пропусков": missing_before,
                "Заполнение": fill_value if missing_before else "—",
                "Статус": status,
            }
        )
    return prepared, report_rows


def _encode_categorical_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    encoded_parts: list[pd.DataFrame] = []
    report_rows: list[dict[str, object]] = []

    for column in df.columns:
        values = df[column].fillna("Не указано").map(_safe_column_name)
        dummies = pd.get_dummies(values, prefix=str(column), dtype=float)
        encoded_parts.append(dummies)
        report_rows.append(
            {
                "Колонка": column,
                "Тип": "категориальная",
                "Пропусков": int(df[column].isna().sum()),
                "Заполнение": "Не указано",
                "Статус": f"one-hot: {len(dummies.columns)} колонок",
            }
        )

    if encoded_parts:
        return pd.concat(encoded_parts, axis=1), report_rows
    return pd.DataFrame(index=df.index), report_rows


def prepare_pca_input(
    feature_df: pd.DataFrame,
    categorical_df: Optional[pd.DataFrame] = None,
    categorical_columns: Optional[Iterable[str]] = None,
) -> PCAPreparationResult:
    """Подготовить матрицу для PCA.

    Что делает этап подготовки:
    - приводит признаки к числовому типу;
    - обрабатывает пропуски в числовых признаках медианой;
    - полностью пустые числовые признаки заполняет нулём;
    - при необходимости кодирует категориальные колонки one-hot.

    Стандартизация пока остаётся в существующем блоке PCA (`run_pca`), чтобы не
    сломать текущий UI с выбором режима масштабирования. На следующих этапах PCA
    этот шаг будет перенесён в единый pipeline подготовки/анализа.
    """
    if feature_df is None or feature_df.empty:
        raise ValueError("Нет признаков для подготовки PCA.")

    work_df = feature_df.copy()
    if "id" in work_df.columns:
        work_df = work_df.drop(columns=["id"])

    if work_df.empty:
        raise ValueError("Матрица признаков для PCA пуста после удаления служебных колонок.")

    numeric_prepared, report_rows = _prepare_numeric_columns(work_df)
    prepared_parts = [numeric_prepared]

    if categorical_df is not None and categorical_columns:
        available_columns = [col for col in categorical_columns if col in categorical_df.columns]
        if available_columns:
            categorical_prepared, categorical_report = _encode_categorical_columns(categorical_df[available_columns])
            categorical_prepared.index = numeric_prepared.index
            prepared_parts.append(categorical_prepared)
            report_rows.extend(categorical_report)

    prepared = pd.concat(prepared_parts, axis=1)
    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    if prepared.isna().any().any():
        prepared = prepared.fillna(0.0)
        report_rows.append(
            {
                "Колонка": "*",
                "Тип": "числовая",
                "Пропусков": "после inf/-inf",
                "Заполнение": 0.0,
                "Статус": "inf/-inf заменены на 0",
            }
        )

    if prepared.empty or prepared.shape[1] == 0:
        raise ValueError("После подготовки не осталось признаков для PCA.")

    return PCAPreparationResult(matrix=prepared.astype(float), report=pd.DataFrame(report_rows))
