from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CovarianceMatrixResult:
    matrix: pd.DataFrame
    mean: pd.Series
    centered_values: np.ndarray


def compute_covariance_matrix(feature_df: pd.DataFrame) -> CovarianceMatrixResult:
    """Вычислить ковариационную матрицу для PCA.

    Ожидаемый формат входа:
    - строки = объекты/наборы признаков;
    - столбцы = подготовленные числовые признаки.

    Ковариация считается после центрирования каждого признака:
        C = X_centered.T @ X_centered / (n_samples - 1)
    """
    if feature_df is None or feature_df.empty:
        raise ValueError("Нет подготовленных данных для расчёта ковариационной матрицы.")
    if len(feature_df) < 2:
        raise ValueError("Для ковариационной матрицы нужно минимум два набора признаков.")

    numeric = feature_df.astype(float)
    mean = numeric.mean(axis=0)
    centered = numeric - mean
    values = centered.to_numpy(dtype=float)
    covariance_values = (values.T @ values) / float(len(numeric) - 1)
    covariance_df = pd.DataFrame(covariance_values, index=numeric.columns, columns=numeric.columns)
    covariance_df.index.name = "Признак"
    return CovarianceMatrixResult(matrix=covariance_df, mean=mean, centered_values=values)


def covariance_matrix_to_table(covariance_df: pd.DataFrame) -> pd.DataFrame:
    if covariance_df is None or covariance_df.empty:
        return pd.DataFrame()
    return covariance_df.reset_index()
