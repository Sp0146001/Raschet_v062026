from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EigenDecompositionResult:
    eigenvalues: np.ndarray
    eigenvectors: pd.DataFrame


def compute_eigendecomposition(covariance_df: pd.DataFrame) -> EigenDecompositionResult:
    """Найти собственные значения и собственные векторы ковариационной матрицы.

    На этом этапе порядок возвращается таким, как его отдаёт `numpy.linalg.eigh`
    для симметричной матрицы: обычно по возрастанию собственных значений.
    Сортировка по убыванию будет отдельным следующим этапом pipeline.
    """
    if covariance_df is None or covariance_df.empty:
        raise ValueError("Нет ковариационной матрицы для поиска собственных значений и векторов.")
    if covariance_df.shape[0] != covariance_df.shape[1]:
        raise ValueError("Ковариационная матрица должна быть квадратной.")

    values = covariance_df.to_numpy(dtype=float)
    # Защита от микроскопической численной несимметричности после вычислений.
    values = (values + values.T) / 2.0
    eigenvalues, eigenvectors_values = np.linalg.eigh(values)

    component_names = [f"EV{i + 1}" for i in range(len(eigenvalues))]
    eigenvectors = pd.DataFrame(eigenvectors_values, index=covariance_df.index, columns=component_names)
    eigenvectors.index.name = "Признак"
    return EigenDecompositionResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)


def eigenvalues_to_table(result: EigenDecompositionResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Компонента": list(result.eigenvectors.columns),
            "Собственное значение": result.eigenvalues,
        }
    )


def eigenvectors_to_table(result: EigenDecompositionResult) -> pd.DataFrame:
    if result.eigenvectors is None or result.eigenvectors.empty:
        return pd.DataFrame()
    return result.eigenvectors.reset_index()
