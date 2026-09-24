from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class PCAProjectionResult:
    scores: pd.DataFrame


def project_to_new_coordinates(
    centered_values: np.ndarray,
    sample_index: Iterable[object],
    basis: pd.DataFrame,
) -> PCAProjectionResult:
    """Спроецировать подготовленные данные на новую систему координат PCA.

    Вход:
    - centered_values: центрированная матрица объектов, строки = объекты,
      столбцы = признаки;
    - basis: новая база координат, строки = признаки, столбцы = выбранные PC.

    Выход:
    - координаты объектов в пространстве выбранных главных компонент.
    """
    if centered_values is None or centered_values.size == 0:
        raise ValueError("Нет данных для проекции PCA.")
    if basis is None or basis.empty:
        raise ValueError("Нет новой системы координат PCA для проекции.")

    values = np.asarray(centered_values, dtype=float)
    basis_values = basis.to_numpy(dtype=float)
    if values.shape[1] != basis_values.shape[0]:
        raise ValueError(
            "Размерность данных не совпадает с новой системой координат PCA: "
            f"признаков в данных = {values.shape[1]}, признаков в базе = {basis_values.shape[0]}."
        )

    projected = values @ basis_values
    scores = pd.DataFrame(projected, index=list(sample_index), columns=list(basis.columns))
    scores.index.name = "Объект"
    return PCAProjectionResult(scores=scores)


def projection_to_table(result: PCAProjectionResult) -> pd.DataFrame:
    if result.scores is None or result.scores.empty:
        return pd.DataFrame()
    return result.scores.reset_index(drop=True)
