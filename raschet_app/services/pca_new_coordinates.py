from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass
class NewCoordinateSystemResult:
    basis: pd.DataFrame
    selected_components: list[str]


def build_new_coordinate_system(eigenvectors_df: pd.DataFrame, n_components: int) -> NewCoordinateSystemResult:
    """Сформировать новую систему координат PCA.

    Вход:
    - строки eigenvectors_df = исходные признаки;
    - столбцы eigenvectors_df = отсортированные компоненты PC1, PC2, ...;
    - n_components = сколько первых компонент оставить по правилу выбора.

    Выход:
    - базис новой системы координат: первые N собственных векторов.
    """
    if eigenvectors_df is None or eigenvectors_df.empty:
        raise ValueError("Нет собственных векторов для формирования новой системы координат PCA.")
    if n_components <= 0:
        raise ValueError("Число выбранных компонент PCA должно быть больше нуля.")

    max_components = eigenvectors_df.shape[1]
    keep = min(int(n_components), max_components)
    selected_components = list(eigenvectors_df.columns[:keep])
    basis = eigenvectors_df.loc[:, selected_components].copy()
    basis.index.name = eigenvectors_df.index.name or "Признак"
    return NewCoordinateSystemResult(basis=basis, selected_components=selected_components)


def new_coordinate_system_to_table(result: NewCoordinateSystemResult) -> pd.DataFrame:
    if result.basis is None or result.basis.empty:
        return pd.DataFrame()
    return result.basis.reset_index()
