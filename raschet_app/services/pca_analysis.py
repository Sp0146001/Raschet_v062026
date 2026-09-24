from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from raschet_app.services.pca_component_selection import select_components
from raschet_app.services.pca_covariance import compute_covariance_matrix, covariance_matrix_to_table
from raschet_app.services.pca_eigendecomposition import compute_eigendecomposition, eigenvalues_to_table, eigenvectors_to_table
from raschet_app.services.pca_new_coordinates import build_new_coordinate_system, new_coordinate_system_to_table
from raschet_app.services.pca_projection import project_to_new_coordinates, projection_to_table
from raschet_app.services.pca_sorting import sort_eigen_components_desc


SCALE_MODES = {
    "Без масштабирования": "none",
    "Только центрирование": "center",
    "Стандартизация (z-score)": "zscore",
}


def apply_scaling(df: pd.DataFrame, mode: str) -> Tuple[np.ndarray, pd.Series, pd.Series]:
    numeric = df.astype(float)
    mean = numeric.mean(axis=0)
    std = numeric.std(axis=0, ddof=0).replace(0, 1.0)

    if mode == "zscore":
        scaled = (numeric - mean) / std
    elif mode == "center":
        scaled = numeric - mean
    else:
        scaled = numeric.copy()
    return scaled.to_numpy(dtype=float), mean, std


def run_pca(
    feature_df: pd.DataFrame,
    scale_mode: str,
    variance_threshold: float = 0.95,
) -> Dict[str, pd.DataFrame]:
    if feature_df.empty:
        raise ValueError("Нет данных для PCA.")
    if len(feature_df) < 2:
        raise ValueError("Для PCA нужно как минимум два набора признаков.")

    X, mean, std = apply_scaling(feature_df, scale_mode)
    scaled_df = pd.DataFrame(X, index=feature_df.index, columns=feature_df.columns)
    covariance = compute_covariance_matrix(scaled_df)
    eigendecomposition = compute_eigendecomposition(covariance.matrix)
    sorted_components = sort_eigen_components_desc(eigendecomposition)
    selection = select_components(sorted_components.eigenvalues, variance_threshold)
    new_coordinates = build_new_coordinate_system(sorted_components.eigenvectors, selection.n_components)
    projection = project_to_new_coordinates(covariance.centered_values, feature_df.index, new_coordinates.basis)

    variance_df = selection.table[["Компонента", "Собственное значение", "Доля дисперсии, %", "Накопленная доля, %"]].copy()
    scores_df = projection_to_table(projection)
    loadings_df = new_coordinate_system_to_table(new_coordinates)
    means_df = pd.DataFrame({"Признак": feature_df.columns, "Среднее": mean.values, "Std": std.values})

    return {
        "variance": variance_df,
        "scores": scores_df,
        "loadings": loadings_df,
        "means": means_df,
        "covariance": covariance_matrix_to_table(covariance.matrix),
        "eigenvalues": eigenvalues_to_table(sorted_components),
        "eigenvectors": eigenvectors_to_table(sorted_components),
        "component_selection": selection.table,
        "selected_components": pd.DataFrame(
            {
                "Параметр": ["Режим", "Порог, %", "Выбрано компонент", "Компоненты новой базы"],
                "Значение": [
                    selection.mode,
                    selection.threshold * 100.0,
                    selection.n_components,
                    ", ".join(new_coordinates.selected_components),
                ],
            }
        ),
        "new_coordinates": new_coordinate_system_to_table(new_coordinates),
    }
