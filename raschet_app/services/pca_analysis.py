from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from raschet_app.services.pca_covariance import compute_covariance_matrix, covariance_matrix_to_table
from raschet_app.services.pca_eigendecomposition import compute_eigendecomposition, eigenvalues_to_table, eigenvectors_to_table


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


def run_pca(feature_df: pd.DataFrame, scale_mode: str) -> Dict[str, pd.DataFrame]:
    if feature_df.empty:
        raise ValueError("Нет данных для PCA.")
    if len(feature_df) < 2:
        raise ValueError("Для PCA нужно как минимум два набора признаков.")

    X, mean, std = apply_scaling(feature_df, scale_mode)
    scaled_df = pd.DataFrame(X, index=feature_df.index, columns=feature_df.columns)
    covariance = compute_covariance_matrix(scaled_df)
    eigendecomposition = compute_eigendecomposition(covariance.matrix)
    U, S, VT = np.linalg.svd(X, full_matrices=False)

    n_samples = X.shape[0]
    eigenvalues = (S ** 2) / max(n_samples - 1, 1)
    total_variance = float(np.sum(eigenvalues)) if np.sum(eigenvalues) else 1.0
    explained_ratio = eigenvalues / total_variance
    cum_ratio = np.cumsum(explained_ratio)

    pc_names = [f"PC{i+1}" for i in range(len(eigenvalues))]
    variance_df = pd.DataFrame(
        {
            "Компонента": pc_names,
            "Собственное значение": eigenvalues,
            "Доля дисперсии, %": explained_ratio * 100.0,
            "Накопленная доля, %": cum_ratio * 100.0,
        }
    )

    scores = U * S
    scores_df = pd.DataFrame(scores, columns=pc_names)

    loadings = VT.T
    loadings_df = pd.DataFrame(loadings, index=feature_df.columns, columns=pc_names).reset_index().rename(columns={"index": "Признак"})

    means_df = pd.DataFrame({"Признак": feature_df.columns, "Среднее": mean.values, "Std": std.values})

    return {
        "variance": variance_df,
        "scores": scores_df,
        "loadings": loadings_df,
        "means": means_df,
        "covariance": covariance_matrix_to_table(covariance.matrix),
        "eigenvalues": eigenvalues_to_table(eigendecomposition),
        "eigenvectors": eigenvectors_to_table(eigendecomposition),
    }
