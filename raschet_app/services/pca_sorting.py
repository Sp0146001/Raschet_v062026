from __future__ import annotations

import numpy as np
import pandas as pd

from raschet_app.services.pca_eigendecomposition import EigenDecompositionResult


def sort_eigen_components_desc(result: EigenDecompositionResult) -> EigenDecompositionResult:
    """Отсортировать компоненты PCA по убыванию собственных значений.

    После сортировки первая компонента PC1 соответствует максимальной дисперсии,
    PC2 — следующей по величине дисперсии и т.д.
    """
    if result.eigenvalues is None or len(result.eigenvalues) == 0:
        raise ValueError("Нет собственных значений для сортировки компонент PCA.")
    if result.eigenvectors is None or result.eigenvectors.empty:
        raise ValueError("Нет собственных векторов для сортировки компонент PCA.")

    order = np.argsort(result.eigenvalues)[::-1]
    sorted_values = np.asarray(result.eigenvalues, dtype=float)[order]
    sorted_vectors_values = result.eigenvectors.to_numpy(dtype=float)[:, order]
    component_names = [f"PC{i + 1}" for i in range(len(sorted_values))]
    sorted_vectors = pd.DataFrame(
        sorted_vectors_values,
        index=result.eigenvectors.index,
        columns=component_names,
    )
    sorted_vectors.index.name = result.eigenvectors.index.name or "Признак"
    return EigenDecompositionResult(eigenvalues=sorted_values, eigenvectors=sorted_vectors)
