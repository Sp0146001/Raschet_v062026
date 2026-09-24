from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ComponentSelectionResult:
    n_components: int
    threshold: float
    mode: str
    table: pd.DataFrame


def select_components(
    eigenvalues: np.ndarray,
    variance_threshold: float = 0.95,
    min_components: int = 2,
) -> ComponentSelectionResult:
    """Выбрать число главных компонент по порогу накопленной дисперсии.

    По умолчанию выбирается минимальное число компонент, которое объясняет
    не меньше 95% общей дисперсии, но не меньше двух компонент, если это
    позволяет размерность данных. Это нужно для стабильного 2D-представления.
    """
    values = np.asarray(eigenvalues, dtype=float)
    if values.size == 0:
        raise ValueError("Нет собственных значений для выбора числа компонент PCA.")

    # Из-за численных ошибок очень малые отрицательные значения ковариационной
    # матрицы могут появиться как -1e-15. Для долей дисперсии их обнуляем.
    values = np.where(values < 0, 0.0, values)
    total = float(np.sum(values))
    if total <= 0:
        explained_ratio = np.zeros_like(values, dtype=float)
    else:
        explained_ratio = values / total
    cumulative_ratio = np.cumsum(explained_ratio)

    max_components = len(values)
    threshold = min(max(float(variance_threshold), 0.0), 1.0)
    effective_min_components = min(max(int(min_components), 1), max_components)

    reached = np.where(cumulative_ratio >= threshold)[0]
    auto_components = int(reached[0] + 1) if len(reached) else max_components
    n_components = min(max(auto_components, effective_min_components), max_components)
    mode = f"Авто по порогу {threshold * 100:.1f}%, минимум {effective_min_components} PC"

    rows = []
    for idx, value in enumerate(values):
        rows.append(
            {
                "Компонента": f"PC{idx + 1}",
                "Собственное значение": value,
                "Доля дисперсии, %": explained_ratio[idx] * 100.0,
                "Накопленная доля, %": cumulative_ratio[idx] * 100.0,
                "Выбрана": "Да" if idx < n_components else "Нет",
            }
        )

    return ComponentSelectionResult(
        n_components=n_components,
        threshold=threshold,
        mode=mode,
        table=pd.DataFrame(rows),
    )
