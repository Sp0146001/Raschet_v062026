from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


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
