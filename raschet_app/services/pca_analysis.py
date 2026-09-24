from __future__ import annotations

from typing import Dict

import pandas as pd

from raschet_app.services.pca_pipeline import run_pca_pipeline
from raschet_app.services.pca_scaling import SCALE_MODES, apply_scaling


def run_pca(
    feature_df: pd.DataFrame,
    scale_mode: str,
    variance_threshold: float = 0.95,
    min_components: int = 2,
) -> Dict[str, pd.DataFrame]:
    """Совместимый вход для UI. Вся логика PCA находится в pca_pipeline.py."""
    return run_pca_pipeline(feature_df, scale_mode, variance_threshold, min_components=min_components)
