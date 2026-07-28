from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from raschet_app.models import ChannelData


def compute_channel_features(channels: Iterable[ChannelData]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_long: List[Dict[str, object]] = []
    rows_wide: List[Dict[str, object]] = []

    for channel in channels:
        t = np.asarray(channel.time, dtype=float)
        y = np.asarray(channel.resistance, dtype=float)
        if len(t) == 0 or len(y) == 0:
            continue

        duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        delta_r = float(y[-1] - y[0])
        mean_r = float(np.mean(y))
        median_r = float(np.median(y))
        std_r = float(np.std(y))
        min_r = float(np.min(y))
        max_r = float(np.max(y))
        amplitude = float(max_r - min_r)
        if len(t) > 1:
            trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
            auc = float(trapz_fn(y, t))
        else:
            auc = float(y[0])
        slope = float(delta_r / duration) if duration != 0 else 0.0
        rms = float(np.sqrt(np.mean(np.square(y))))
        abs_diff_sum = float(np.sum(np.abs(np.diff(y)))) if len(y) > 1 else 0.0
        q10 = float(np.quantile(y, 0.10))
        q90 = float(np.quantile(y, 0.90))

        features = {
            "t_start": float(t[0]),
            "t_end": float(t[-1]),
            "duration": duration,
            "r_start": float(y[0]),
            "r_end": float(y[-1]),
            "delta_r": delta_r,
            "mean_r": mean_r,
            "median_r": median_r,
            "std_r": std_r,
            "min_r": min_r,
            "max_r": max_r,
            "amplitude": amplitude,
            "auc": auc,
            "slope": slope,
            "rms": rms,
            "abs_diff_sum": abs_diff_sum,
            "q10": q10,
            "q90": q90,
        }

        wide_row: Dict[str, object] = {
            "channel_index": channel.index,
            "channel_name": channel.display_name,
        }
        wide_row.update(features)
        rows_wide.append(wide_row)

        for key, value in features.items():
            rows_long.append(
                {
                    "channel_index": channel.index,
                    "channel_name": channel.display_name,
                    "feature_key": key,
                    "feature_value": value,
                }
            )

    return pd.DataFrame(rows_wide), pd.DataFrame(rows_long)
