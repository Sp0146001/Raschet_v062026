from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from raschet_app.constants import CALC_MODES
from raschet_app.models import ChannelData


def format_sci(value: float) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    return f"{value:.2e}"


def normalize_interval(start: float, end: float) -> Optional[Tuple[float, float]]:
    try:
        start = float(start)
        end = float(end)
    except (TypeError, ValueError):
        return None
    if math.isclose(start, end):
        return None
    return (start, end) if start < end else (end, start)


def slice_channel(channel: ChannelData, interval: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    start, end = interval
    mask = (channel.time >= start) & (channel.time <= end)
    return channel.time[mask], channel.resistance[mask]


def build_slice_dataframe(channels: Iterable[ChannelData], interval: Tuple[float, float]) -> pd.DataFrame:
    data: Dict[str, pd.Series] = {}
    for ch in channels:
        time, resistance = slice_channel(ch, interval)
        if len(time) == 0:
            continue
        data[f"{ch.display_name}__Time_s"] = pd.Series(time)
        data[f"{ch.display_name}__Resistance_Ohm"] = pd.Series(resistance)
    return pd.DataFrame(data)


def interpolate_value(x: np.ndarray, y: np.ndarray, x_target: float) -> float:
    if len(x) == 0:
        return float("nan")
    if x_target <= x[0]:
        return float(y[0])
    if x_target >= x[-1]:
        return float(y[-1])
    return float(np.interp(x_target, x, y))


def find_crossing_time(x: np.ndarray, y: np.ndarray, target: float) -> float:
    if len(x) == 0:
        return float("nan")

    diff = y - target
    for i in range(len(diff) - 1):
        d1 = diff[i]
        d2 = diff[i + 1]
        if d1 == 0:
            return float(x[i])
        if d1 * d2 <= 0:
            x1, x2 = float(x[i]), float(x[i + 1])
            y1, y2 = float(y[i]), float(y[i + 1])
            if y2 == y1:
                return x1
            return x1 + (target - y1) * (x2 - x1) / (y2 - y1)

    idx = int(np.argmin(np.abs(diff)))
    return float(x[idx])


def build_xline_dataframe(channels: Iterable[ChannelData], x_value: float) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for channel in channels:
        nearest_idx = int(np.argmin(np.abs(channel.time - x_value)))
        nearest_time = float(channel.time[nearest_idx])
        nearest_r = float(channel.resistance[nearest_idx])
        interpolated = interpolate_value(channel.time, channel.resistance, x_value)
        rows.append(
            {
                "Канал": channel.display_name,
                "Rᵢ (interp)": format_sci(interpolated),
                "t ближ.": f"{nearest_time:.4f}",
                "R ближ.": format_sci(nearest_r),
            }
        )
    return pd.DataFrame(rows)


def calculate_results(channels: Iterable[ChannelData], interval: Tuple[float, float], variant: str, avg_points: int) -> pd.DataFrame:
    mode = CALC_MODES[variant]
    fraction = float(mode["fraction"])
    fraction_label = str(mode["fraction_label"])
    s_mode = str(mode["s_mode"])
    time_row_1, time_row_2 = mode["time_rows"]

    metric_order = [time_row_1, time_row_2, "R0", "R(g)", f"R({fraction_label})", f"t({fraction_label})", "S"]
    results_by_channel: Dict[str, Dict[str, str]] = {}

    for channel in channels:
        x, y = slice_channel(channel, interval)
        if len(x) == 0:
            results_by_channel[channel.display_name] = {metric: "N/A" for metric in metric_order}
            continue

        avg_n = max(1, min(int(avg_points), len(y)))
        t_start = float(x[0])
        t_end = float(x[-1])
        r0 = float(np.mean(y[:avg_n]))
        rg = float(np.mean(y[-avg_n:]))
        r_fraction = r0 + fraction * (rg - r0)
        t_cross_abs = find_crossing_time(x, y, r_fraction)
        t_fraction = t_cross_abs - t_start

        if s_mode == "grow":
            s = (rg / r0) if r0 != 0 else np.nan
        else:
            s = (r0 / rg) if rg != 0 else np.nan

        results_by_channel[channel.display_name] = {
            time_row_1: f"{t_start:.4f}",
            time_row_2: f"{t_end:.4f}",
            "R0": format_sci(r0),
            "R(g)": format_sci(rg),
            f"R({fraction_label})": format_sci(r_fraction),
            f"t({fraction_label})": f"{t_fraction:.4f}",
            "S": format_sci(float(s)),
        }

    result = pd.DataFrame(results_by_channel)
    return result.reindex(metric_order)
