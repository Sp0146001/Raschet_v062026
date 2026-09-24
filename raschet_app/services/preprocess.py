from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from raschet_app.models import ChannelData


OPERATIONS = [
    ("log", "log(x)"),
    ("x_div_ref", "x/Xref"),
    ("ref_div_x", "Xref/x"),
    ("x_div_median", "x/median(x)"),
    ("zscore", "(x-mean(x))/std(x)"),
]


def operations_map() -> Dict[str, str]:
    return dict(OPERATIONS)


def algorithm_to_text(operations: Sequence[str]) -> str:
    labels = operations_map()
    known_keys = set(labels)
    normalized_ops = [op for op in operations if op in known_keys]
    if not normalized_ops:
        return "R"
    return " -> ".join(labels[op] for op in normalized_ops)


def _normalize_algorithm_token(token: str) -> str:
    return (
        token.strip()
        .lower()
        .replace(" ", "")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )


def parse_preprocess_algorithm(text: str) -> List[str]:
    raw_text = (text or "").strip()
    if not raw_text or _normalize_algorithm_token(raw_text) in {"r", "безпредобработки", "none", "нет"}:
        return []

    aliases = {
        "log": "log",
        "logx": "log",
        "log(x)": "log",
        "ln": "log",
        "ln(x)": "log",
        "x_div_ref": "x_div_ref",
        "x/xref": "x_div_ref",
        "x÷xref": "x_div_ref",
        "xdivxref": "x_div_ref",
        "xdivref": "x_div_ref",
        "xref_div_x": "ref_div_x",
        "ref_div_x": "ref_div_x",
        "xref/x": "ref_div_x",
        "xref÷x": "ref_div_x",
        "xrefdivx": "ref_div_x",
        "x_div_median": "x_div_median",
        "x/median": "x_div_median",
        "x/median(x)": "x_div_median",
        "x÷median(x)": "x_div_median",
        "xdivmedian": "x_div_median",
        "xdivmedian(x)": "x_div_median",
        "zscore": "zscore",
        "z-score": "zscore",
        "standardize": "zscore",
        "standardization": "zscore",
        "стандартизация": "zscore",
        "(x-mean(x))/std(x)": "zscore",
        "x-mean(x)/std(x)": "zscore",
        "(x-mean)/std": "zscore",
    }
    for key, label in OPERATIONS:
        aliases[_normalize_algorithm_token(key)] = key
        aliases[_normalize_algorithm_token(label)] = key

    parts = re.split(r"\s*(?:->|→|;|,|\n|\r)+\s*", raw_text)
    operations: List[str] = []
    unknown: List[str] = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        normalized = _normalize_algorithm_token(token)
        if normalized in {"r", "безпредобработки", "none", "нет"}:
            continue
        op_key = aliases.get(normalized)
        if op_key is None:
            unknown.append(token)
        else:
            operations.append(op_key)

    if unknown:
        supported = ", ".join(["R"] + [label for _, label in OPERATIONS])
        raise ValueError(
            "Не удалось распознать операцию(и) в алгоритме: "
            + "; ".join(unknown)
            + f". Поддерживаются: {supported}. Разделяйте операции через ->, запятую или точку с запятой."
        )
    return operations


def profile_to_json(profile: Dict[str, object]) -> str:
    return json.dumps(profile, ensure_ascii=False)


def profile_from_json(text: str) -> Dict[str, object]:
    return json.loads(text)


def build_common_time_matrix(channels: Iterable[ChannelData]) -> Tuple[np.ndarray, np.ndarray, List[ChannelData]]:
    channels = list(channels)
    if not channels:
        return np.array([]), np.empty((0, 0)), []

    base_channel = max(channels, key=lambda ch: len(ch.time))
    common_time = np.array(base_channel.time, dtype=float)
    matrix_cols = []
    for ch in channels:
        y = np.interp(common_time, ch.time, ch.resistance)
        matrix_cols.append(y)
    matrix = np.column_stack(matrix_cols)
    return common_time, matrix, channels


def _reference_vector(ref_channels: Iterable[ChannelData], ordered_channels: List[ChannelData]) -> np.ndarray:
    ref_map = {ch.index: np.array(ch.resistance, dtype=float) for ch in ref_channels}
    values = []
    for ch in ordered_channels:
        if ch.index not in ref_map:
            raise ValueError(f"В референсе отсутствует канал {ch.display_name}")
        arr = ref_map[ch.index]
        if len(arr) == 0:
            raise ValueError(f"В референсе нет данных по каналу {ch.display_name}")
        values.append(float(np.mean(arr)))
    return np.array(values, dtype=float)


def apply_preprocess_pipeline(
    channels: Iterable[ChannelData],
    profile: Dict[str, object],
    reference_channels: Optional[Iterable[ChannelData]] = None,
) -> List[ChannelData]:
    common_time, matrix, ordered_channels = build_common_time_matrix(channels)
    if matrix.size == 0:
        return []

    operations = profile.get("operations", []) or []
    ref_vector = None
    if any(op in {"x_div_ref", "ref_div_x"} for op in operations):
        if reference_channels is None:
            raise ValueError("Для операций Xref требуется выбрать референсный сегмент.")
        ref_vector = _reference_vector(reference_channels, ordered_channels)

    eps = 1.27e-127
    for op in operations:
        if op == "log":
            matrix = np.where(matrix <= 0, 1.0, matrix)
            matrix = np.log(matrix)
        elif op == "x_div_ref":
            matrix = matrix / (ref_vector[np.newaxis, :] + eps)
        elif op == "ref_div_x":
            matrix = ref_vector[np.newaxis, :] / (matrix + eps)
        elif op == "x_div_median":
            # Нормируем каждый канал на его собственную медиану по времени.
            # matrix имеет форму: строки = время, столбцы = каналы.
            med = np.median(matrix, axis=0, keepdims=True)
            matrix = matrix / (med + eps)
        elif op == "zscore":
            mean = np.mean(matrix, axis=0, keepdims=True)
            std = np.std(matrix, axis=0, keepdims=True)
            std = np.where(std == 0, 1.0, std)
            matrix = (matrix - mean) / std

    result: List[ChannelData] = []
    for idx, ch in enumerate(ordered_channels):
        result.append(
            ChannelData(
                index=ch.index,
                original_name=ch.original_name,
                display_name=ch.display_name,
                time=np.array(common_time, dtype=float),
                resistance=np.array(matrix[:, idx], dtype=float),
                color=ch.color,
            )
        )
    return result
