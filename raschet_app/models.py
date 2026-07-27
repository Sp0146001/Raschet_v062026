from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class ChannelData:
    index: int
    original_name: str
    display_name: str
    time: np.ndarray
    resistance: np.ndarray
    color: str
    visible: bool = True


@dataclass
class ParsedFile:
    file_path: Path
    file_name: str
    metadata_lines: List[str]
    metadata: Dict[str, str]
    channels: List[ChannelData]
    raw_row_count: int
    column_count: int
    notes: List[str] = field(default_factory=list)
