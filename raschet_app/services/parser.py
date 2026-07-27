from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from raschet_app.constants import DEFAULT_COLORS
from raschet_app.models import ChannelData, ParsedFile


CHANNEL_RE = re.compile(r"\bchan{1,2}als?\b|\bchannels?\b|\bchannals?\b", re.IGNORECASE)
TIME_RESISTANCE_HEADER_RE = re.compile(r"\btime\b.*\bresist", re.IGNORECASE)
NUMBER_TOKEN_RE = re.compile(r"^[-+]?(?:(?:\d+(?:[.,]\d*)?)|(?:[.,]\d+))(?:[eE][-+]?\d+)?$")
NUMBER_FALLBACK_RE = re.compile(r"[-+]?(?:(?:\d+(?:[.,]\d*)?)|(?:[.,]\d+))(?:[eE][-+]?\d+)?")
METADATA_LABELS = [
    "chip",
    "voltage",
    "temperature",
    "temp",
    "gas",
    "ppm",
    "rpm",
    "date",
    "time",
    "sample",
]


class SmartTxtParser:
    @classmethod
    def parse(cls, file_path: str) -> ParsedFile:
        path = Path(file_path)
        lines = cls._read_lines(path)

        if not lines:
            raise ValueError("Файл пустой.")

        channel_idx = cls._find_channel_line_index(lines)
        data_header_idx = cls._find_data_header_index(lines)

        metadata_end_idx = data_header_idx if data_header_idx is not None else (channel_idx if channel_idx is not None else 0)
        metadata_lines = [line.strip() for line in lines[:metadata_end_idx] if line.strip()]
        metadata = cls._extract_metadata(metadata_lines)

        channel_numbers = cls._extract_channel_names(lines[channel_idx]) if channel_idx is not None else []

        if data_header_idx is not None:
            numeric_lines = lines[data_header_idx + 1 :]
        elif channel_idx is not None:
            numeric_lines = lines[channel_idx + 1 :]
        else:
            numeric_lines = lines

        numeric_rows = cls._extract_numeric_rows(numeric_lines)
        if not numeric_rows and channel_idx is not None and data_header_idx is not None and channel_idx > data_header_idx:
            # Запасной сценарий: если строка с каналами неожиданно идёт после заголовка данных.
            numeric_rows = cls._extract_numeric_rows(lines[data_header_idx + 1 : channel_idx])
        if not numeric_rows:
            raise ValueError("Не удалось найти числовые строки в TXT-файле.")

        target_column_count = cls._detect_column_count(numeric_rows, expected_pairs=len(channel_numbers))
        if target_column_count < 2:
            raise ValueError("Недостаточно числовых столбцов для разбора каналов.")

        normalized_rows = cls._normalize_rows(numeric_rows, target_column_count)
        df = pd.DataFrame(normalized_rows).dropna(axis=1, how="all")
        column_count = df.shape[1]
        if column_count < 2:
            raise ValueError("После очистки данных не осталось валидных столбцов.")

        channel_count = min(len(channel_numbers), column_count // 2) if channel_numbers else column_count // 2
        if channel_count == 0:
            channel_count = column_count // 2

        notes = []
        if column_count % 2 != 0:
            notes.append("Обнаружено нечётное количество столбцов; последний неполный столбец отброшен.")
            column_count -= 1
            df = df.iloc[:, :column_count]

        channels: List[ChannelData] = []
        for idx in range(column_count // 2):
            time_series = pd.to_numeric(df.iloc[:, idx * 2], errors="coerce")
            res_series = pd.to_numeric(df.iloc[:, idx * 2 + 1], errors="coerce")
            mask = time_series.notna() & res_series.notna()
            time = time_series[mask].to_numpy(dtype=float)
            resistance = res_series[mask].to_numpy(dtype=float)
            if len(time) == 0:
                continue

            order = np.argsort(time)
            time = time[order]
            resistance = resistance[order]

            base_name = channel_numbers[idx] if idx < len(channel_numbers) else str(idx + 1)
            display_name = f"Канал {base_name}"
            channels.append(
                ChannelData(
                    index=idx,
                    original_name=base_name,
                    display_name=display_name,
                    time=time,
                    resistance=resistance,
                    color=DEFAULT_COLORS[idx % len(DEFAULT_COLORS)],
                )
            )

        if not channels:
            raise ValueError("Не удалось собрать ни одного канала из файла.")

        return ParsedFile(
            file_path=path,
            file_name=path.name,
            metadata_lines=metadata_lines,
            metadata=metadata,
            channels=channels,
            raw_row_count=len(numeric_rows),
            column_count=column_count,
            notes=notes,
        )

    @staticmethod
    def _read_lines(path: Path) -> List[str]:
        encodings = ["utf-8-sig", "cp1251", "utf-8", "latin-1"]
        last_exc: Optional[Exception] = None
        for encoding in encodings:
            try:
                with path.open("r", encoding=encoding, errors="strict") as f:
                    return [line.rstrip("\n\r") for line in f]
            except Exception as exc:
                last_exc = exc
                continue

        with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
            return [line.rstrip("\n\r") for line in f]

    @staticmethod
    def _find_channel_line_index(lines: List[str]) -> Optional[int]:
        for i, line in enumerate(lines):
            if CHANNEL_RE.search(line or ""):
                return i
        return None

    @staticmethod
    def _find_data_header_index(lines: List[str]) -> Optional[int]:
        for i, line in enumerate(lines):
            if TIME_RESISTANCE_HEADER_RE.search(line or ""):
                return i
        return None

    @staticmethod
    def _extract_channel_names(line: str) -> List[str]:
        if not line:
            return []
        tail = line
        if ":" in line:
            tail = line.split(":", 1)[1]
        elif " " in line:
            tail = line.split(" ", 1)[1]
        tokens = [token.strip() for token in re.split(r"[,;\t ]+", tail) if token.strip()]
        return [token for token in tokens if token and not CHANNEL_RE.search(token)]

    @classmethod
    def _extract_metadata(cls, lines: List[str]) -> Dict[str, str]:
        metadata: Dict[str, str] = {}
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                metadata[key.strip()] = value.strip()
                continue
            low = stripped.lower()
            for label in METADATA_LABELS:
                if label in low:
                    metadata[label] = stripped
                    break
        return metadata

    @classmethod
    def _extract_numeric_rows(cls, lines: List[str]) -> List[List[float]]:
        rows: List[List[float]] = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            row = cls._parse_numeric_row(line)
            if row and len(row) >= 2:
                if len(row) % 2 != 0:
                    row = row[:-1]
                if len(row) >= 2:
                    rows.append(row)
        return rows

    @classmethod
    def _parse_numeric_row(cls, line: str) -> List[float]:
        if CHANNEL_RE.search(line) or TIME_RESISTANCE_HEADER_RE.search(line):
            return []

        # 1) Таб-разделённые строки: сохраняем пустые ячейки, чтобы не сдвигать пары Time/Resistance
        if "\t" in line:
            parts = [part.strip() for part in line.split("\t")]
            parsed = cls._parse_preserving_empty(parts)
            if parsed is not None:
                return parsed

        # 2) Строки с ';' (экспорт CSV): тоже сохраняем пустые ячейки
        if ";" in line:
            parts = [part.strip() for part in line.split(";")]
            parsed = cls._parse_preserving_empty(parts)
            if parsed is not None:
                return parsed

        # 3) Пробельный формат исходных приборных файлов
        parts = [p.strip() for p in re.split(r"\s+", line) if p.strip()]
        if len(parts) >= 2 and cls._parts_are_numeric(parts):
            return [cls._to_float(part) for part in parts]

        # 4) Запасной сценарий: извлечение чисел regex'ом
        tokens = NUMBER_FALLBACK_RE.findall(line)
        if len(tokens) >= 2:
            return [cls._to_float(token) for token in tokens]
        return []

    @classmethod
    def _parse_preserving_empty(cls, parts: List[str]) -> Optional[List[float]]:
        # Разрешаем пустые ячейки как NaN, но все непустые должны быть числами.
        non_empty = [part for part in parts if part != ""]
        if len(non_empty) < 2:
            return None
        if not cls._parts_are_numeric(non_empty):
            return None
        parsed: List[float] = []
        for part in parts:
            if part == "":
                parsed.append(np.nan)
            else:
                parsed.append(cls._to_float(part))
        return parsed

    @staticmethod
    def _parts_are_numeric(parts: List[str]) -> bool:
        return all(NUMBER_TOKEN_RE.match(part) for part in parts)

    @staticmethod
    def _to_float(token: str) -> float:
        token = token.strip().replace(",", ".")
        return float(token)

    @staticmethod
    def _detect_column_count(rows: List[List[float]], expected_pairs: int = 0) -> int:
        counts = Counter(len(row) - (len(row) % 2) for row in rows if len(row) >= 2)
        counts = Counter({k: v for k, v in counts.items() if k >= 2})
        if not counts:
            return 0

        expected_count = expected_pairs * 2
        if expected_count in counts:
            return expected_count

        return max(counts, key=lambda x: (counts[x], x))

    @staticmethod
    def _normalize_rows(rows: List[List[float]], target_columns: int) -> List[List[float]]:
        normalized: List[List[float]] = []
        for row in rows:
            if len(row) < 2:
                continue
            if len(row) >= target_columns:
                normalized.append(row[:target_columns])
            else:
                padded = row + [np.nan] * (target_columns - len(row))
                normalized.append(padded)
        return normalized
