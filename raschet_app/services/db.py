from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from raschet_app.constants import DEFAULT_COLORS
from raschet_app.models import ChannelData, ParsedFile


DB_TABLES = [
    "meta",
    "source_files",
    "source_channels",
    "source_points",
    "segments",
    "segment_labels",
    "segment_channels",
    "segment_points",
]


@dataclass
class ProjectInfo:
    path: Path
    source_files_count: int
    segments_count: int


class ProjectDatabase:
    def __init__(self) -> None:
        self.path: Optional[Path] = None
        self.conn: Optional[sqlite3.Connection] = None

    # --------------------------------------------------------- lifecycle
    def create(self, path: str) -> None:
        db_path = Path(path)
        if db_path.exists():
            db_path.unlink()
        self._connect(db_path)
        self._init_schema()
        self.set_meta("created_at", datetime.now().isoformat(timespec="seconds"))

    def open(self, path: str) -> None:
        db_path = Path(path)
        if not db_path.exists():
            raise FileNotFoundError(f"Файл проекта не найден: {db_path}")
        self._connect(db_path)
        self._init_schema()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
        self.conn = None
        self.path = None

    def is_open(self) -> bool:
        return self.conn is not None and self.path is not None

    def _connect(self, path: Path) -> None:
        if self.conn is not None:
            self.conn.close()
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def _init_schema(self) -> None:
        assert self.conn is not None
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS source_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                file_path TEXT,
                imported_at TEXT,
                file_type TEXT,
                row_count INTEGER,
                channel_count INTEGER,
                metadata_json TEXT,
                notes_json TEXT
            );

            CREATE TABLE IF NOT EXISTS source_channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file_id INTEGER NOT NULL,
                channel_index INTEGER NOT NULL,
                original_name TEXT,
                display_name TEXT,
                FOREIGN KEY (source_file_id) REFERENCES source_files(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS source_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file_id INTEGER NOT NULL,
                channel_index INTEGER NOT NULL,
                time REAL NOT NULL,
                resistance REAL NOT NULL,
                FOREIGN KEY (source_file_id) REFERENCES source_files(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file_id INTEGER,
                source_file_name TEXT,
                segment_name TEXT,
                t_start REAL,
                t_end REAL,
                created_at TEXT,
                points_count INTEGER,
                comment TEXT,
                FOREIGN KEY (source_file_id) REFERENCES source_files(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS segment_labels (
                segment_id INTEGER PRIMARY KEY,
                gas_name TEXT,
                concentration_ppm TEXT,
                temperature_c TEXT,
                humidity_pct TEXT,
                light_mode TEXT,
                sample_group TEXT,
                class_label TEXT,
                comment TEXT,
                FOREIGN KEY (segment_id) REFERENCES segments(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS segment_channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id INTEGER NOT NULL,
                channel_index INTEGER NOT NULL,
                channel_name TEXT,
                FOREIGN KEY (segment_id) REFERENCES segments(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS segment_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id INTEGER NOT NULL,
                channel_index INTEGER NOT NULL,
                time REAL NOT NULL,
                resistance REAL NOT NULL,
                FOREIGN KEY (segment_id) REFERENCES segments(id) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()

    # --------------------------------------------------------- meta/info
    def set_meta(self, key: str, value: str) -> None:
        assert self.conn is not None
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        assert self.conn is not None
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def project_info(self) -> Optional[ProjectInfo]:
        if not self.is_open():
            return None
        assert self.conn is not None and self.path is not None
        row = self.conn.execute(
            "SELECT (SELECT COUNT(*) FROM source_files) AS files_count, (SELECT COUNT(*) FROM segments) AS segments_count"
        ).fetchone()
        return ProjectInfo(self.path, int(row["files_count"]), int(row["segments_count"]))

    # --------------------------------------------------------- import raw file
    def import_source_file(self, parsed: ParsedFile) -> int:
        assert self.conn is not None
        metadata_json = json.dumps(parsed.metadata, ensure_ascii=False)
        notes_json = json.dumps(parsed.notes, ensure_ascii=False)
        cur = self.conn.execute(
            """
            INSERT INTO source_files(file_name, file_path, imported_at, file_type, row_count, channel_count, metadata_json, notes_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                parsed.file_name,
                str(parsed.file_path),
                datetime.now().isoformat(timespec="seconds"),
                "txt",
                parsed.raw_row_count,
                len(parsed.channels),
                metadata_json,
                notes_json,
            ),
        )
        source_file_id = int(cur.lastrowid)

        channel_rows = []
        point_rows = []
        for channel in parsed.channels:
            channel_rows.append((source_file_id, channel.index, channel.original_name, channel.display_name))
            point_rows.extend(
                (source_file_id, channel.index, float(t), float(r))
                for t, r in zip(channel.time, channel.resistance)
            )

        self.conn.executemany(
            "INSERT INTO source_channels(source_file_id, channel_index, original_name, display_name) VALUES (?, ?, ?, ?)",
            channel_rows,
        )
        self.conn.executemany(
            "INSERT INTO source_points(source_file_id, channel_index, time, resistance) VALUES (?, ?, ?, ?)",
            point_rows,
        )
        self.conn.commit()
        return source_file_id

    def list_source_files(self) -> pd.DataFrame:
        if not self.is_open():
            return pd.DataFrame()
        assert self.conn is not None
        return pd.read_sql_query(
            "SELECT id, file_name, file_path, imported_at, row_count, channel_count FROM source_files ORDER BY id DESC",
            self.conn,
        )

    def load_source_file(self, source_file_id: int) -> ParsedFile:
        assert self.conn is not None
        src = self.conn.execute("SELECT * FROM source_files WHERE id=?", (source_file_id,)).fetchone()
        if src is None:
            raise ValueError("Файл-источник не найден в проекте.")

        ch_df = pd.read_sql_query(
            "SELECT channel_index, original_name, display_name FROM source_channels WHERE source_file_id=? ORDER BY channel_index",
            self.conn,
            params=(source_file_id,),
        )
        pt_df = pd.read_sql_query(
            "SELECT channel_index, time, resistance FROM source_points WHERE source_file_id=? ORDER BY channel_index, time",
            self.conn,
            params=(source_file_id,),
        )

        channels: List[ChannelData] = []
        for _, row in ch_df.iterrows():
            sub = pt_df[pt_df["channel_index"] == row["channel_index"]]
            channels.append(
                ChannelData(
                    index=int(row["channel_index"]),
                    original_name=str(row["original_name"]),
                    display_name=str(row["display_name"]),
                    time=sub["time"].to_numpy(dtype=float),
                    resistance=sub["resistance"].to_numpy(dtype=float),
                    color=DEFAULT_COLORS[int(row["channel_index"]) % len(DEFAULT_COLORS)],
                )
            )

        metadata = json.loads(src["metadata_json"] or "{}")
        notes = json.loads(src["notes_json"] or "[]")
        metadata_lines = [f"{k}: {v}" for k, v in metadata.items()] if metadata else []
        return ParsedFile(
            file_path=Path(src["file_path"] or src["file_name"]),
            file_name=str(src["file_name"]),
            metadata_lines=metadata_lines,
            metadata=metadata,
            channels=channels,
            raw_row_count=int(src["row_count"] or 0),
            column_count=int(src["channel_count"] or 0) * 2,
            notes=notes,
        )

    # --------------------------------------------------------- segments
    def save_segment(
        self,
        source_file_id: Optional[int],
        source_file_name: str,
        segment_name: str,
        interval: Tuple[float, float],
        labels: Dict[str, str],
        channels: Iterable[ChannelData],
    ) -> int:
        assert self.conn is not None
        channels = list(channels)
        total_points = sum(len(ch.time) for ch in channels)
        cur = self.conn.execute(
            """
            INSERT INTO segments(source_file_id, source_file_name, segment_name, t_start, t_end, created_at, points_count, comment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_file_id,
                source_file_name,
                segment_name,
                float(interval[0]),
                float(interval[1]),
                datetime.now().isoformat(timespec="seconds"),
                total_points,
                labels.get("comment", ""),
            ),
        )
        segment_id = int(cur.lastrowid)
        self.conn.execute(
            """
            INSERT INTO segment_labels(segment_id, gas_name, concentration_ppm, temperature_c, humidity_pct, light_mode, sample_group, class_label, comment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                segment_id,
                labels.get("gas_name", ""),
                labels.get("concentration_ppm", ""),
                labels.get("temperature_c", ""),
                labels.get("humidity_pct", ""),
                labels.get("light_mode", ""),
                labels.get("sample_group", ""),
                labels.get("class_label", ""),
                labels.get("comment", ""),
            ),
        )

        self.conn.executemany(
            "INSERT INTO segment_channels(segment_id, channel_index, channel_name) VALUES (?, ?, ?)",
            [(segment_id, ch.index, ch.display_name) for ch in channels],
        )
        point_rows = []
        for ch in channels:
            point_rows.extend((segment_id, ch.index, float(t), float(r)) for t, r in zip(ch.time, ch.resistance))
        self.conn.executemany(
            "INSERT INTO segment_points(segment_id, channel_index, time, resistance) VALUES (?, ?, ?, ?)",
            point_rows,
        )
        self.conn.commit()
        return segment_id

    def list_segments(self) -> pd.DataFrame:
        if not self.is_open():
            return pd.DataFrame()
        assert self.conn is not None
        return pd.read_sql_query(
            """
            SELECT s.id, s.segment_name, s.source_file_name, s.t_start, s.t_end, s.points_count, s.created_at,
                   l.gas_name, l.concentration_ppm, l.temperature_c, l.humidity_pct, l.light_mode, l.sample_group, l.class_label, l.comment
            FROM segments s
            LEFT JOIN segment_labels l ON l.segment_id = s.id
            ORDER BY s.id DESC
            """,
            self.conn,
        )

    def load_segment(self, segment_id: int) -> ParsedFile:
        assert self.conn is not None
        seg = self.conn.execute(
            "SELECT s.*, l.* FROM segments s LEFT JOIN segment_labels l ON l.segment_id=s.id WHERE s.id=?",
            (segment_id,),
        ).fetchone()
        if seg is None:
            raise ValueError("Сегмент не найден.")

        ch_df = pd.read_sql_query(
            "SELECT channel_index, channel_name FROM segment_channels WHERE segment_id=? ORDER BY channel_index",
            self.conn,
            params=(segment_id,),
        )
        pt_df = pd.read_sql_query(
            "SELECT channel_index, time, resistance FROM segment_points WHERE segment_id=? ORDER BY channel_index, time",
            self.conn,
            params=(segment_id,),
        )

        channels: List[ChannelData] = []
        for _, row in ch_df.iterrows():
            sub = pt_df[pt_df["channel_index"] == row["channel_index"]]
            channels.append(
                ChannelData(
                    index=int(row["channel_index"]),
                    original_name=str(row["channel_name"]),
                    display_name=str(row["channel_name"]),
                    time=sub["time"].to_numpy(dtype=float),
                    resistance=sub["resistance"].to_numpy(dtype=float),
                    color=DEFAULT_COLORS[int(row["channel_index"]) % len(DEFAULT_COLORS)],
                )
            )

        metadata = {
            "source_file_id": str(seg["source_file_id"] or ""),
            "gas": seg["gas_name"] or "",
            "concentration_ppm": seg["concentration_ppm"] or "",
            "temperature_c": seg["temperature_c"] or "",
            "light_mode": seg["light_mode"] or "",
            "comment": seg["comment"] or "",
        }
        metadata_lines = [
            f"gas: {metadata['gas']}" if metadata.get("gas") else "",
            f"concentration_ppm: {metadata['concentration_ppm']}" if metadata.get("concentration_ppm") else "",
            f"temperature_c: {metadata['temperature_c']}" if metadata.get("temperature_c") else "",
            f"light_mode: {metadata['light_mode']}" if metadata.get("light_mode") else "",
        ]
        metadata_lines = [line for line in metadata_lines if line]
        notes = [f"Сегмент: {seg['segment_name']}", f"Источник: {seg['source_file_name']}"]
        return ParsedFile(
            file_path=Path(seg["source_file_name"] or f"segment_{segment_id}"),
            file_name=f"{seg['segment_name']}",
            metadata_lines=metadata_lines,
            metadata=metadata,
            channels=channels,
            raw_row_count=int(seg["points_count"] or 0),
            column_count=len(channels) * 2,
            notes=notes,
        )

    # --------------------------------------------------------- export/import
    def export_project_copy(self, target_path: str) -> None:
        if not self.is_open() or self.path is None:
            raise ValueError("Сначала откройте проект SQLite.")
        self.conn.commit()
        shutil.copy2(self.path, target_path)

    def export_tables(self, target_path: str) -> str:
        assert self.conn is not None
        views = self._export_view_bundle()
        path = Path(target_path)
        if path.suffix.lower() == ".xlsx":
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for sheet_name, df in views.items():
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            return str(path)

        folder = path.with_name(path.stem + "_tables")
        folder.mkdir(parents=True, exist_ok=True)
        for sheet_name, df in views.items():
            safe_name = sheet_name.replace(" ", "_")
            df.to_csv(folder / f"{safe_name}.csv", sep=";", index=False, encoding="utf-8-sig")
        return str(folder)

    def import_tables(self, source_path: str) -> None:
        path = Path(source_path)
        if path.suffix.lower() == ".xlsx":
            data = pd.read_excel(path, sheet_name=None)
        elif path.is_dir():
            data = {csv.stem: pd.read_csv(csv, sep=";", encoding="utf-8-sig") for csv in path.glob("*.csv")}
        else:
            raise ValueError("Поддерживается импорт .xlsx или папки с CSV-таблицами.")
        normalized = self._normalize_import_bundle(data)
        self._import_table_bundle(normalized)

    def _export_view_bundle(self) -> Dict[str, pd.DataFrame]:
        assert self.conn is not None
        views: Dict[str, pd.DataFrame] = {}
        views["Исходные_файлы"] = pd.read_sql_query(
            "SELECT id AS 'ID файла', file_name AS 'Файл', row_count AS 'Строк', channel_count AS 'Каналов' FROM source_files ORDER BY id",
            self.conn,
        )
        views["Каналы_файлов"] = pd.read_sql_query(
            "SELECT source_file_id AS 'ID файла', channel_index AS '№ канала', original_name AS 'Исходное имя', display_name AS 'Отображаемое имя' FROM source_channels ORDER BY source_file_id, channel_index",
            self.conn,
        )
        views["Точки_файлов"] = pd.read_sql_query(
            "SELECT source_file_id AS 'ID файла', channel_index AS '№ канала', time AS 'Время, s', resistance AS 'Сопротивление, Ohm' FROM source_points ORDER BY source_file_id, channel_index, time",
            self.conn,
        )
        views["Сегменты"] = pd.read_sql_query(
            """
            SELECT s.id AS 'ID сегмента', s.source_file_id AS 'ID файла-источника', s.source_file_name AS 'Источник',
                   s.segment_name AS 'Сегмент', s.t_start AS 'Начало, s', s.t_end AS 'Конец, s', s.points_count AS 'Точек',
                   l.gas_name AS 'Газ', l.concentration_ppm AS 'Концентрация, ppm', l.temperature_c AS 'Температура, °C',
                   l.light_mode AS 'Свет', l.comment AS 'Комментарий'
            FROM segments s
            LEFT JOIN segment_labels l ON l.segment_id = s.id
            ORDER BY s.id
            """,
            self.conn,
        )
        views["Каналы_сегментов"] = pd.read_sql_query(
            "SELECT segment_id AS 'ID сегмента', channel_index AS '№ канала', channel_name AS 'Имя канала' FROM segment_channels ORDER BY segment_id, channel_index",
            self.conn,
        )
        views["Точки_сегментов"] = pd.read_sql_query(
            "SELECT segment_id AS 'ID сегмента', channel_index AS '№ канала', time AS 'Время, s', resistance AS 'Сопротивление, Ohm' FROM segment_points ORDER BY segment_id, channel_index, time",
            self.conn,
        )
        return views

    def _normalize_import_bundle(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        mapping = {
            "Исходные_файлы": {
                "ID файла": "id",
                "Файл": "file_name",
                "Строк": "row_count",
                "Каналов": "channel_count",
            },
            "Каналы_файлов": {
                "ID файла": "source_file_id",
                "№ канала": "channel_index",
                "Исходное имя": "original_name",
                "Отображаемое имя": "display_name",
            },
            "Точки_файлов": {
                "ID файла": "source_file_id",
                "№ канала": "channel_index",
                "Время, s": "time",
                "Сопротивление, Ohm": "resistance",
            },
            "Сегменты": {
                "ID сегмента": "id",
                "ID файла-источника": "source_file_id",
                "Источник": "source_file_name",
                "Сегмент": "segment_name",
                "Начало, s": "t_start",
                "Конец, s": "t_end",
                "Точек": "points_count",
                "Газ": "gas_name",
                "Концентрация, ppm": "concentration_ppm",
                "Температура, °C": "temperature_c",
                "Свет": "light_mode",
                "Комментарий": "comment",
            },
            "Каналы_сегментов": {
                "ID сегмента": "segment_id",
                "№ канала": "channel_index",
                "Имя канала": "channel_name",
            },
            "Точки_сегментов": {
                "ID сегмента": "segment_id",
                "№ канала": "channel_index",
                "Время, s": "time",
                "Сопротивление, Ohm": "resistance",
            },
        }
        normalized: Dict[str, pd.DataFrame] = {}
        for key, df in data.items():
            if key in mapping:
                normalized_name = {
                    "Исходные_файлы": "source_files",
                    "Каналы_файлов": "source_channels",
                    "Точки_файлов": "source_points",
                    "Сегменты": "segments",
                    "Каналы_сегментов": "segment_channels",
                    "Точки_сегментов": "segment_points",
                }[key]
                normalized[normalized_name] = df.rename(columns=mapping[key])
            else:
                normalized[key] = df
        return normalized

    def _import_table_bundle(self, data: Dict[str, pd.DataFrame]) -> None:
        assert self.conn is not None
        source_map: Dict[int, int] = {}
        segment_map: Dict[int, int] = {}
        with self.conn:
            for _, row in data.get("source_files", pd.DataFrame()).iterrows():
                cur = self.conn.execute(
                    """
                    INSERT INTO source_files(file_name, file_path, imported_at, file_type, row_count, channel_count, metadata_json, notes_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("file_name", ""),
                        row.get("file_path", ""),
                        row.get("imported_at", datetime.now().isoformat(timespec="seconds")),
                        row.get("file_type", "txt"),
                        int(row.get("row_count", 0) or 0),
                        int(row.get("channel_count", 0) or 0),
                        row.get("metadata_json", "{}"),
                        row.get("notes_json", "[]"),
                    ),
                )
                source_map[int(row.get("id", len(source_map) + 1))] = int(cur.lastrowid)

            for _, row in data.get("source_channels", pd.DataFrame()).iterrows():
                old_id = int(row.get("source_file_id"))
                if old_id not in source_map:
                    continue
                self.conn.execute(
                    "INSERT INTO source_channels(source_file_id, channel_index, original_name, display_name) VALUES (?, ?, ?, ?)",
                    (
                        source_map[old_id],
                        int(row.get("channel_index", 0)),
                        row.get("original_name", ""),
                        row.get("display_name", ""),
                    ),
                )

            for _, row in data.get("source_points", pd.DataFrame()).iterrows():
                old_id = int(row.get("source_file_id"))
                if old_id not in source_map:
                    continue
                self.conn.execute(
                    "INSERT INTO source_points(source_file_id, channel_index, time, resistance) VALUES (?, ?, ?, ?)",
                    (
                        source_map[old_id],
                        int(row.get("channel_index", 0)),
                        float(row.get("time", 0.0)),
                        float(row.get("resistance", 0.0)),
                    ),
                )

            for _, row in data.get("segments", pd.DataFrame()).iterrows():
                old_source = row.get("source_file_id")
                mapped_source = source_map.get(int(old_source)) if pd.notna(old_source) else None
                cur = self.conn.execute(
                    """
                    INSERT INTO segments(source_file_id, source_file_name, segment_name, t_start, t_end, created_at, points_count, comment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mapped_source,
                        row.get("source_file_name", ""),
                        row.get("segment_name", ""),
                        float(row.get("t_start", 0.0)),
                        float(row.get("t_end", 0.0)),
                        row.get("created_at", datetime.now().isoformat(timespec="seconds")),
                        int(row.get("points_count", 0) or 0),
                        row.get("comment", ""),
                    ),
                )
                segment_map[int(row.get("id", len(segment_map) + 1))] = int(cur.lastrowid)

            segment_labels_df = data.get("segment_labels", pd.DataFrame())
            if segment_labels_df.empty and "segments" in data:
                seg_copy = data["segments"].copy()
                if "id" in seg_copy.columns:
                    segment_labels_df = pd.DataFrame(
                        {
                            "segment_id": seg_copy.get("id"),
                            "gas_name": seg_copy.get("gas_name", ""),
                            "concentration_ppm": seg_copy.get("concentration_ppm", ""),
                            "temperature_c": seg_copy.get("temperature_c", ""),
                            "humidity_pct": "",
                            "light_mode": seg_copy.get("light_mode", ""),
                            "sample_group": "",
                            "class_label": "",
                            "comment": seg_copy.get("comment", ""),
                        }
                    )
            for _, row in segment_labels_df.iterrows():
                old_segment = int(row.get("segment_id"))
                if old_segment not in segment_map:
                    continue
                self.conn.execute(
                    """
                    INSERT INTO segment_labels(segment_id, gas_name, concentration_ppm, temperature_c, humidity_pct, light_mode, sample_group, class_label, comment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        segment_map[old_segment],
                        row.get("gas_name", ""),
                        row.get("concentration_ppm", ""),
                        row.get("temperature_c", ""),
                        row.get("humidity_pct", ""),
                        row.get("light_mode", ""),
                        row.get("sample_group", ""),
                        row.get("class_label", ""),
                        row.get("comment", ""),
                    ),
                )

            for _, row in data.get("segment_channels", pd.DataFrame()).iterrows():
                old_segment = int(row.get("segment_id"))
                if old_segment not in segment_map:
                    continue
                self.conn.execute(
                    "INSERT INTO segment_channels(segment_id, channel_index, channel_name) VALUES (?, ?, ?)",
                    (
                        segment_map[old_segment],
                        int(row.get("channel_index", 0)),
                        row.get("channel_name", ""),
                    ),
                )

            for _, row in data.get("segment_points", pd.DataFrame()).iterrows():
                old_segment = int(row.get("segment_id"))
                if old_segment not in segment_map:
                    continue
                self.conn.execute(
                    "INSERT INTO segment_points(segment_id, channel_index, time, resistance) VALUES (?, ?, ?, ?)",
                    (
                        segment_map[old_segment],
                        int(row.get("channel_index", 0)),
                        float(row.get("time", 0.0)),
                        float(row.get("resistance", 0.0)),
                    ),
                )

    def delete_segment(self, segment_id: int) -> None:
        assert self.conn is not None
        self.conn.execute("DELETE FROM segments WHERE id=?", (segment_id,))
        self.conn.commit()
