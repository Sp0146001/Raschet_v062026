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
    "preprocess_profiles",
    "preprocessed_runs",
    "preprocessed_channels",
    "preprocessed_points",
    "feature_sets",
    "feature_values",
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
                is_reference INTEGER DEFAULT 0,
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

            CREATE TABLE IF NOT EXISTS preprocess_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_name TEXT NOT NULL,
                pipeline_json TEXT NOT NULL,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS preprocessed_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id INTEGER NOT NULL,
                profile_id INTEGER,
                reference_segment_id INTEGER,
                run_name TEXT,
                created_at TEXT,
                profile_json TEXT,
                FOREIGN KEY (segment_id) REFERENCES segments(id) ON DELETE CASCADE,
                FOREIGN KEY (profile_id) REFERENCES preprocess_profiles(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS preprocessed_channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                channel_index INTEGER NOT NULL,
                channel_name TEXT,
                FOREIGN KEY (run_id) REFERENCES preprocessed_runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS preprocessed_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                channel_index INTEGER NOT NULL,
                time REAL NOT NULL,
                resistance REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES preprocessed_runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS feature_sets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_set_name TEXT NOT NULL,
                segment_id INTEGER,
                run_id INTEGER,
                profile_id INTEGER,
                source_kind TEXT,
                created_at TEXT,
                note TEXT,
                FOREIGN KEY (segment_id) REFERENCES segments(id) ON DELETE SET NULL,
                FOREIGN KEY (run_id) REFERENCES preprocessed_runs(id) ON DELETE SET NULL,
                FOREIGN KEY (profile_id) REFERENCES preprocess_profiles(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS feature_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_set_id INTEGER NOT NULL,
                channel_index INTEGER NOT NULL,
                channel_name TEXT,
                feature_key TEXT,
                feature_value REAL,
                FOREIGN KEY (feature_set_id) REFERENCES feature_sets(id) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()
        self._ensure_schema_updates()

    def _ensure_schema_updates(self) -> None:
        assert self.conn is not None
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(segment_labels)").fetchall()}
        if "is_reference" not in cols:
            self.conn.execute("ALTER TABLE segment_labels ADD COLUMN is_reference INTEGER DEFAULT 0")
        self._cleanup_legacy_feature_notes()
        self.conn.commit()

    def _cleanup_legacy_feature_notes(self) -> None:
        """Заменить старый служебный комментарий наборов признаков на комментарий исходного сегмента."""
        assert self.conn is not None
        self.conn.execute(
            """
            UPDATE feature_sets
            SET note = COALESCE(
                (
                    SELECT COALESCE(l.comment, s.comment, '')
                    FROM segments s
                    LEFT JOIN segment_labels l ON l.segment_id = s.id
                    WHERE s.id = COALESCE(
                        feature_sets.segment_id,
                        (SELECT r.segment_id FROM preprocessed_runs r WHERE r.id = feature_sets.run_id)
                    )
                ),
                ''
            )
            WHERE TRIM(COALESCE(note, '')) IN ('Рассчитано в Raschet', 'Рассчитано в Raschet.')
            """
        )

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
            INSERT INTO segment_labels(segment_id, gas_name, concentration_ppm, temperature_c, humidity_pct, light_mode, sample_group, class_label, comment, is_reference)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                1 if labels.get("is_reference") in {True, "1", "true", "True", "Да", "да", "yes", "YES"} else 0,
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
                   l.gas_name, l.concentration_ppm, l.temperature_c, l.humidity_pct, l.light_mode, l.sample_group, l.class_label, l.comment, l.is_reference
            FROM segments s
            LEFT JOIN segment_labels l ON l.segment_id = s.id
            ORDER BY s.id DESC
            """,
            self.conn,
        )

    def get_segment_comment(self, segment_id: int) -> str:
        if not self.is_open():
            return ""
        assert self.conn is not None
        row = self.conn.execute(
            """
            SELECT COALESCE(l.comment, s.comment, '') AS comment
            FROM segments s
            LEFT JOIN segment_labels l ON l.segment_id = s.id
            WHERE s.id = ?
            """,
            (segment_id,),
        ).fetchone()
        return str(row["comment"] or "") if row is not None else ""

    def find_reference_segment_id(self, segment_id: int) -> Optional[int]:
        """Найти референсный сегмент для указанного сегмента.

        Приоритет:
        1) другой сегмент из того же исходного файла;
        2) совпадение температуры;
        3) совпадение режима света;
        4) самый новый подходящий референс.

        Если другого референса нет, допускается сам segment_id, если он помечен как референсный.
        """
        if not self.is_open():
            return None
        assert self.conn is not None
        row = self.conn.execute(
            """
            SELECT rs.id
            FROM segments ts
            LEFT JOIN segment_labels tl ON tl.segment_id = ts.id
            JOIN segment_labels rl ON COALESCE(rl.is_reference, 0) = 1
            JOIN segments rs ON rs.id = rl.segment_id
            WHERE ts.id = ?
            ORDER BY
                CASE WHEN rs.id <> ts.id THEN 0 ELSE 1 END,
                CASE WHEN rs.source_file_id = ts.source_file_id THEN 0 ELSE 1 END,
                CASE WHEN COALESCE(rl.temperature_c, '') = COALESCE(tl.temperature_c, '') THEN 0 ELSE 1 END,
                CASE WHEN COALESCE(rl.light_mode, '') = COALESCE(tl.light_mode, '') THEN 0 ELSE 1 END,
                rs.id DESC
            LIMIT 1
            """,
            (segment_id,),
        ).fetchone()
        return int(row["id"]) if row is not None else None

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
            "is_reference": "1" if int(seg["is_reference"] or 0) else "0",
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
            "SELECT id AS 'ID файла', file_name AS 'Файл', file_path AS 'Источник', row_count AS 'Строк', channel_count AS 'Каналов' FROM source_files ORDER BY id",
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
                   l.light_mode AS 'Свет', CASE WHEN COALESCE(l.is_reference, 0)=1 THEN 'Да' ELSE 'Нет' END AS 'Референс', l.comment AS 'Комментарий'
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
        views["Профили_предобработки"] = pd.read_sql_query(
            "SELECT id AS 'ID профиля', profile_name AS 'Профиль', pipeline_json AS 'Pipeline JSON' FROM preprocess_profiles ORDER BY id",
            self.conn,
        )
        views["Запуски_предобработки"] = pd.read_sql_query(
            """
            SELECT r.id AS 'ID запуска', r.run_name AS 'Название', r.segment_id AS 'ID сегмента', s.segment_name AS 'Сегмент',
                   r.profile_id AS 'ID профиля', p.profile_name AS 'Профиль', r.reference_segment_id AS 'ID референса', r.profile_json AS 'Pipeline JSON'
            FROM preprocessed_runs r
            LEFT JOIN segments s ON s.id = r.segment_id
            LEFT JOIN preprocess_profiles p ON p.id = r.profile_id
            ORDER BY r.id
            """,
            self.conn,
        )
        views["Каналы_предобработки"] = pd.read_sql_query(
            "SELECT run_id AS 'ID запуска', channel_index AS '№ канала', channel_name AS 'Имя канала' FROM preprocessed_channels ORDER BY run_id, channel_index",
            self.conn,
        )
        views["Точки_предобработки"] = pd.read_sql_query(
            "SELECT run_id AS 'ID запуска', channel_index AS '№ канала', time AS 'Время, s', resistance AS 'Значение' FROM preprocessed_points ORDER BY run_id, channel_index, time",
            self.conn,
        )
        views["Наборы_признаков"] = pd.read_sql_query(
            """
            SELECT f.id AS 'ID набора', f.feature_set_name AS 'Набор признаков', f.segment_id AS 'ID сегмента', s.segment_name AS 'Сегмент',
                   f.run_id AS 'ID запуска', r.run_name AS 'Результат предобработки', f.profile_id AS 'ID профиля', p.profile_name AS 'Профиль',
                   f.source_kind AS 'Источник признаков', f.created_at AS 'Создан', f.note AS 'Комментарий'
            FROM feature_sets f
            LEFT JOIN segments s ON s.id = f.segment_id
            LEFT JOIN preprocessed_runs r ON r.id = f.run_id
            LEFT JOIN preprocess_profiles p ON p.id = f.profile_id
            ORDER BY f.id
            """,
            self.conn,
        )
        views["Значения_признаков"] = pd.read_sql_query(
            "SELECT feature_set_id AS 'ID набора', channel_index AS '№ канала', channel_name AS 'Имя канала', feature_key AS 'Признак', feature_value AS 'Значение' FROM feature_values ORDER BY feature_set_id, channel_index, feature_key",
            self.conn,
        )
        return views

    def _normalize_import_bundle(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        mapping = {
            "Исходные_файлы": {
                "ID файла": "id",
                "Файл": "file_name",
                "Источник": "file_path",
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
                "Референс": "is_reference",
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
            "Профили_предобработки": {
                "ID профиля": "id",
                "Профиль": "profile_name",
                "Pipeline JSON": "pipeline_json",
            },
            "Запуски_предобработки": {
                "ID запуска": "id",
                "Название": "run_name",
                "ID сегмента": "segment_id",
                "Сегмент": "segment_name",
                "ID профиля": "profile_id",
                "Профиль": "profile_name",
                "ID референса": "reference_segment_id",
                "Pipeline JSON": "profile_json",
            },
            "Каналы_предобработки": {
                "ID запуска": "run_id",
                "№ канала": "channel_index",
                "Имя канала": "channel_name",
            },
            "Точки_предобработки": {
                "ID запуска": "run_id",
                "№ канала": "channel_index",
                "Время, s": "time",
                "Значение": "resistance",
            },
            "Наборы_признаков": {
                "ID набора": "id",
                "Набор признаков": "feature_set_name",
                "ID сегмента": "segment_id",
                "Сегмент": "segment_name",
                "ID запуска": "run_id",
                "Результат предобработки": "run_name",
                "ID профиля": "profile_id",
                "Профиль": "profile_name",
                "Источник признаков": "source_kind",
                "Создан": "created_at",
                "Комментарий": "note",
            },
            "Значения_признаков": {
                "ID набора": "feature_set_id",
                "№ канала": "channel_index",
                "Имя канала": "channel_name",
                "Признак": "feature_key",
                "Значение": "feature_value",
            },
        }
        normalized_names = {
            "Исходные_файлы": "source_files",
            "Каналы_файлов": "source_channels",
            "Точки_файлов": "source_points",
            "Сегменты": "segments",
            "Каналы_сегментов": "segment_channels",
            "Точки_сегментов": "segment_points",
            "Профили_предобработки": "preprocess_profiles",
            "Запуски_предобработки": "preprocessed_runs",
            "Каналы_предобработки": "preprocessed_channels",
            "Точки_предобработки": "preprocessed_points",
            "Наборы_признаков": "feature_sets",
            "Значения_признаков": "feature_values",
        }
        normalized: Dict[str, pd.DataFrame] = {}
        for key, df in data.items():
            if key in mapping:
                normalized[normalized_names[key]] = df.rename(columns=mapping[key])
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
                            "is_reference": seg_copy.get("is_reference", 0),
                        }
                    )
            for _, row in segment_labels_df.iterrows():
                old_segment = int(row.get("segment_id"))
                if old_segment not in segment_map:
                    continue
                self.conn.execute(
                    """
                    INSERT INTO segment_labels(segment_id, gas_name, concentration_ppm, temperature_c, humidity_pct, light_mode, sample_group, class_label, comment, is_reference)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        1 if row.get("is_reference", 0) in {1, "1", True, "Да", "да", "yes", "YES", "true", "True"} else 0,
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

            profile_map: Dict[int, int] = {}
            for _, row in data.get("preprocess_profiles", pd.DataFrame()).iterrows():
                cur = self.conn.execute(
                    "INSERT INTO preprocess_profiles(profile_name, pipeline_json, created_at) VALUES (?, ?, ?)",
                    (
                        row.get("profile_name", ""),
                        row.get("pipeline_json", "{}"),
                        row.get("created_at", datetime.now().isoformat(timespec="seconds")),
                    ),
                )
                profile_map[int(row.get("id", len(profile_map) + 1))] = int(cur.lastrowid)

            run_map: Dict[int, int] = {}
            for _, row in data.get("preprocessed_runs", pd.DataFrame()).iterrows():
                old_segment = row.get("segment_id")
                mapped_segment = segment_map.get(int(old_segment)) if pd.notna(old_segment) else None
                old_profile = row.get("profile_id")
                mapped_profile = profile_map.get(int(old_profile)) if pd.notna(old_profile) else None
                old_ref = row.get("reference_segment_id")
                mapped_ref = segment_map.get(int(old_ref)) if pd.notna(old_ref) else None
                cur = self.conn.execute(
                    """
                    INSERT INTO preprocessed_runs(segment_id, profile_id, reference_segment_id, run_name, created_at, profile_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mapped_segment,
                        mapped_profile,
                        mapped_ref,
                        row.get("run_name", ""),
                        row.get("created_at", datetime.now().isoformat(timespec="seconds")),
                        row.get("profile_json", "{}"),
                    ),
                )
                run_map[int(row.get("id", len(run_map) + 1))] = int(cur.lastrowid)

            for _, row in data.get("preprocessed_channels", pd.DataFrame()).iterrows():
                old_run = int(row.get("run_id"))
                if old_run not in run_map:
                    continue
                self.conn.execute(
                    "INSERT INTO preprocessed_channels(run_id, channel_index, channel_name) VALUES (?, ?, ?)",
                    (run_map[old_run], int(row.get("channel_index", 0)), row.get("channel_name", "")),
                )

            for _, row in data.get("preprocessed_points", pd.DataFrame()).iterrows():
                old_run = int(row.get("run_id"))
                if old_run not in run_map:
                    continue
                self.conn.execute(
                    "INSERT INTO preprocessed_points(run_id, channel_index, time, resistance) VALUES (?, ?, ?, ?)",
                    (
                        run_map[old_run],
                        int(row.get("channel_index", 0)),
                        float(row.get("time", 0.0)),
                        float(row.get("resistance", 0.0)),
                    ),
                )

            feature_set_map: Dict[int, int] = {}
            for _, row in data.get("feature_sets", pd.DataFrame()).iterrows():
                old_segment = row.get("segment_id")
                mapped_segment = segment_map.get(int(old_segment)) if pd.notna(old_segment) else None
                old_run = row.get("run_id")
                mapped_run = run_map.get(int(old_run)) if pd.notna(old_run) else None
                old_profile = row.get("profile_id")
                mapped_profile = profile_map.get(int(old_profile)) if pd.notna(old_profile) else None
                cur = self.conn.execute(
                    """
                    INSERT INTO feature_sets(feature_set_name, segment_id, run_id, profile_id, source_kind, created_at, note)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("feature_set_name", ""),
                        mapped_segment,
                        mapped_run,
                        mapped_profile,
                        row.get("source_kind", "segment"),
                        row.get("created_at", datetime.now().isoformat(timespec="seconds")),
                        row.get("note", ""),
                    ),
                )
                feature_set_map[int(row.get("id", len(feature_set_map) + 1))] = int(cur.lastrowid)

            for _, row in data.get("feature_values", pd.DataFrame()).iterrows():
                old_feature_set = int(row.get("feature_set_id"))
                if old_feature_set not in feature_set_map:
                    continue
                self.conn.execute(
                    "INSERT INTO feature_values(feature_set_id, channel_index, channel_name, feature_key, feature_value) VALUES (?, ?, ?, ?, ?)",
                    (
                        feature_set_map[old_feature_set],
                        int(row.get("channel_index", 0)),
                        row.get("channel_name", ""),
                        row.get("feature_key", ""),
                        float(row.get("feature_value", 0.0)),
                    ),
                )

    def delete_segment(self, segment_id: int) -> None:
        assert self.conn is not None
        self.conn.execute("DELETE FROM segments WHERE id=?", (segment_id,))
        self.conn.commit()

    # --------------------------------------------------------- preprocess profiles / runs
    def save_preprocess_profile(self, profile_name: str, pipeline_json: str) -> int:
        assert self.conn is not None
        cur = self.conn.execute(
            "INSERT INTO preprocess_profiles(profile_name, pipeline_json, created_at) VALUES (?, ?, ?)",
            (profile_name, pipeline_json, datetime.now().isoformat(timespec="seconds")),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def update_preprocess_profile(self, profile_id: int, profile_name: str, pipeline_json: str) -> None:
        assert self.conn is not None
        self.conn.execute(
            "UPDATE preprocess_profiles SET profile_name=?, pipeline_json=? WHERE id=?",
            (profile_name, pipeline_json, profile_id),
        )
        self.conn.commit()

    def list_preprocess_profiles(self) -> pd.DataFrame:
        if not self.is_open():
            return pd.DataFrame()
        assert self.conn is not None
        return pd.read_sql_query(
            "SELECT id, profile_name, pipeline_json, created_at FROM preprocess_profiles ORDER BY id DESC",
            self.conn,
        )

    def load_preprocess_profile(self, profile_id: int) -> Dict[str, object]:
        assert self.conn is not None
        row = self.conn.execute("SELECT * FROM preprocess_profiles WHERE id=?", (profile_id,)).fetchone()
        if row is None:
            raise ValueError("Профиль предобработки не найден.")
        return {"id": int(row["id"]), "profile_name": row["profile_name"], "pipeline_json": row["pipeline_json"]}

    def delete_preprocess_profile(self, profile_id: int) -> None:
        assert self.conn is not None
        self.conn.execute("DELETE FROM preprocess_profiles WHERE id=?", (profile_id,))
        self.conn.commit()

    def find_preprocessed_run_duplicate(
        self,
        segment_id: int,
        operations: Iterable[str],
        reference_segment_id: Optional[int] = None,
    ) -> Optional[Dict[str, object]]:
        if not self.is_open():
            return None
        assert self.conn is not None
        target_operations = [str(op) for op in operations]
        rows = self.conn.execute(
            """
            SELECT id, run_name, reference_segment_id, profile_json, created_at
            FROM preprocessed_runs
            WHERE segment_id = ?
            ORDER BY id DESC
            """,
            (segment_id,),
        ).fetchall()
        for row in rows:
            saved_ref = row["reference_segment_id"]
            if saved_ref is not None:
                saved_ref = int(saved_ref)
            if saved_ref != reference_segment_id:
                continue

            saved_operations: List[str] = []
            try:
                payload = json.loads(row["profile_json"] or "{}")
                saved_operations = [str(op) for op in (payload.get("operations") or [])]
            except Exception:
                saved_operations = []
            if saved_operations == target_operations:
                return {
                    "id": int(row["id"]),
                    "run_name": row["run_name"] or "",
                    "reference_segment_id": saved_ref,
                    "created_at": row["created_at"] or "",
                }
        return None

    def save_preprocessed_run(
        self,
        segment_id: int,
        run_name: str,
        profile_json: str,
        channels: Iterable[ChannelData],
        profile_id: Optional[int] = None,
        reference_segment_id: Optional[int] = None,
    ) -> int:
        assert self.conn is not None
        channels = list(channels)
        cur = self.conn.execute(
            """
            INSERT INTO preprocessed_runs(segment_id, profile_id, reference_segment_id, run_name, created_at, profile_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                segment_id,
                profile_id,
                reference_segment_id,
                run_name,
                datetime.now().isoformat(timespec="seconds"),
                profile_json,
            ),
        )
        run_id = int(cur.lastrowid)
        self.conn.executemany(
            "INSERT INTO preprocessed_channels(run_id, channel_index, channel_name) VALUES (?, ?, ?)",
            [(run_id, ch.index, ch.display_name) for ch in channels],
        )
        point_rows = []
        for ch in channels:
            point_rows.extend((run_id, ch.index, float(t), float(r)) for t, r in zip(ch.time, ch.resistance))
        self.conn.executemany(
            "INSERT INTO preprocessed_points(run_id, channel_index, time, resistance) VALUES (?, ?, ?, ?)",
            point_rows,
        )
        self.conn.commit()
        return run_id

    def list_preprocessed_runs(self) -> pd.DataFrame:
        if not self.is_open():
            return pd.DataFrame()
        assert self.conn is not None
        return pd.read_sql_query(
            """
            SELECT r.id, r.run_name, r.segment_id, s.segment_name, r.profile_id, p.profile_name, r.reference_segment_id, r.created_at
            FROM preprocessed_runs r
            LEFT JOIN segments s ON s.id = r.segment_id
            LEFT JOIN preprocess_profiles p ON p.id = r.profile_id
            ORDER BY r.id DESC
            """,
            self.conn,
        )

    def load_preprocessed_run(self, run_id: int) -> ParsedFile:
        assert self.conn is not None
        row = self.conn.execute(
            "SELECT r.*, s.segment_name FROM preprocessed_runs r LEFT JOIN segments s ON s.id=r.segment_id WHERE r.id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError("Предобработанный результат не найден.")
        ch_df = pd.read_sql_query(
            "SELECT channel_index, channel_name FROM preprocessed_channels WHERE run_id=? ORDER BY channel_index",
            self.conn,
            params=(run_id,),
        )
        pt_df = pd.read_sql_query(
            "SELECT channel_index, time, resistance FROM preprocessed_points WHERE run_id=? ORDER BY channel_index, time",
            self.conn,
            params=(run_id,),
        )
        channels: List[ChannelData] = []
        for _, ch in ch_df.iterrows():
            sub = pt_df[pt_df["channel_index"] == ch["channel_index"]]
            channels.append(
                ChannelData(
                    index=int(ch["channel_index"]),
                    original_name=str(ch["channel_name"]),
                    display_name=str(ch["channel_name"]),
                    time=sub["time"].to_numpy(dtype=float),
                    resistance=sub["resistance"].to_numpy(dtype=float),
                    color=DEFAULT_COLORS[int(ch["channel_index"]) % len(DEFAULT_COLORS)],
                )
            )
        profile_json = row["profile_json"] or ""
        metadata = {
            "segment_id": str(row["segment_id"] or ""),
            "reference_segment_id": str(row["reference_segment_id"] or ""),
            "profile_json": profile_json,
        }
        try:
            payload = json.loads(profile_json) if profile_json else {}
            metadata["algorithm"] = str(payload.get("algorithm") or payload.get("profile_name") or "")
        except Exception:
            metadata["algorithm"] = ""
        notes = ["Предобработанный результат", f"Профиль JSON: {profile_json}"]
        return ParsedFile(
            file_path=Path(f"preprocessed_run_{run_id}"),
            file_name=row["run_name"] or f"preprocessed_run_{run_id}",
            metadata_lines=[f"segment: {row['segment_name'] or ''}"] if row['segment_name'] else [],
            metadata=metadata,
            channels=channels,
            raw_row_count=sum(len(ch.time) for ch in channels),
            column_count=len(channels) * 2,
            notes=notes,
        )

    def save_feature_set(
        self,
        feature_set_name: str,
        features_long: pd.DataFrame,
        segment_id: Optional[int] = None,
        run_id: Optional[int] = None,
        profile_id: Optional[int] = None,
        source_kind: str = "segment",
        note: str = "",
    ) -> int:
        assert self.conn is not None
        cur = self.conn.execute(
            """
            INSERT INTO feature_sets(feature_set_name, segment_id, run_id, profile_id, source_kind, created_at, note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feature_set_name,
                segment_id,
                run_id,
                profile_id,
                source_kind,
                datetime.now().isoformat(timespec="seconds"),
                note,
            ),
        )
        feature_set_id = int(cur.lastrowid)
        rows = [
            (
                feature_set_id,
                int(row["channel_index"]),
                row["channel_name"],
                row["feature_key"],
                float(row["feature_value"]),
            )
            for _, row in features_long.iterrows()
        ]
        self.conn.executemany(
            "INSERT INTO feature_values(feature_set_id, channel_index, channel_name, feature_key, feature_value) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()
        return feature_set_id

    def list_feature_sets(self) -> pd.DataFrame:
        if not self.is_open():
            return pd.DataFrame()
        assert self.conn is not None
        return pd.read_sql_query(
            """
            SELECT f.id, f.feature_set_name, f.segment_id, s.segment_name, f.run_id, r.run_name, f.profile_id, p.profile_name,
                   f.source_kind, f.created_at, f.note,
                   l.gas_name, l.concentration_ppm, l.temperature_c, l.light_mode
            FROM feature_sets f
            LEFT JOIN segments s ON s.id = f.segment_id
            LEFT JOIN segment_labels l ON l.segment_id = s.id
            LEFT JOIN preprocessed_runs r ON r.id = f.run_id
            LEFT JOIN preprocess_profiles p ON p.id = f.profile_id
            ORDER BY f.id DESC
            """,
            self.conn,
        )

    def load_feature_set_values(self, feature_set_id: int) -> pd.DataFrame:
        assert self.conn is not None
        long_df = pd.read_sql_query(
            "SELECT channel_index, channel_name, feature_key, feature_value FROM feature_values WHERE feature_set_id=? ORDER BY channel_index, feature_key",
            self.conn,
            params=(feature_set_id,),
        )
        if long_df.empty:
            return long_df
        wide = long_df.pivot_table(index=["channel_index", "channel_name"], columns="feature_key", values="feature_value", aggfunc="first").reset_index()
        wide.columns.name = None
        return wide

    def load_feature_matrix(self, feature_set_ids: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        assert self.conn is not None
        if not feature_set_ids:
            return pd.DataFrame(), pd.DataFrame()
        placeholders = ",".join(["?"] * len(feature_set_ids))
        meta_df = pd.read_sql_query(
            f"""
            SELECT f.id, f.feature_set_name, f.segment_id, s.segment_name, f.run_id, r.run_name, f.profile_id, p.profile_name,
                   f.source_kind, f.created_at, f.note,
                   l.gas_name, l.concentration_ppm, l.temperature_c, l.light_mode
            FROM feature_sets f
            LEFT JOIN segments s ON s.id = f.segment_id
            LEFT JOIN segment_labels l ON l.segment_id = s.id
            LEFT JOIN preprocessed_runs r ON r.id = f.run_id
            LEFT JOIN preprocess_profiles p ON p.id = f.profile_id
            WHERE f.id IN ({placeholders})
            ORDER BY f.id
            """,
            self.conn,
            params=feature_set_ids,
        )
        long_df = pd.read_sql_query(
            f"SELECT feature_set_id, channel_name, feature_key, feature_value FROM feature_values WHERE feature_set_id IN ({placeholders})",
            self.conn,
            params=feature_set_ids,
        )
        if long_df.empty:
            return meta_df, pd.DataFrame()
        long_df["feature_name"] = long_df["channel_name"].astype(str) + "::" + long_df["feature_key"].astype(str)
        wide_df = long_df.pivot_table(index="feature_set_id", columns="feature_name", values="feature_value", aggfunc="first")
        wide_df.columns.name = None
        wide_df = wide_df.reset_index().rename(columns={"feature_set_id": "id"})
        return meta_df, wide_df
