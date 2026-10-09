"""SQLite olay deposu (storage/events.db). Thread'ler arası kullanım: tek bağlantı + kilit."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from core.config import resolve_path
from core.schema import Alert, Event

DEFAULT_PATH = "storage/events.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT, type TEXT, confidence REAL,
    camera_id TEXT, zone_id TEXT, ts REAL, snapshot TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    level TEXT, reasons TEXT, zone_id TEXT,
    created_at REAL, status TEXT, snapshot TEXT, clip TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_events_zone ON events(zone_id);
"""


def _row_to_alert(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["reasons"] = json.loads(d["reasons"])
    return d


class Storage:
    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        if str(path) != ":memory:":
            path = resolve_path(path)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    def _exec(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
            self._db.commit()
        return rows

    def clear(self) -> None:
        """Tüm olay ve uyarıları siler (sadece demo yeniden başlatmada çağrılır)."""
        self._exec("DELETE FROM alerts")
        self._exec("DELETE FROM events")

    def save_event(self, e: Event) -> None:
        self._exec(
            "INSERT INTO events (source, type, confidence, camera_id, zone_id, ts, snapshot) "
            "VALUES (?,?,?,?,?,?,?)",
            (e.source, e.type, e.confidence, e.camera_id, e.zone_id, e.ts, e.snapshot),
        )

    def save_alert(self, a: Alert) -> None:
        """Ekler; aynı id varsa günceller (dedup/seviye yükselmesi aynı id ile gelir). status'a dokunmaz."""
        self._exec(
            "INSERT INTO alerts (id, level, reasons, zone_id, created_at, status, snapshot, clip) "
            "VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET level=excluded.level, reasons=excluded.reasons, "
            "snapshot=excluded.snapshot, clip=excluded.clip",
            (a.id, a.level, json.dumps(a.reasons, ensure_ascii=False), a.zone_id,
             a.created_at, a.status, a.snapshot, a.clip),
        )

    def update_alert_status(self, alert_id: str, status: str) -> dict[str, Any] | None:
        self._exec("UPDATE alerts SET status=? WHERE id=?", (status, alert_id))
        return self.get_alert(alert_id)

    def update_alert_clip(self, alert_id: str, clip: str) -> None:
        self._exec("UPDATE alerts SET clip=? WHERE id=?", (clip, alert_id))

    def append_reason(self, alert_id: str, reason: str) -> dict[str, Any] | None:
        """Mevcut uyarının sebeplerine metin ekler (aynı metin zaten varsa eklemez)."""
        alert = self.get_alert(alert_id)
        if alert is None:
            return None
        if reason not in alert["reasons"]:
            alert["reasons"].append(reason)
            self._exec("UPDATE alerts SET reasons=? WHERE id=?",
                       (json.dumps(alert["reasons"], ensure_ascii=False), alert_id))
        return alert

    def get_alert(self, alert_id: str) -> dict[str, Any] | None:
        rows = self._exec("SELECT * FROM alerts WHERE id=?", (alert_id,))
        return _row_to_alert(rows[0]) if rows else None

    def list_alerts(self, limit: int = 100) -> list[dict[str, Any]]:
        """En yeni en başta."""
        rows = self._exec("SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,))
        return [_row_to_alert(r) for r in rows]

    def weekly_summary(self, since_ts: float | None = None) -> dict[str, dict[str, int]]:
        """bölge -> {yellow, orange, red}. since_ts verilmezse hepsi."""
        rows = self._exec(
            "SELECT zone_id, level, COUNT(*) AS n FROM alerts WHERE created_at >= ? GROUP BY zone_id, level",
            (since_ts or 0.0,),
        )
        out: dict[str, dict[str, int]] = {}
        for r in rows:
            out.setdefault(r["zone_id"], {"yellow": 0, "orange": 0, "red": 0})[r["level"]] = r["n"]
        return out

    def heatmap(self) -> dict[str, int]:
        """bölge -> olay sayısı."""
        rows = self._exec("SELECT zone_id, COUNT(*) AS n FROM events GROUP BY zone_id")
        return {r["zone_id"]: r["n"] for r in rows}
