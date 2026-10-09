"""Fusion kuralları: saf fonksiyonlar (I/O yok). Tüm eşikler config.yaml > alerts altında."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from core.schema import AlertLevel, Event

LEVEL_ORDER: dict[str, int] = {"yellow": 1, "orange": 2, "red": 3}


def max_level(a: AlertLevel | None, b: AlertLevel | None) -> AlertLevel | None:
    if a is None:
        return b
    if b is None:
        return a
    return a if LEVEL_ORDER[a] >= LEVEL_ORDER[b] else b


def _pct(conf: float) -> str:
    return f"%{round(conf * 100)}"


def classify(event: Event, cfg: dict[str, Any], crowd_size: int = 0) -> tuple[AlertLevel, str] | None:
    """Tek event'ten (seviye, Türkçe sebep) üretir. Uyarı gerektirmiyorsa None."""
    alerts = cfg["alerts"]
    t, c = event.type, event.confidence

    if t in ("gun", "knife"):
        if c >= alerts["red"]["weapon_min_conf"]:
            name = "Silah" if t == "gun" else "Bıçak"
            return "red", f"{name} tespit edildi ({_pct(c)})"
        return None
    if t == "gunshot":
        if c >= alerts["red"]["gunshot_min_conf"]:
            return "red", f"Silah sesi duyuldu ({_pct(c)})"
        return None

    if t == "fight":
        return "orange", "Kavga şüphesi"
    if t == "fall":
        return "orange", "Yere düşme tespit edildi"
    if t == "scream":
        if crowd_size >= alerts["orange"]["scream_crowd_min_people"]:
            return "orange", f"Çığlık ve kalabalık ({crowd_size} kişi)"
        return "yellow", "Çığlık duyuldu"

    if t == "shout":
        if c >= alerts["yellow"]["shout_min_conf"]:
            return "yellow", f"Bağırma sesi ({_pct(c)})"
        return None
    if t == "running":
        return "yellow", "Ani koşuşma"
    if t == "glass":
        return "yellow", "Cam kırılma sesi"
    return None


def _to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def is_break(now: datetime, cfg: dict[str, Any]) -> bool:
    """now teneffüs aralıklarından birinde mi? (schedule.breaks, HH:MM; başlangıç dahil, bitiş hariç)"""
    minutes = now.hour * 60 + now.minute
    for start, end in cfg.get("schedule", {}).get("breaks", []):
        if _to_minutes(start) <= minutes < _to_minutes(end):
            return True
    return False
