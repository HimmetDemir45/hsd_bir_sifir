"""Ortak veri sözleşmesi (bkz. CLAUDE.md > Veri şemaları).

DONDURULDU: Bu dosya iki kişinin arasındaki sözleşmedir. Değiştirmeden önce diğer kişiye haber verin.
- Event: detektörler (Kişi A) -> fusion (Kişi B), queue.Queue üzerinden
- Alert: fusion -> API / DB / Telegram
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Literal

Source = Literal["weapon", "pose", "audio"]
EventType = Literal["gun", "knife", "fight", "fall", "running", "scream", "shout", "gunshot", "glass"]
AlertLevel = Literal["yellow", "orange", "red"]
AlertStatus = Literal["pending", "confirmed", "dismissed"]


@dataclass
class Event:
    source: Source
    type: EventType
    confidence: float
    camera_id: str
    zone_id: str
    ts: float = field(default_factory=time.time)
    snapshot: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class Alert:
    level: AlertLevel
    reasons: list[str]
    zone_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    status: AlertStatus = "pending"
    snapshot: str | None = None
    clip: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
