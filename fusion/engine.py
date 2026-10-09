"""Fusion motoru: event -> alert, dedup ve eskalasyon.

Zaman her zaman event.ts'ten gelir (time.sleep / time.time yok) -> test edilebilir.
Dedup'ta mevcut uyarı güncellenirse on_alert AYNI id ile tekrar çağrılır (API/DB "güncelle" olarak ele alır).
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Callable

from core.schema import Alert, Event
from fusion.rules import LEVEL_ORDER, classify, max_level


class FusionEngine:
    def __init__(
        self,
        cfg: dict[str, Any],
        in_queue: "queue.Queue[Event]",
        on_alert: Callable[[Alert], None],
        crowd_size_by_zone: Callable[[], dict[str, int]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.in_queue = in_queue
        self.on_alert = on_alert
        self.crowd_size_by_zone = crowd_size_by_zone  # (a)/(b) kararı gelene kadar dışarıdan verilir
        self._dedup: dict[tuple[str, str], tuple[Alert, float]] = {}  # (zone, type) -> (alert, son event ts)
        self._yellows: dict[str, list[float]] = {}  # zone -> sarı uyarıların oluşma zamanları

    # --- tek event işleme (testler doğrudan bunu çağırır) ---
    def process(self, event: Event, crowd_size: int | None = None) -> list[Alert]:
        """Event'i işler; yeni veya güncellenen uyarıları döndürür (aynı zamanda on_alert'e de iletir)."""
        if crowd_size is None:
            crowd_size = (self.crowd_size_by_zone() if self.crowd_size_by_zone else {}).get(event.zone_id, 0)
        result = classify(event, self.cfg, crowd_size)
        if result is None:
            return []
        level, reason = result
        out: list[Alert] = []
        window = self.cfg["alerts"]["dedup_window_s"]
        key = (event.zone_id, event.type)

        existing = self._dedup.get(key)
        if existing and event.ts - existing[1] <= window:
            alert = existing[0]
            changed = False
            if reason not in alert.reasons:
                alert.reasons.append(reason)
                changed = True
            if LEVEL_ORDER[level] > LEVEL_ORDER[alert.level]:
                alert.level = level
                changed = True
            if event.snapshot and not alert.snapshot:
                alert.snapshot = event.snapshot
                changed = True
            self._dedup[key] = (alert, event.ts)
            if changed:
                out.append(alert)
        else:
            alert = Alert(level=level, reasons=[reason], zone_id=event.zone_id,
                          created_at=event.ts, snapshot=event.snapshot)
            self._dedup[key] = (alert, event.ts)
            out.append(alert)
            if level == "yellow":
                esc = self._check_escalation(event)
                if esc:
                    out.append(esc)

        for a in out:
            self.on_alert(a)
        return out

    def _check_escalation(self, event: Event) -> Alert | None:
        """Aynı bölgede window_s içinde yellow_count sarı uyarı -> turuncu."""
        esc = self.cfg["alerts"]["escalation"]
        times = [t for t in self._yellows.get(event.zone_id, []) if event.ts - t <= esc["window_s"]]
        times.append(event.ts)
        if len(times) < esc["yellow_count"]:
            self._yellows[event.zone_id] = times
            return None
        self._yellows[event.zone_id] = []  # sayaç sıfırlanır, aynı sarılar tekrar turuncu üretmesin
        return Alert(
            level="orange",
            reasons=[f"{len(times)} sarı uyarı {esc['window_s']} sn içinde (eskalasyon)"],
            zone_id=event.zone_id,
            created_at=event.ts,
            snapshot=event.snapshot,
        )

    def run(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                event = self.in_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.process(event)
            except Exception as e:  # tek bozuk event pipeline'ı durdurmasın
                print(f"[fusion] event işlenemedi: {e!r}")
