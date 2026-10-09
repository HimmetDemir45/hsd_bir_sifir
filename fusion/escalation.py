"""Onaylanmayan kırmızı uyarı zaman aşımı (alerts.auto_escalate_red). Saf mantık: I/O ve zaman dışarıdan verilir.

İLKE (CLAUDE.md 2): polis/112 bildirimi OTOMATİK DEĞİL. Bu bileşen yalnızca "kırmızı uyarı timeout_s
içinde onaylanmadı" sonucunu üretir; çağıran taraf SADECE ek bildirim (Telegram) ve ekranda uyarı yapar.
Karar yine insandadır. enabled=false (varsayılan) iken hiçbir şey izlenmez ve hiçbir şey tetiklenmez.
"""
from __future__ import annotations

from typing import Any, Callable


class RedTimeoutWatcher:
    def __init__(self, cfg: dict[str, Any]) -> None:
        esc = cfg["alerts"]["auto_escalate_red"]
        self.enabled: bool = bool(esc.get("enabled", False))
        self.timeout_s: float = float(esc.get("timeout_s", 120))
        self._tracked: dict[str, float] = {}  # alert id -> izlemeye başlanan an
        self._fired: set[str] = set()         # her uyarı için en fazla bir kez tetiklenir

    def reset(self) -> None:
        self._tracked.clear()
        self._fired.clear()

    def track(self, alert_id: str, level: str, now: float) -> None:
        """Kırmızı uyarıyı izlemeye alır (dedup güncellemelerinde tekrar çağrılması zararsız)."""
        if not self.enabled or level != "red":
            return
        if alert_id not in self._tracked and alert_id not in self._fired:
            self._tracked[alert_id] = now

    def due(self, now: float, status_of: Callable[[str], str | None]) -> list[str]:
        """Süresi dolmuş ve hâlâ 'pending' olan uyarı id'leri. Karar verilmişse (onay / yanlış alarm) izlemeyi bırakır."""
        out: list[str] = []
        for alert_id, start in list(self._tracked.items()):
            status = status_of(alert_id)
            if status != "pending":
                del self._tracked[alert_id]
            elif now - start >= self.timeout_s:
                del self._tracked[alert_id]
                self._fired.add(alert_id)
                out.append(alert_id)
        return out
