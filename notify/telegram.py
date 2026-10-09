"""Telegram Bot API bildirimi (requests ile sendMessage / sendPhoto).

- telegram.enabled=false ise (veya token/chat_id boşsa) sadece konsola yazar.
- İnternet yoksa / API hata verirse loglar, pipeline'ı DURDURMAZ. Gönderim ayrı thread'de yapılır.
- Demoda tüm roller aynı chat'e gider; mesajda rol adları yazar (config: recipients).
- 112 araması GERÇEK DEĞİL: onay mesajı "simülasyon" der.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import requests

from core.config import resolve_path

ROLE_TR = {
    "nobetci": "Nöbetçi öğretmen",
    "mudur_yrd": "Müdür yardımcısı",
    "rehber": "Rehber öğretmen",
    "mudur": "Okul müdürü",
}
LEVEL_ICON = {"yellow": "🟡 SARI", "orange": "🟠 TURUNCU", "red": "🔴 KIRMIZI"}
API = "https://api.telegram.org/bot{token}/{method}"
TIMEOUT_S = 5


class Notifier:
    def __init__(self, cfg: dict[str, Any], async_send: bool = True) -> None:
        tg = cfg.get("telegram", {})
        self.token: str = tg.get("bot_token") or ""
        self.chat_id: str = str(tg.get("chat_id") or "")
        self.enabled: bool = bool(tg.get("enabled")) and bool(self.token) and bool(self.chat_id)
        if tg.get("enabled") and not self.enabled:
            print("[telegram] enabled ama .env'de token/chat_id yok -> sadece konsol.")
        self.recipients: dict[str, list[str]] = cfg.get("recipients", {})
        self.zones: dict[str, Any] = cfg.get("zones", {})
        self.async_send = async_send
        self._sent: set[tuple[str, str]] = set()  # (alert id, seviye): aynı seviyeyi iki kez bildirme
        self._lock = threading.Lock()

    # --- mesaj metinleri (saf, test edilebilir) ---
    def _zone_name(self, zone_id: str) -> str:
        return self.zones.get(zone_id, {}).get("name", zone_id)

    def format_alert(self, alert: dict[str, Any]) -> str:
        roles = ", ".join(ROLE_TR.get(r, r) for r in self.recipients.get(alert["level"], []))
        lines = [f"{LEVEL_ICON[alert['level']]} UYARI — {self._zone_name(alert['zone_id'])}"]
        lines += [f"• {r}" for r in alert["reasons"]]
        if roles:
            lines.append(f"Bildirim: {roles}")
        return "\n".join(lines)

    def format_confirmation(self, alert: dict[str, Any]) -> str:
        return (f"✅ ONAYLANDI — {self._zone_name(alert['zone_id'])}\n"
                "112 arandı (simülasyon).\n"
                "Tüm öğretmenlere 'sınıfta kalın' mesajı gönderildi.")

    # --- gönderim ---
    def send_alert(self, alert: dict[str, Any]) -> None:
        """Yeni uyarı veya seviyesi yükselmiş uyarı için bildirir; aynı seviye tekrar gitmez."""
        key = (alert["id"], alert["level"])
        with self._lock:
            if key in self._sent:
                return
            self._sent.add(key)
        self._dispatch(self.format_alert(alert), alert.get("snapshot"))

    def send_confirmation(self, alert: dict[str, Any]) -> None:
        self._dispatch(self.format_confirmation(alert), None)

    def _dispatch(self, text: str, snapshot: str | None) -> None:
        print(f"[bildirim] {text.replace(chr(10), ' | ')}", flush=True)
        if not self.enabled:
            return
        if self.async_send:
            threading.Thread(target=self._send, args=(text, snapshot), daemon=True).start()
        else:
            self._send(text, snapshot)

    def _send(self, text: str, snapshot: str | None) -> None:
        try:
            photo = resolve_path(snapshot) if snapshot else None
            if photo is not None and Path(photo).is_file():
                with open(photo, "rb") as f:
                    r = requests.post(API.format(token=self.token, method="sendPhoto"),
                                      data={"chat_id": self.chat_id, "caption": text},
                                      files={"photo": f}, timeout=TIMEOUT_S)
            else:
                r = requests.post(API.format(token=self.token, method="sendMessage"),
                                  data={"chat_id": self.chat_id, "text": text}, timeout=TIMEOUT_S)
            if not r.ok:
                print(f"[telegram] HTTP {r.status_code}: {r.text[:120]}")
        except Exception as e:  # internet yok vb.: sadece logla
            print(f"[telegram] gönderilemedi: {e!r}")
