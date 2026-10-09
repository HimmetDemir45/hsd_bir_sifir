"""FastAPI: REST + WebSocket + MJPEG. main.py ayrı thread'de uvicorn ile çalıştırır.

Thread'ler arası paylaşım: FrameStore (kilitli son JPEG) ve Hub (fusion thread'inden asyncio'ya köprü).
"""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core.config import ROOT, resolve_path
from storage.db import Storage

WEB_DIR = ROOT / "web"
WEEK_S = 7 * 24 * 3600


class FrameStore:
    """camera_id -> son overlay'li JPEG (kamera thread'i yazar, MJPEG okur)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, tuple[int, bytes]] = {}
        self._seq = 0

    def set(self, camera_id: str, jpeg: bytes) -> None:
        with self._lock:
            self._seq += 1
            self._frames[camera_id] = (self._seq, jpeg)

    def get(self, camera_id: str) -> tuple[int, bytes] | None:
        with self._lock:
            return self._frames.get(camera_id)


class Hub:
    """Fusion thread'inden WebSocket istemcilerine alert iletir."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[asyncio.Queue] = set()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, alert: dict[str, Any]) -> None:
        """Alert iletir. Herhangi bir thread'den çağrılabilir."""
        self.publish_message({"type": "alert", "alert": alert})

    def publish_message(self, msg: dict[str, Any]) -> None:
        """Ham WebSocket mesajı (ör. {"type": "reset"})."""
        if self._loop is None:
            return
        for q in list(self._clients):
            self._loop.call_soon_threadsafe(q.put_nowait, msg)


def _placeholder_jpeg(text: str) -> bytes:
    img = np.full((360, 640, 3), 30, np.uint8)
    cv2.putText(img, text, (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


def create_app(cfg: dict[str, Any], storage: Storage, frames: FrameStore, hub: Hub,
               notifier: Any | None = None, demo_reset: Callable[[], None] | None = None,
               level_source: Callable[[], dict | None] | None = None) -> FastAPI:
    app = FastAPI(title="OkulKalkan")
    waiting = _placeholder_jpeg("Kamera bekleniyor...")

    @app.on_event("startup")
    async def _bind() -> None:
        hub.bind_loop(asyncio.get_running_loop())

    @app.middleware("http")
    async def _no_cache(request, call_next):
        # Geliştirme sırasında tarayıcı eski app.js/style.css'i önbellekten çalıştırmasın
        response = await call_next(request)
        if not request.url.path.startswith(("/video/", "/snapshots/")):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/alerts")
    def alerts(limit: int = 100) -> list[dict]:
        return storage.list_alerts(limit)

    @app.get("/api/status")
    def status() -> dict:
        return {"demo": demo_reset is not None}

    @app.get("/api/level")
    def level() -> dict:
        """Canlı ses seviyesi. Ses dedektörü yoksa (--demo, --audio verilmedi) available=false."""
        data = level_source() if level_source else None
        return {"available": False} if data is None else {"available": True, **data}

    @app.post("/api/demo/reset")
    def reset_demo() -> dict:
        # Sadece --demo modunda: gerçek kamera/ses çalışırken kayıtlar ASLA silinmez.
        if demo_reset is None:
            raise HTTPException(404, "Demo modu kapalı")
        demo_reset()
        return {"ok": True}

    def _set_status(alert_id: str, status: str) -> dict:
        # Not: "confirmed" sadece kayıttır; gerçek 112 araması YOK (simülasyon, bkz. CLAUDE.md).
        before = storage.get_alert(alert_id)
        if before is None:
            raise HTTPException(404, "Uyarı bulunamadı")
        alert = storage.update_alert_status(alert_id, status)
        hub.publish(alert)
        # "112 arandı (simülasyon)" mesajı SADECE kırmızıda; turuncu/sarıdaki "Gördüm" bildirim göndermez
        if notifier and status == "confirmed" and before["status"] != "confirmed" and alert["level"] == "red":
            notifier.send_confirmation(alert)  # Telegram'a "112 arandı (simülasyon)" + sınıfta kalın
        return alert

    @app.post("/api/alerts/{alert_id}/confirm")
    def confirm(alert_id: str) -> dict:
        return _set_status(alert_id, "confirmed")

    @app.post("/api/alerts/{alert_id}/dismiss")
    def dismiss(alert_id: str) -> dict:
        return _set_status(alert_id, "dismissed")

    @app.get("/api/summary/weekly")
    def weekly() -> dict:
        return storage.weekly_summary(time.time() - WEEK_S)

    @app.get("/api/summary/false-alarms")
    def false_alarms() -> dict:
        return storage.false_alarm_stats(time.time() - WEEK_S)

    @app.get("/api/heatmap")
    def heatmap() -> dict:
        return storage.heatmap()

    @app.get("/api/zones")
    def zones() -> dict:
        return cfg["zones"]

    @app.get("/snapshots/{name}")
    def snapshot(name: str) -> FileResponse:
        path = resolve_path(cfg["general"]["snapshot_dir"]) / Path(name).name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/clips/{name}")
    def clip(name: str) -> FileResponse:
        # Sadece dosya adı: yol gezintisi (../) engellenir. Uzantı mp4 veya webm olabilir (ClipBuffer.save).
        path = resolve_path(cfg["general"]["clip_dir"]) / Path(name).name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        q: asyncio.Queue = asyncio.Queue()
        hub._clients.add(q)
        try:
            while True:
                msg = await q.get()
                await socket.send_json(msg)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub._clients.discard(q)

    @app.get("/video/{camera_id}")
    async def video(camera_id: str) -> StreamingResponse:
        async def gen():
            last, last_sent = -1, 0.0
            while True:
                item = frames.get(camera_id)
                if item is None:
                    seq, jpeg = 0, waiting
                else:
                    seq, jpeg = item
                # Chrome bir kareyi ancak sonraki kare gelince çizer: görüntü değişmese de 1 sn'de bir tekrar gönder
                if seq != last or time.time() - last_sent > 1.0:
                    last, last_sent = seq, time.time()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(0.05)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    app.mount("/", StaticFiles(directory=WEB_DIR), name="web")  # app.js, style.css (en sonda)
    return app
