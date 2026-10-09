"""Webcam / RTSP / video dosyasından kare okuma.

Canlı kaynaklarda (webcam, RTSP) arka planda sürekli okuyup sadece en son kareyi tutar;
böylece işleme yavaş kalsa bile gecikme birikmez.

Video dosyası varsayılan olarak gerçek zamanlı oynatılır: işleme yetişemezse kare atlanır.
Böylece demo videosu canlı kamera gibi davranır ve hız/süre kuralları doğru çalışır.
`last_ts`, okunan karenin zaman damgasıdır (saniye, time.time() ile aynı eksen).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from core.config import load_config, resolve_path


def parse_source(source: str | int) -> str | int:
    """'0' -> 0 (webcam index), göreli dosya yolu -> mutlak yol, rtsp/http olduğu gibi."""
    if isinstance(source, int):
        return source
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    if s.startswith(("rtsp://", "http://", "https://")):
        return s
    return str(resolve_path(s))


class Camera:
    def __init__(self, source: str | int, loop_file: bool = False, realtime: bool = True,
                 reconnect_s: float | None = None) -> None:
        self.source = parse_source(source)
        self.is_file = isinstance(self.source, str) and Path(self.source).is_file()
        self.loop_file = loop_file
        self.realtime = realtime
        # canlı kaynakta (webcam/RTSP) bu kadar sn kare gelmezse kamera yeniden açılır
        self.reconnect_s = (load_config()["general"]["reconnect_s"] if reconnect_s is None else reconnect_s)
        self.cap = self._open()
        if not self.cap.isOpened():
            raise RuntimeError(f"Kamera kaynağı açılamadı: {source!r}")

        self.last_ts: float = time.time()
        self._file_start: float | None = None  # dosyanın 0. saniyesine karşılık gelen duvar saati
        self.position: float = 0.0  # dosyada son okunan karenin video saniyesi (canlı kaynakta 0)
        self._frame: np.ndarray | None = None
        self._frame_ts = 0.0
        self._lock = threading.Lock()
        self._stopped = False
        self._thread: threading.Thread | None = None
        if not self.is_file:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        if isinstance(self.source, int):
            # Windows'ta DirectShow, varsayılan MSMF'den çok daha hızlı açılır
            return cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        return cv2.VideoCapture(self.source)

    def _reconnect(self, backoff: float) -> bool:
        print(f"[camera] {self.reconnect_s:.0f} sn kare yok -> yeniden bağlanılıyor: {self.source!r}", flush=True)
        self.cap.release()
        time.sleep(backoff)
        cap = self._open()
        if cap.isOpened():
            self.cap = cap
            print("[camera] yeniden bağlandı", flush=True)
            return True
        cap.release()
        return False

    def _reader(self) -> None:
        """Canlı kaynak: sürekli oku, sadece son kareyi tut. Kamera koparsa (USB çıktı, RTSP düştü)
        reconnect_s sonra yeniden aç; başarısızsa artan aralıklarla (en fazla 10 sn) tekrar dene."""
        fail_since: float | None = None
        backoff = min(1.0, self.reconnect_s)
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok:
                now = time.time()
                fail_since = fail_since or now
                if self.reconnect_s > 0 and now - fail_since >= self.reconnect_s:
                    if self._reconnect(backoff):
                        fail_since, backoff = None, min(1.0, self.reconnect_s)
                    else:
                        backoff = min(backoff * 2, 10.0)
                else:
                    time.sleep(0.05)
                continue
            fail_since = None
            with self._lock:
                self._frame = frame
                self._frame_ts = time.time()

    def _read_file(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok and self.loop_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._file_start = None
            ok, frame = self.cap.read()
        if not ok:
            return None

        pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        now = time.time()
        if self._file_start is None:
            self._file_start = now - pos
        if self.realtime:
            # geride kaldıysak kareleri çözmeden atla (grab), öndeysek bekle
            skipped = eof = False
            while pos < now - self._file_start - 0.05:
                if not self.cap.grab():
                    eof = True  # dosya bitti: elimizdeki son kareyi ver, retrieve çağırma
                    break
                skipped = True
                pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if skipped and not eof:
                ok, latest = self.cap.retrieve()
                if ok:
                    frame = latest
            wait = self._file_start + pos - time.time()
            if wait > 0:
                time.sleep(wait)
        self.position = pos
        self.last_ts = self._file_start + pos
        return frame

    def read(self, timeout: float = 5.0) -> np.ndarray | None:
        """Bir sonraki kareyi döndürür; kaynak bittiyse None."""
        if self.is_file:
            return self._read_file()

        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    frame, self._frame = self._frame, None
                    self.last_ts = self._frame_ts
                    return frame
            time.sleep(0.005)
        return None

    def release(self) -> None:
        self._stopped = True
        if self._thread:
            self._thread.join(timeout=1.0)
        self.cap.release()
