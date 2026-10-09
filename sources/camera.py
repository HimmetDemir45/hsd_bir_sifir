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

from core.config import resolve_path


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
    def __init__(self, source: str | int, loop_file: bool = False, realtime: bool = True) -> None:
        self.source = parse_source(source)
        self.is_file = isinstance(self.source, str) and Path(self.source).is_file()
        self.loop_file = loop_file
        self.realtime = realtime
        if isinstance(self.source, int):
            # Windows'ta DirectShow, varsayılan MSMF'den çok daha hızlı açılır
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Kamera kaynağı açılamadı: {source!r}")

        self.last_ts: float = time.time()
        self._file_start: float | None = None  # dosyanın 0. saniyesine karşılık gelen duvar saati
        self._frame: np.ndarray | None = None
        self._frame_ts = 0.0
        self._lock = threading.Lock()
        self._stopped = False
        self._thread: threading.Thread | None = None
        if not self.is_file:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _reader(self) -> None:
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
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
