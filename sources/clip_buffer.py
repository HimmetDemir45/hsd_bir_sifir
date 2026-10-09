"""Olay klibi: son kareleri bellekte tutar, uyarı gelince olayın öncesi + sonrasını videoya yazar.

Gizlilik: add()'e yalnızca yüzleri BULANIKLAŞTIRILMIŞ kareler verilir (main.py blur_faces'ten sonra çağırır).
Klip yalnızca turuncu/kırmızı uyarıda kaydedilir (karar main.py'de, config privacy.record_clips_for).

- Kareler JPEG olarak halka tamponda tutulur (ham kareyle 10 sn ~1 GB, JPEG ile ~20 MB).
- save() uyarı anında ayrı thread'de çağrılır: olaydan sonraki kısmın gelmesini bekler, sonra yazar.
- Kodek: VP8/.webm (Chrome'da oynar, bu OpenCV'de sorunsuz) -> H.264/.mp4 -> mp4v/.mp4 (yedek).
- Video FPS'i gerçek kare zamanlarından hesaplanır; klip gerçek hızında oynar.

Kullanım (main.py):
    buf = ClipBuffer(10)                 # config alerts.orange.clip_seconds
    buf.add(frame, ts)                   # her bulanık karede
    path = buf.save("clips/<id>.mp4")    # dönen yol gerçek uzantıyı içerir (.webm olabilir)
"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from core.config import load_config

# (fourcc, uzantı): sırayla denenir
CODECS = [("VP80", ".webm"), ("avc1", ".mp4"), ("mp4v", ".mp4")]
JPEG_QUALITY = 80


class ClipBuffer:
    def __init__(self, seconds: float, pre_ratio: float | None = None, max_side: int | None = None) -> None:
        if pre_ratio is None or max_side is None:
            general = load_config()["general"]
            pre_ratio = general["clip_pre_ratio"] if pre_ratio is None else pre_ratio
            max_side = general["clip_max_side"] if max_side is None else max_side
        self.seconds = float(seconds)
        self.pre = self.seconds * pre_ratio
        self.post = self.seconds - self.pre
        self.max_side = int(max_side)
        self._frames: deque[tuple[float, bytes]] = deque()  # (zaman, JPEG)
        self._cond = threading.Condition()

    def add(self, frame: np.ndarray, ts: float) -> None:
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2))
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        with self._cond:
            self._frames.append((ts, jpeg.tobytes()))
            # bekleyen save()'ler için olay öncesi + sonrası + pay kadar geçmiş tut
            while self._frames and self._frames[0][0] < ts - self.seconds - 2.0:
                self._frames.popleft()
            self._cond.notify_all()

    def save(self, path: str, event_ts: float | None = None) -> str:
        """[olay - pre, olay + post] aralığını yazar, gerçek dosya yolunu döndürür.
        Olaydan sonraki kareleri bekler (en fazla post + 3 sn; kaynak kesilirse elde olanla yazar)."""
        event_ts = time.time() if event_ts is None else event_ts
        start, end = event_ts - self.pre, event_ts + self.post
        with self._cond:
            self._cond.wait_for(lambda: bool(self._frames) and self._frames[-1][0] >= end,
                                timeout=self.post + 3.0)
            frames = [(t, j) for t, j in self._frames if start <= t <= end]
        if len(frames) < 2:
            raise RuntimeError(f"klip için yeterli kare yok ({len(frames)})")
        return self._write(path, frames)

    @staticmethod
    def _write(path: str, frames: list[tuple[float, bytes]]) -> str:
        first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
        h, w = first.shape[:2]
        duration = frames[-1][0] - frames[0][0]
        fps = float(np.clip((len(frames) - 1) / duration, 1.0, 30.0)) if duration > 0 else 10.0
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        for fourcc, ext in CODECS:
            out = str(Path(path).with_suffix(ext))
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if not writer.isOpened():
                writer.release()
                continue
            try:
                for _, jpeg in frames:
                    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if img.shape[:2] != (h, w):
                        img = cv2.resize(img, (w, h))
                    writer.write(img)
            finally:
                writer.release()
            if Path(out).is_file() and Path(out).stat().st_size > 0:
                return out
        raise RuntimeError("hiçbir video kodeki açılamadı")
