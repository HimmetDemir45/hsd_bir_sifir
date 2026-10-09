"""Mikrofon / wav dosyasından ses okuma.

Çıktı: float32 mono, [-1, 1] aralığında, istenen örnekleme hızında parçalar (chunk).
- Mikrofon: sounddevice ile arka planda okunur, parçalar kuyruğa düşer.
- Wav/flac/ogg: tamamı okunur, gerekirse yeniden örneklenir, gerçek zamanlı hızda verilir
  (Camera'daki gibi; demo dosyası canlı mikrofon gibi davranır).
`last_ts`, son okunan parçanın bitiş zamanıdır (time.time() ekseni).
"""
from __future__ import annotations

import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from core.config import resolve_path


def resample(wav: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Basit doğrusal yeniden örnekleme (ek paket gerektirmez; sınıflandırma için yeterli)."""
    if sr_from == sr_to or len(wav) == 0:
        return wav.astype(np.float32, copy=False)
    n = int(round(len(wav) * sr_to / sr_from))
    x_new = np.linspace(0, len(wav) - 1, n)
    return np.interp(x_new, np.arange(len(wav)), wav).astype(np.float32)


def list_devices() -> str:
    return str(sd.query_devices())


class AudioSource:
    def __init__(
        self,
        source: str | int | None,
        sample_rate: int,
        block_s: float = 0.1,
        loop_file: bool = False,
        realtime: bool = True,
    ) -> None:
        """source: None / "default" / "mic" = varsayılan mikrofon, int veya "3" = cihaz numarası,
        aksi halde ses dosyası yolu."""
        self.sample_rate = sample_rate
        self.block = max(1, int(sample_rate * block_s))
        self.loop_file = loop_file
        self.realtime = realtime
        self.last_ts = time.time()
        self._stream: sd.InputStream | None = None

        s = None if source is None else str(source).strip()
        self.is_file = bool(s) and s.lower() not in ("default", "mic") and not s.isdigit()
        if self.is_file:
            path = resolve_path(s)  # type: ignore[arg-type]
            if not Path(path).is_file():
                raise RuntimeError(f"Ses dosyası bulunamadı: {path}")
            wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
            self._wav = resample(wav.mean(axis=1), sr, sample_rate)
            self._pos = 0
            self._start: float | None = None
        else:
            device = int(s) if s and s.isdigit() else None
            self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
            self._stream_sr = sample_rate
            try:
                self._stream = self._open_stream(device, sample_rate)
            except Exception:
                # bazı cihazlar 32 kHz desteklemez: cihazın kendi hızında aç, yazılımla çevir
                self._stream_sr = int(sd.query_devices(device, "input")["default_samplerate"])
                self._stream = self._open_stream(device, self._stream_sr)
            self._stream.start()

    def _open_stream(self, device: int | None, sr: int) -> sd.InputStream:
        def callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
            try:
                self._q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass  # işleme yetişemiyorsa eski parçaları düşür

        return sd.InputStream(device=device, samplerate=sr, channels=1, dtype="float32",
                              blocksize=int(sr * self.block / self.sample_rate), callback=callback)

    def read(self, timeout: float = 2.0) -> np.ndarray | None:
        """Bir sonraki parçayı döndürür; dosya bittiyse None."""
        if self.is_file:
            return self._read_file()
        try:
            chunk = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        self.last_ts = time.time()
        return resample(chunk, self._stream_sr, self.sample_rate)

    def _read_file(self) -> np.ndarray | None:
        if self._pos >= len(self._wav):
            if not self.loop_file:
                return None
            self._pos, self._start = 0, None
        if self._start is None:
            self._start = time.time()
        chunk = self._wav[self._pos:self._pos + self.block]
        self._pos += len(chunk)
        end_t = self._start + self._pos / self.sample_rate
        if self.realtime:
            wait = end_t - time.time()
            if wait > 0:
                time.sleep(wait)
        self.last_ts = end_t
        return chunk

    def release(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
