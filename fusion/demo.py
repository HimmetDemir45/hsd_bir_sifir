"""Sahte event üreteci (demo senaryosu). Gerçek detektörler olmadan fusion hattını besler.

Kullanım: python -m fusion.demo [--speed 5]   (speed: zamanı hızlandırır, 1 = gerçek zaman)
"""
from __future__ import annotations

import argparse
import queue
import threading
import time

from core.config import load_config
from core.schema import Event
from fusion.engine import FusionEngine

# (bir önceki adımdan sonra bekleme sn, source, type, confidence, kamera, bölge)
SCENARIO: list[tuple[float, str, str, float, str, str]] = [
    # Başta iki küçük, birbirinden bağımsız sarı olay: ısı haritasında başka bölgeleri de gösterir (eskalasyon yok)
    (1, "pose", "running", 0.65, "cam2", "koridor_1"),   # sarı, koridor
    (3, "audio", "shout", 0.60, "cam3", "sinif_1a"),     # sarı, sınıf
    # Ana hikâye: Kantin
    (3, "audio", "shout", 0.55, "cam1", "kantin"),       # sarı
    (6, "pose", "running", 0.70, "cam1", "kantin"),      # sarı
    (6, "audio", "glass", 0.60, "cam1", "kantin"),       # 3. sarı -> eskalasyonla turuncu
    (8, "pose", "fight", 0.80, "cam1", "kantin"),        # turuncu
    (4, "audio", "scream", 0.75, "cam1", "kantin"),
    (6, "weapon", "knife", 0.82, "cam1", "kantin"),      # kırmızı
    (3, "weapon", "knife", 0.88, "cam1", "kantin"),      # dedup: aynı uyarıya eklenir
]


def feed(out_queue: "queue.Queue[Event]", stop_event: threading.Event, speed: float = 1.0) -> None:
    """Senaryoyu out_queue'ya basar (main.py --demo da bunu thread'de çağırır)."""
    for delay, source, etype, conf, cam, zone in SCENARIO:
        if stop_event.wait(delay / speed):
            return
        out_queue.put(Event(source=source, type=etype, confidence=conf,  # type: ignore[arg-type]
                            camera_id=cam, zone_id=zone, ts=time.time()))


class DemoController:
    """Senaryoyu başlatır / durdurur / baştan başlatır (dashboard'daki "demoyu yeniden başlat" için)."""

    def __init__(self, out_queue: "queue.Queue[Event]", speed: float = 1.0) -> None:
        self.out_queue = out_queue
        self.speed = speed
        self._lock = threading.Lock()
        self._run_stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            self._stop_locked()
            self._run_stop = threading.Event()
            self._thread = threading.Thread(target=feed, args=(self.out_queue, self._run_stop, self.speed),
                                            name="demo", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._run_stop is not None:
            self._run_stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._run_stop = self._thread = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=1.0, help="zaman hızlandırma çarpanı")
    args = ap.parse_args()

    cfg = load_config()
    q: "queue.Queue[Event]" = queue.Queue()
    stop = threading.Event()

    def show(alert) -> None:
        print(f"[{alert.level.upper():6}] {alert.to_json()}", flush=True)

    engine = FusionEngine(cfg, q, show)
    t = threading.Thread(target=engine.run, args=(stop,), daemon=True)
    t.start()
    feed(q, stop, args.speed)
    while not q.empty():
        time.sleep(0.1)
    time.sleep(0.5)
    stop.set()


if __name__ == "__main__":
    main()
