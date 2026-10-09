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
    (1, "audio", "shout", 0.55, "cam1", "kantin"),       # sarı
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
