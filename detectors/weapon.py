"""Silah / bıçak tespiti (YOLO).

- models/weapon.pt varsa onu kullanır (Roboflow weapon modeli), sınıfları config'deki class_map ile
  gun/knife'a eşler.
- Yoksa fallback: COCO yolo11n.pt ile sadece "knife" sınıfı.
- Zamansal filtre: bir tip, son `window` karenin en az `min_hits`'inde conf >= min_conf ise Event üretir.

Tek başına çalıştırma (webcam):
    python -m detectors.weapon
    python -m detectors.weapon --source test_media/bicak.mp4
"""
from __future__ import annotations

import argparse
import queue
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from core.config import load_config, resolve_path
from core.schema import Event
from detectors import apply_thread_limit

COLORS = {"gun": (0, 0, 255), "knife": (0, 128, 255)}  # BGR
WEAK_COLOR = (160, 160, 160)


@dataclass
class Detection:
    type: str            # gun | knife
    label: str           # modelin kendi sınıf adı
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2


class WeaponDetector:
    def __init__(
        self,
        cfg: dict,
        camera_id: str,
        zone_id: str,
        out_queue: queue.Queue | None = None,
    ) -> None:
        self.cfg = cfg["weapon"]
        self._full_cfg = cfg
        self.device = cfg["general"]["device"]
        self.imgsz = cfg["general"]["imgsz"]
        self.camera_id = camera_id
        self.zone_id = zone_id
        self.out_queue = out_queue

        self.model, class_map, self.using_fallback = self._load_model()
        # model sınıf id -> event tipi; sadece eşlenen sınıflar tahmin edilir (daha hızlı)
        self.id_to_type: dict[int, str] = {
            cid: class_map[name.lower()]
            for cid, name in self.model.names.items()
            if name.lower() in class_map
        }
        if not self.id_to_type:
            raise RuntimeError(
                f"Modelin sınıfları ({list(self.model.names.values())}) config'deki "
                f"weapon.class_map ile eşleşmiyor."
            )

        self.min_conf: float = self.cfg["min_conf"]
        self.display_conf: float = self.cfg["display_conf"]
        self.min_hits: int = self.cfg["min_hits"]
        self.cooldown: float = self.cfg["emit_cooldown_s"]
        types = set(self.id_to_type.values())
        self.history: dict[str, deque[float]] = {t: deque(maxlen=self.cfg["window"]) for t in types}
        self.last_emit: dict[str, float] = {t: 0.0 for t in types}

    def _load_model(self) -> tuple[YOLO, dict[str, str], bool]:
        primary = resolve_path(self.cfg["model"])
        if primary.is_file():
            class_map = {k.lower(): v for k, v in self.cfg["class_map"].items()}
            return YOLO(str(primary)), class_map, False

        fallback = resolve_path(self.cfg["fallback_model"])
        fallback.parent.mkdir(parents=True, exist_ok=True)
        print(f"[weapon] {primary.name} bulunamadı -> fallback {fallback.name} (sadece bıçak)")
        class_map = {k.lower(): v for k, v in self.cfg["fallback_classes"].items()}
        return YOLO(str(fallback)), class_map, True

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.display_conf,
            classes=list(self.id_to_type),
            device=self.device,
            verbose=False,
        )[0]
        apply_thread_limit(self._full_cfg)  # ultralytics ilk tahminde thread sayısını eziyor: her tahminden sonra kontrol
        detections: list[Detection] = []
        for box in result.boxes:
            cid = int(box.cls)
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    type=self.id_to_type[cid],
                    label=self.model.names[cid],
                    confidence=float(box.conf),
                    box=(x1, y1, x2, y2),
                )
            )
        return detections

    def update(self, detections: list[Detection], now: float | None = None) -> list[Event]:
        """Kare sonuçlarını zamansal pencereye ekler, koşul sağlanan tipler için Event döndürür."""
        now = time.time() if now is None else now
        events: list[Event] = []
        for etype, hist in self.history.items():
            best = max((d.confidence for d in detections if d.type == etype), default=0.0)
            hist.append(best)
            hits = [c for c in hist if c >= self.min_conf]
            if len(hits) >= self.min_hits and now - self.last_emit[etype] >= self.cooldown:
                self.last_emit[etype] = now
                event = Event(
                    source="weapon",
                    type=etype,  # type: ignore[arg-type]
                    confidence=round(sum(hits) / len(hits), 3),
                    camera_id=self.camera_id,
                    zone_id=self.zone_id,
                    ts=now,
                    snapshot=None,  # M3: yüzü bulanıklaştırılmış snapshot (privacy/blur.py) eklenecek
                )
                events.append(event)
                if self.out_queue is not None:
                    self.out_queue.put(event)
        return events

    def process(self, frame: np.ndarray, now: float | None = None) -> tuple[list[Detection], list[Event]]:
        detections = self.detect(frame)
        return detections, self.update(detections, now)

    def draw(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        for d in detections:
            strong = d.confidence >= self.min_conf
            color = COLORS.get(d.type, (0, 0, 255)) if strong else WEAK_COLOR
            x1, y1, x2, y2 = d.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if strong else 1)
            cv2.putText(frame, f"{d.label} {d.confidence:.2f}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return frame


def main() -> None:
    from sources.camera import Camera

    cfg = load_config()
    cam_cfg = cfg["cameras"][0]

    parser = argparse.ArgumentParser(description="Silah/bıçak detektörünü tek başına çalıştır.")
    parser.add_argument("--source", default=str(cam_cfg["source"]),
                        help="0 = webcam, rtsp://... veya video dosyası")
    parser.add_argument("--device", default=None, help="cpu | cuda (config'i ezer)")
    parser.add_argument("--imgsz", type=int, default=None, help="config'i ezer")
    parser.add_argument("--no-show", action="store_true", help="pencere açma, sadece konsol")
    args = parser.parse_args()
    if args.device:
        cfg["general"]["device"] = args.device
    if args.imgsz:
        cfg["general"]["imgsz"] = args.imgsz

    detector = WeaponDetector(cfg, camera_id=cam_cfg["id"], zone_id=cam_cfg["zone_id"])
    camera = Camera(args.source)
    mode = "FALLBACK (sadece bıçak)" if detector.using_fallback else "weapon.pt"
    print(f"[weapon] model={mode} sınıflar={sorted(set(detector.id_to_type.values()))} "
          f"device={cfg['general']['device']} imgsz={cfg['general']['imgsz']}  (çıkış: q / Ctrl+C)")

    min_dt = 1.0 / cfg["general"]["target_fps"]
    fps = 0.0
    try:
        while True:
            t0 = time.time()
            frame = camera.read()
            if frame is None:
                print("[weapon] kaynak bitti / kare gelmiyor.")
                break
            detections, events = detector.process(frame, camera.last_ts)
            for ev in events:
                print(ev.to_json(), flush=True)

            dt = time.time() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6)) if fps else 1.0 / max(dt, 1e-6)
            if not args.no_show:
                detector.draw(frame, detections)
                cv2.putText(frame, f"{fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("OkulKalkan - weapon", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # işleme hızını target_fps ile sınırla (laptop CPU'sunu boşuna yakmasın)
            if dt < min_dt and not camera.is_file:  # dosya kendi hızında oynar
                time.sleep(min_dt - dt)
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
