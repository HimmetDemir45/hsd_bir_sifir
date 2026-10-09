"""Silah / bıçak tespiti (YOLO).

- config weapon.models: birden fazla model birlikte çalışır (tabanca: models/weapon_guns.pt, bıçak: COCO yolo11n).
  Her modelin sınıfları class_map ile gun/knife'a eşlenir; dosyası olmayan model uyarıyla atlanır.
- Zamansal filtre: bir tip, son `window` karenin en az `min_hits`'inde conf >= min_conf[tip] ise Event üretir.

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
from pathlib import Path

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

        # [(model, sınıf id -> event tipi)]; sadece eşlenen sınıflar tahmin edilir (daha hızlı)
        self.models: list[tuple[YOLO, dict[int, str]]] = self._load_models()
        if not self.models:
            raise RuntimeError("weapon.models içindeki hiçbir model yüklenemedi (python -m tools.modelleri_indir)")

        self.min_conf: dict[str, float] = dict(self.cfg["min_conf"])
        self.display_conf: float = self.cfg["display_conf"]
        self.min_hits: int = self.cfg["min_hits"]
        self.cooldown: float = self.cfg["emit_cooldown_s"]
        types = {t for _, id_to_type in self.models for t in id_to_type.values()}
        self.history: dict[str, deque[float]] = {t: deque(maxlen=self.cfg["window"]) for t in types}
        # -1e9 (0.0 değil): zaman 0'dan başlayan kaynakta (değerlendirme) ilk 2 sn hiç event basılmıyordu
        self.last_emit: dict[str, float] = {t: -1e9 for t in types}
        self._frame_idx = 0
        self._evaluated: set[str] | None = None          # None: update() doğrudan çağrıldı (testler) -> tüm tipler
        self._last_by_model: dict[int, list[Detection]] = {}

    def set_zone(self, zone_id: str) -> None:
        """Kaynağın bölgesi değişti (demo videosunda parça başına bölge): yeni bölgede ilk tespit beklemeden basılsın."""
        if zone_id != self.zone_id:
            self.zone_id = zone_id
            for t in self.history:
                self.history[t].clear()
                self.last_emit[t] = -1e9

    def _load_models(self) -> list[tuple[YOLO, dict[int, str]]]:
        loaded = []
        for spec in self.cfg["models"]:
            path = resolve_path(spec["path"])
            class_map = {k.lower(): v for k, v in spec["class_map"].items()}
            # COCO gibi ultralytics'in bildiği modeller dosya yoksa kendisi indirir; diğerleri atlanır
            if not path.is_file() and not path.name.startswith(("yolo11", "yolov8")):
                print(f"[weapon] UYARI: {path.name} yok -> {sorted(set(class_map.values()))} tespiti kapalı "
                      f"(python -m tools.modelleri_indir)")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            model = YOLO(str(path))
            id_to_type = {cid: class_map[name.lower()] for cid, name in model.names.items()
                          if name.lower() in class_map}
            if not id_to_type:
                print(f"[weapon] UYARI: {path.name} sınıfları {list(model.names.values())} class_map ile eşleşmiyor, atlandı")
                continue
            loaded.append((model, id_to_type))
        return loaded

    def describe(self) -> str:
        return ", ".join(f"{Path(m.ckpt_path or '').name or 'model'}->{sorted(set(t.values()))}" for m, t in self.models)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """weapon.alternate_models açıksa modeller karelere sırayla dağıtılır (kare başına tek model).
        İki model birlikte kare başına ~95 ms sürüp FPS'i 13 -> 7-11'e düşürdü ve kavga tespiti (FPS'e duyarlı)
        demo videosunda kayboldu. Her tip yine sn'de ~7 kez kontrol edilir. Çizim için diğer modelin son
        sonuçları da döner; zamansal pencereye (update) sadece bu karede çalışan modelin tipleri girer."""
        self._frame_idx += 1
        alternate = self.cfg.get("alternate_models", False) and len(self.models) > 1
        detections: list[Detection] = []
        self._evaluated = set()
        for i, (model, id_to_type) in enumerate(self.models):
            if alternate and self._frame_idx % len(self.models) != i:
                detections += self._last_by_model.get(i, [])   # sadece çizim için
                continue
            self._evaluated.update(id_to_type.values())
            result = model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.display_conf,
                classes=list(id_to_type),
                device=self.device,
                verbose=False,
            )[0]
            fresh = []
            for box in result.boxes:
                cid = int(box.cls)
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                fresh.append(Detection(type=id_to_type[cid], label=model.names[cid],
                                       confidence=float(box.conf), box=(x1, y1, x2, y2)))
            self._last_by_model[i] = fresh
            detections += fresh
        apply_thread_limit(self._full_cfg)  # ultralytics ilk tahminde thread sayısını eziyor: her tahminden sonra kontrol
        return detections

    def update(self, detections: list[Detection], now: float | None = None) -> list[Event]:
        """Kare sonuçlarını zamansal pencereye ekler, koşul sağlanan tipler için Event döndürür."""
        now = time.time() if now is None else now
        events: list[Event] = []
        for etype, hist in self.history.items():
            if self._evaluated is not None and etype not in self._evaluated:
                continue   # bu karede bu tipin modeli çalışmadı (sırayla çalıştırma)
            best = max((d.confidence for d in detections if d.type == etype), default=0.0)
            hist.append(best)
            hits = [c for c in hist if c >= self.min_conf[etype]]
            if len(hits) >= self.min_hits and now - self.last_emit[etype] >= self.cooldown:
                self.last_emit[etype] = now
                event = Event(
                    source="weapon",
                    type=etype,  # type: ignore[arg-type]
                    confidence=round(sum(hits) / len(hits), 3),
                    camera_id=self.camera_id,
                    zone_id=self.zone_id,
                    ts=now,
                    snapshot=None,  # snapshot uyarı seviyesinde main.py alır (bulanık yayın karesinden)
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
            strong = d.confidence >= self.min_conf.get(d.type, 1.01)
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
    print(f"[weapon] modeller: {detector.describe()} eşikler={detector.min_conf} "
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
