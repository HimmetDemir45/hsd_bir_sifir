"""Poz (iskelet) tabanlı davranış tespiti: kavga, yere düşme, ani koşuşma.

İLKE: Kimlik/yüz/mimik analizi yok. Sadece 17 iskelet keypoint'i (COCO) kullanılır.

Kurallar (eşikler config.yaml > pose):
- fight:   iki kişi yakın (merkez mesafesi < kişi boyu * proximity_ratio) ve en az birinin
           bilek/dirsek hızı yüksek; durum >= min_duration_s sürerse.
- fall:    gövde yatay (omuz-kalça ekseni dikeyle > torso_angle_deg) >= min_duration_s.
- running: ortalama keypoint hızı > mean_speed olan en az min_people kişi, >= min_duration_s.
Hızlar "kişi boyu / saniye" cinsindendir; böylece kameraya uzaklıktan bağımsızdır.

Ayrıca `crowd_size` (en kalabalık kümedeki kişi sayısı) hesaplanır; "çığlık + kümelenme"
kuralı için fusion bunu kullanabilir (Event şemasında karşılığı yok, bkz. PoseDetector.crowd_size).

Tek başına çalıştırma:
    python -m detectors.pose
    python -m detectors.pose --source test_media/kavga.mp4
"""
from __future__ import annotations

import argparse
import itertools
import math
import queue
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

from core.config import load_config, resolve_path
from core.schema import Event

# COCO keypoint indeksleri
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12
LIMBS = (L_EL, R_EL, L_WR, R_WR)
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


@dataclass
class Person:
    box: tuple[int, int, int, int]
    conf: float
    kps: np.ndarray     # (17, 2) piksel
    kpc: np.ndarray     # (17,) keypoint güvenleri

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

    @property
    def scale(self) -> float:
        """Kişi boyu yaklaşığı (yere düşmüşse genişlik büyük olur, o yüzden max)."""
        x1, y1, x2, y2 = self.box
        return float(max(x2 - x1, y2 - y1, 1))


@dataclass
class Track:
    id: int
    person: Person
    last_ts: float
    limb_speed: float = 0.0
    mean_speed: float = 0.0
    last_fast_ts: float = -1e9
    fallen_since: float | None = None
    fall_emitted: bool = False
    horizontal: bool = False


@dataclass
class PoseStatus:
    """Çizim/debug için son karenin özeti."""
    fighting_ids: set[int] = field(default_factory=set)
    fallen_ids: set[int] = field(default_factory=set)
    running_count: int = 0
    crowd_size: int = 0


class PoseDetector:
    def __init__(
        self,
        cfg: dict,
        camera_id: str,
        zone_id: str,
        out_queue: queue.Queue | None = None,
    ) -> None:
        self.cfg = cfg["pose"]
        self.device = cfg["general"]["device"]
        self.imgsz = cfg["general"]["imgsz"]
        self.camera_id = camera_id
        self.zone_id = zone_id
        self.out_queue = out_queue

        model_path = resolve_path(self.cfg["model"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(model_path))  # yoksa models/ altına otomatik iner

        # Yüz bulanıklaştırma için düşük güvenli kişiler de (kısmen kapanmış, kalabalıkta) lazım
        self.privacy_conf: float = cfg.get("privacy", {}).get("person_conf", self.cfg["person_conf"])
        self.privacy_persons: list[Person] = []
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self._frame_idx = 0
        self._fight_since: float | None = None
        self._fight_last = -1e9
        self._running_since: float | None = None
        self._last_emit: dict[str, float] = {}
        self.status = PoseStatus()

    # ---------- algılama ----------
    def detect(self, frame: np.ndarray) -> list[Person]:
        """Kurallar için person_conf üstü kişileri döndürür. Gizlilik için daha düşük eşikteki
        (privacy.person_conf) tüm kişiler ayrıca self.privacy_persons'a yazılır (aynı model çağrısı)."""
        result = self.model.predict(
            frame, imgsz=self.imgsz, conf=min(self.cfg["person_conf"], self.privacy_conf),
            device=self.device, verbose=False,
        )[0]
        if result.keypoints is None or len(result.boxes) == 0:
            self.privacy_persons = []
            return []
        kps = result.keypoints.xy.cpu().numpy()
        kpc = result.keypoints.conf
        kpc = kpc.cpu().numpy() if kpc is not None else np.ones(kps.shape[:2], np.float32)
        persons = []
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            persons.append(Person((x1, y1, x2, y2), float(box.conf), kps[i], kpc[i]))
        self.privacy_persons = persons
        return [p for p in persons if p.conf >= self.cfg["person_conf"]]

    # ---------- takip + hız ----------
    def _update_tracks(self, persons: list[Person], now: float) -> list[Track]:
        max_dist = self.cfg["track_max_dist"]
        ema = self.cfg["speed_ema"]
        kp_conf = self.cfg["kp_conf"]

        candidates = []
        for tid, tr in self.tracks.items():
            for pi, p in enumerate(persons):
                d = float(np.linalg.norm(tr.person.center - p.center)) / max(tr.person.scale, p.scale)
                if d < max_dist:
                    candidates.append((d, tid, pi))
        candidates.sort()

        used_t: set[int] = set()
        used_p: set[int] = set()
        active: list[Track] = []
        for _, tid, pi in candidates:
            if tid in used_t or pi in used_p:
                continue
            used_t.add(tid)
            used_p.add(pi)
            tr, p = self.tracks[tid], persons[pi]
            dt = now - tr.last_ts
            if dt > 1e-3:
                valid = (tr.person.kpc >= kp_conf) & (p.kpc >= kp_conf)
                if valid.any():
                    disp = np.linalg.norm(p.kps - tr.person.kps, axis=1) / p.scale / dt
                    mean_v = float(disp[valid].mean())
                    limb_valid = [i for i in LIMBS if valid[i]]
                    limb_v = float(max(disp[i] for i in limb_valid)) if limb_valid else 0.0
                    tr.mean_speed = ema * tr.mean_speed + (1 - ema) * mean_v
                    tr.limb_speed = ema * tr.limb_speed + (1 - ema) * limb_v
            tr.person, tr.last_ts = p, now
            active.append(tr)

        for pi, p in enumerate(persons):
            if pi not in used_p:
                tr = Track(self._next_id, p, now)
                self.tracks[tr.id] = tr
                self._next_id += 1
                active.append(tr)

        ttl = self.cfg["track_ttl_s"]
        self.tracks = {tid: t for tid, t in self.tracks.items() if now - t.last_ts <= ttl}
        return active

    def _is_horizontal(self, p: Person) -> bool:
        kp_conf = self.cfg["kp_conf"]
        if all(p.kpc[i] >= kp_conf for i in (L_SH, R_SH, L_HIP, R_HIP)):
            sh = (p.kps[L_SH] + p.kps[R_SH]) / 2
            hip = (p.kps[L_HIP] + p.kps[R_HIP]) / 2
            dx, dy = abs(hip[0] - sh[0]), abs(hip[1] - sh[1])
            return math.degrees(math.atan2(dx, dy + 1e-6)) > self.cfg["fall"]["torso_angle_deg"]
        x1, y1, x2, y2 = p.box
        return (x2 - x1) / max(y2 - y1, 1) > self.cfg["fall"]["aspect_ratio"]

    def crowd_size(self, tracks: list[Track]) -> int:
        """En kalabalık kümedeki kişi sayısı (yarıçap = kişi boyu * radius_ratio)."""
        r = self.cfg["crowd"]["radius_ratio"]
        best = 0
        for a in tracks:
            n = sum(
                1 for b in tracks
                if np.linalg.norm(a.person.center - b.person.center) <= r * a.person.scale
            )
            best = max(best, n)
        return best

    # ---------- kurallar ----------
    def _emit(self, etype: str, conf: float, now: float) -> Event | None:
        if now - self._last_emit.get(etype, -1e9) < self.cfg["emit_cooldown_s"]:
            return None
        self._last_emit[etype] = now
        event = Event(
            source="pose",
            type=etype,  # type: ignore[arg-type]
            confidence=round(conf, 3),
            camera_id=self.camera_id,
            zone_id=self.zone_id,
            ts=now,
            snapshot=None,  # M3: yüzü bulanıklaştırılmış snapshot (privacy/blur.py) eklenecek
        )
        if self.out_queue is not None:
            self.out_queue.put(event)
        return event

    def _apply_rules(self, tracks: list[Track], now: float) -> list[Event]:
        c = self.cfg
        events: list[Event] = []
        status = PoseStatus()

        for tr in tracks:
            if tr.limb_speed >= c["fight"]["limb_speed"]:
                tr.last_fast_ts = now

        # kavga: yakın iki kişi + en az birinde yakın zamanda hızlı kol hareketi.
        # Süre çift bazında değil sahne bazında tutulur: hızlı hareket ve düşük FPS'te takip numaraları
        # değişiyor, çift bazında sayaç sürekli sıfırlanıyordu (Kavga.mp4 testinde 26 sn'de tek event).
        fight_ids: set[int] = set()
        fight_confs: list[float] = []
        for a, b in itertools.combinations(tracks, 2):
            dist = float(np.linalg.norm(a.person.center - b.person.center))
            close = dist < c["fight"]["proximity_ratio"] * (a.person.scale + b.person.scale) / 2
            fast = now - max(a.last_fast_ts, b.last_fast_ts) <= c["fight"]["hold_s"]
            if close and fast:
                fight_ids.update((a.id, b.id))
                fight_confs.append((a.person.conf + b.person.conf) / 2)
        if fight_ids:
            self._fight_last = now
            if self._fight_since is None:
                self._fight_since = now
        elif self._fight_since is not None and now - self._fight_last > c["fight"]["gap_s"]:
            self._fight_since = None
        if fight_ids and now - self._fight_since >= c["fight"]["min_duration_s"]:
            status.fighting_ids = fight_ids
            ev = self._emit("fight", max(fight_confs), now)
            if ev:
                events.append(ev)

        # düşme: gövde yatay, min süre; her düşme bölümünde bir kez
        for tr in tracks:
            tr.horizontal = self._is_horizontal(tr.person)
            if not tr.horizontal:
                tr.fallen_since, tr.fall_emitted = None, False
                continue
            if tr.fallen_since is None:
                tr.fallen_since = now
            if now - tr.fallen_since >= c["fall"]["min_duration_s"]:
                status.fallen_ids.add(tr.id)
                if not tr.fall_emitted:
                    ev = self._emit("fall", tr.person.conf, now)
                    if ev:
                        tr.fall_emitted = True
                        events.append(ev)

        # koşuşma: yeterli sayıda hızlı hareket eden kişi
        runners = [t for t in tracks if t.mean_speed >= c["running"]["mean_speed"]]
        status.running_count = len(runners)
        if len(runners) >= c["running"]["min_people"]:
            if self._running_since is None:
                self._running_since = now
            if now - self._running_since >= c["running"]["min_duration_s"]:
                ev = self._emit("running", sum(t.person.conf for t in runners) / len(runners), now)
                if ev:
                    events.append(ev)
        else:
            self._running_since = None

        status.crowd_size = self.crowd_size(tracks)
        self.status = status
        return events

    def process(self, frame: np.ndarray, now: float | None = None) -> tuple[list[Track], list[Event]]:
        """Her every_n_frames karede bir model çalışır; aradaki karelerde son sonuç döner."""
        now = time.time() if now is None else now
        self._frame_idx += 1
        if (self._frame_idx - 1) % self.cfg["every_n_frames"] != 0:
            return [t for t in self.tracks.values() if t.last_ts >= now - 0.5], []
        tracks = self._update_tracks(self.detect(frame), now)
        return tracks, self._apply_rules(tracks, now)

    # ---------- çizim ----------
    def draw(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        kp_conf = self.cfg["kp_conf"]
        st = self.status
        for tr in tracks:
            p = tr.person
            if tr.id in st.fighting_ids:
                color = (0, 0, 255)
            elif tr.id in st.fallen_ids:
                color = (0, 128, 255)
            elif tr.mean_speed >= self.cfg["running"]["mean_speed"]:
                color = (0, 255, 255)
            else:
                color = (0, 200, 0)
            for i, j in SKELETON:
                if p.kpc[i] >= kp_conf and p.kpc[j] >= kp_conf:
                    cv2.line(frame, tuple(map(int, p.kps[i])), tuple(map(int, p.kps[j])), color, 2)
            x1, y1, x2, y2 = p.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            label = f"#{tr.id} v={tr.mean_speed:.1f} kol={tr.limb_speed:.1f}" + (" YATAY" if tr.horizontal else "")
            cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        info = f"kisi={len(tracks)} kume={st.crowd_size} kosan={st.running_count}"
        cv2.putText(frame, info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return frame


def main() -> None:
    from sources.camera import Camera

    cfg = load_config()
    cam_cfg = cfg["cameras"][0]

    parser = argparse.ArgumentParser(description="Poz detektörünü (kavga/düşme/koşuşma) tek başına çalıştır.")
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

    detector = PoseDetector(cfg, camera_id=cam_cfg["id"], zone_id=cam_cfg["zone_id"])
    camera = Camera(args.source)
    print(f"[pose] device={cfg['general']['device']} imgsz={cfg['general']['imgsz']} "
          f"her {cfg['pose']['every_n_frames']} karede bir  (çıkış: q / Ctrl+C)")

    min_dt = 1.0 / cfg["general"]["target_fps"]
    fps = 0.0
    try:
        while True:
            t0 = time.time()
            frame = camera.read()
            if frame is None:
                print("[pose] kaynak bitti / kare gelmiyor.")
                break
            tracks, events = detector.process(frame, camera.last_ts)
            for ev in events:
                print(ev.to_json(), flush=True)

            dt = time.time() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6)) if fps else 1.0 / max(dt, 1e-6)
            if not args.no_show:
                detector.draw(frame, tracks)
                cv2.putText(frame, f"{fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("OkulKalkan - pose", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if dt < min_dt and not camera.is_file:  # dosya kendi hızında oynar
                time.sleep(min_dt - dt)
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
