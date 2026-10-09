"""Tüm pipeline'ı başlatır.

    python main.py --demo                                  # sahte senaryo (kamera/model gerekmez)
    python main.py                                         # config'deki ilk kamera (webcam)
    python main.py --source test_media/kavga.mp4 [--audio test_media/ciglik.wav]

Thread'ler: kamera (detektörler) / ses / fusion / API (uvicorn). Aralarında sadece kuyruk ve kilitli değişken.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

# Windows'ta çıktı dosyaya/bazı terminallere yönlenince cp1254 kullanılır; emoji/özel karakter
# print'i UnicodeEncodeError ile thread'leri (ör. fusion) öldürüyordu. Her zaman UTF-8, basılamayan karakter '?'.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import cv2
import uvicorn

from api.server import FrameStore, Hub, _placeholder_jpeg, create_app
from core.config import load_config, resolve_path
from core.schema import Alert, Event
from fusion import demo
from fusion.engine import FusionEngine
from notify.telegram import Notifier
from storage.db import Storage

try:  # Kişi A'nın privacy/blur.py'si; hazır değilse None
    from privacy.blur import blur_faces
except ImportError:
    blur_faces = None

try:  # Kişi A'nın sources/clip_buffer.py'si; hazır değilse klip kaydı yok
    from sources.clip_buffer import ClipBuffer
except ImportError:
    ClipBuffer = None

JPEG_QUALITY = 70


def camera_loop(cfg: dict, cam_cfg: dict, source: str, events: "queue.Queue[Event]",
                frames: FrameStore, pose_holder: dict, stop: threading.Event,
                clip_buf: "ClipBuffer | None" = None) -> None:
    # Ağır importlar sadece gerçek kamera modunda (--demo bunlara bağımlı olmasın)
    from detectors.pose import PoseDetector
    from detectors.weapon import WeaponDetector
    from sources.camera import Camera

    cam_id, zone_id = cam_cfg["id"], cam_cfg["zone_id"]
    weapon = WeaponDetector(cfg, cam_id, zone_id, events)
    pose = PoseDetector(cfg, cam_id, zone_id, events)
    pose_holder[zone_id] = pose
    camera = Camera(source, loop_file=True)
    min_dt = 1.0 / cfg["general"]["target_fps"]
    n_frames, fps_t0 = 0, time.time()
    try:
        while not stop.is_set():
            t0 = time.time()
            frame = camera.read()
            if frame is None:
                print("[camera] kare gelmiyor, tekrar deneniyor...")
                time.sleep(0.5)
                continue
            w_dets, _ = weapon.process(frame, camera.last_ts)
            tracks, _ = pose.process(frame, camera.last_ts)
            weapon.draw(frame, w_dets)
            pose.draw(frame, tracks)
            if blur_faces is not None:
                # pose.privacy_persons: düşük güvenli kişiler dahil (kalabalıkta yarım görünen yüzler)
                frame = blur_faces(frame, pose.privacy_persons)  # yayın ve snapshot'tan ÖNCE (gizlilik ilkesi)
            if clip_buf is not None:
                clip_buf.add(frame, camera.last_ts)  # sadece bulanık kareler
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                frames.set(cam_id, buf.tobytes())
            dt = time.time() - t0
            n_frames += 1
            if t0 - fps_t0 >= 10.0:  # işleme hızı: düşükse hız/süre kuralları (kavga vb.) kaçırılabilir
                print(f"[camera] işleme hızı: {n_frames / (t0 - fps_t0):.1f} FPS", flush=True)
                n_frames, fps_t0 = 0, t0
            if dt < min_dt and not camera.is_file:
                time.sleep(min_dt - dt)
    finally:
        camera.release()


def audio_loop(cfg: dict, zone_id: str, audio_src: str, events: "queue.Queue[Event]",
               stop: threading.Event) -> None:
    try:
        from detectors.audio_cls import AudioDetector
    except ImportError:
        print("[audio] detectors.audio_cls henüz yok (Kişi A'nın işi); ses atlanıyor.")
        return
    try:
        # "mic" -> config'deki audio.input (mikrofon); aksi halde wav dosyası (gerçek zamanlı oynatılır)
        source = None if audio_src == "mic" else audio_src
        AudioDetector(cfg, zone_id, events).run(stop, source=source)
    except Exception as e:  # ses bozulsa görüntü hattı çalışmaya devam etsin
        print(f"[audio] durdu: {e!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="OkulKalkan")
    ap.add_argument("--source", default=None, help="0 = webcam, rtsp://..., video dosyası (config'i ezer)")
    ap.add_argument("--audio", nargs="?", const="mic", default=None, help="'mic' veya wav dosyası")
    ap.add_argument("--demo", action="store_true", help="kamera yerine hazır sahte senaryo")
    ap.add_argument("--speed", type=float, default=1.0, help="--demo zaman hızlandırma")
    ap.add_argument("--db", default="storage/events.db")
    ap.add_argument("--fresh", action="store_true", help="başlamadan önce DB'yi sil (temiz demo)")
    ap.add_argument("--port", type=int, default=None, help="config api.port'u ezer (8000 doluysa)")
    args = ap.parse_args()

    cfg = load_config()
    if args.port:
        cfg["api"]["port"] = args.port
    cam_cfg = cfg["cameras"][0]
    events: "queue.Queue[Event]" = queue.Queue()
    stop = threading.Event()
    if args.fresh:
        db_path = resolve_path(args.db)
        if db_path.is_file():
            db_path.unlink()
            print(f"[db] {db_path.name} silindi (--fresh)")
    storage = Storage(args.db)
    frames, hub = FrameStore(), Hub()
    pose_holder: dict = {}  # zone_id -> PoseDetector (kalabalık bilgisi için)

    notifier = Notifier(cfg)
    snap_levels = set(cfg["privacy"]["record_clips_for"])  # snapshot/klip sadece turuncu-kırmızıda
    snap_dir = resolve_path(cfg["general"]["snapshot_dir"])
    # Gerçek kamerada yüz bulanıklaştırma yoksa snapshot ALMA (gizlilik); demo karesi sahte, sorun yok.
    snapshots_allowed = args.demo or blur_faces is not None
    if not snapshots_allowed:
        print("[privacy] privacy/blur.py yok -> snapshot alınmıyor, yayın bulanıklaştırılmıyor (A'dan bekleniyor).")

    def take_snapshot(alert: Alert) -> None:
        if alert.snapshot or alert.level not in snap_levels or not snapshots_allowed:
            return
        item = frames.get(cam_cfg["id"])  # yayındaki kare zaten bulanık
        if item is None:
            return
        snap_dir.mkdir(parents=True, exist_ok=True)
        path = snap_dir / f"{alert.id}.jpg"
        path.write_bytes(item[1])
        alert.snapshot = f"{cfg['general']['snapshot_dir']}/{path.name}"

    # Klip: yalnızca bulanıklaştırma varken ve turuncu/kırmızıda (gizlilik ilkesi)
    clip_buf = None
    if ClipBuffer is not None and blur_faces is not None and not args.demo:
        clip_buf = ClipBuffer(cfg["alerts"]["orange"]["clip_seconds"])
    elif not args.demo:
        print("[clip] ClipBuffer ve/veya blur_faces yok -> klip kaydı kapalı (A'dan bekleniyor).")
    clip_dir = resolve_path(cfg["general"]["clip_dir"])
    clipped: set[str] = set()

    def save_clip(alert_id: str) -> None:
        try:
            clip_dir.mkdir(parents=True, exist_ok=True)
            path = clip_buf.save(str(clip_dir / f"{alert_id}.mp4"))
            storage.update_alert_clip(alert_id, path)
            latest = storage.get_alert(alert_id)
            if latest:
                hub.publish(latest)
        except Exception as e:  # klip hatası uyarı akışını bozmasın
            print(f"[clip] kaydedilemedi: {e!r}")

    def on_alert(alert: Alert) -> None:
        take_snapshot(alert)
        storage.save_alert(alert)
        if clip_buf is not None and alert.level in snap_levels and alert.id not in clipped:
            clipped.add(alert.id)
            threading.Thread(target=save_clip, args=(alert.id,), daemon=True).start()
        latest = storage.get_alert(alert.id) or alert.to_dict()  # DB'deki güncel status ile
        hub.publish(latest)
        notifier.send_alert(latest)
        print(f"[ALERT {alert.level.upper()}] {alert.zone_id}: {'; '.join(alert.reasons)}", flush=True)

    def crowd() -> dict[str, int]:
        return {z: p.status.crowd_size for z, p in pose_holder.items()}

    engine = FusionEngine(cfg, events, on_alert, crowd_size_by_zone=crowd, on_event=storage.save_event)

    threads = [threading.Thread(target=engine.run, args=(stop,), name="fusion", daemon=True)]
    if args.demo:
        frames.set(cam_cfg["id"], _placeholder_jpeg("DEMO modu (sahte olaylar)"))
        threads.append(threading.Thread(target=demo.feed, args=(events, stop, args.speed),
                                        name="demo", daemon=True))
    else:
        threads.append(threading.Thread(
            target=camera_loop, name="camera", daemon=True,
            args=(cfg, cam_cfg, args.source if args.source is not None else str(cam_cfg["source"]),
                  events, frames, pose_holder, stop, clip_buf)))
    if args.audio and not args.demo:
        threads.append(threading.Thread(target=audio_loop, name="audio", daemon=True,
                                        args=(cfg, cam_cfg["zone_id"], args.audio, events, stop)))

    app = create_app(cfg, storage, frames, hub, notifier)
    server = uvicorn.Server(uvicorn.Config(app, host=cfg["api"]["host"], port=cfg["api"]["port"],
                                           log_level="warning"))
    threads.append(threading.Thread(target=server.run, name="api", daemon=True))

    for t in threads:
        t.start()
    print(f"Dashboard: http://localhost:{cfg['api']['port']}   (çıkış: Ctrl+C)")
    try:
        while not stop.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.should_exit = True
        time.sleep(0.5)


if __name__ == "__main__":
    main()
