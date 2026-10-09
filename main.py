"""Tüm pipeline'ı başlatır.

    python main.py --demo                                  # sahte senaryo (kamera/model gerekmez)
    python main.py                                         # config'deki ilk kamera (webcam)
    python main.py --source test_media/kavga.mp4 [--audio test_media/ciglik.wav]

Thread'ler: kamera (detektörler) / ses / fusion / API (uvicorn). Aralarında sadece kuyruk ve kilitli değişken.
"""
from __future__ import annotations

import argparse
import queue
import threading
import time

import cv2
import uvicorn

from api.server import FrameStore, Hub, _placeholder_jpeg, create_app
from core.config import load_config
from core.schema import Alert, Event
from fusion import demo
from fusion.engine import FusionEngine
from storage.db import Storage

JPEG_QUALITY = 70


def camera_loop(cfg: dict, cam_cfg: dict, source: str, events: "queue.Queue[Event]",
                frames: FrameStore, pose_holder: dict, stop: threading.Event) -> None:
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
            # M3: frame = blur_faces(frame)  (privacy/blur.py hazır olunca, JPEG'e çevirmeden ÖNCE)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                frames.set(cam_id, buf.tobytes())
            dt = time.time() - t0
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
    args = ap.parse_args()

    cfg = load_config()
    cam_cfg = cfg["cameras"][0]
    events: "queue.Queue[Event]" = queue.Queue()
    stop = threading.Event()
    storage = Storage(args.db)
    frames, hub = FrameStore(), Hub()
    pose_holder: dict = {}  # zone_id -> PoseDetector (kalabalık bilgisi için)

    def on_alert(alert: Alert) -> None:
        storage.save_alert(alert)
        latest = storage.get_alert(alert.id) or alert.to_dict()  # DB'deki güncel status ile
        hub.publish(latest)
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
                  events, frames, pose_holder, stop)))
    if args.audio and not args.demo:
        threads.append(threading.Thread(target=audio_loop, name="audio", daemon=True,
                                        args=(cfg, cam_cfg["zone_id"], args.audio, events, stop)))

    app = create_app(cfg, storage, frames, hub)
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
