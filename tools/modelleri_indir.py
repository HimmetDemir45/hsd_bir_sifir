"""Tüm model dosyalarını models/ altına indirir (demo internetsiz çalışmalı: sunumdan ÖNCE bir kez çalıştırın).

    python -m tools.modelleri_indir

Zaten varsa tekrar indirmez; tabanca modeli SHA-256 ile doğrulanır.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

from core.config import load_config, resolve_path

URLS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "yamnet.tflite":
        "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite",
    # Tabanca modeli (MIT): JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8, runs/detect/Normal/weights/best.pt
    "weapon_guns.pt":
        "https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8/raw/main/runs/detect/Normal/weights/best.pt",
}
# .pt dosyaları pickle: yüklenince kod çalıştırabilir. Bu dosyanın içi 2026-10-09'da statik tarandı (sadece torch,
# ultralytics, collections import ediyor); farklı bir dosya inerse kullanılmaz.
SHA256 = {
    "weapon_guns.pt": "ede8bf51003edc7ed1c4274c2f8ec5bcd8843a7f5f300ce740a86f477cbbf4f2",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(name: str, url: str, dest: Path) -> bool:
    expected = SHA256.get(name)
    if dest.is_file() and dest.stat().st_size > 0:
        if expected and sha256(dest) != expected:
            print(f"  [HATA]  {name}: SHA-256 beklenenden farklı, dosyayı silip tekrar indirin")
            return False
        print(f"  [var]   {name}")
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        print(f"  [iniyor] {name} ...", end="", flush=True)
        urllib.request.urlretrieve(url, tmp)
        if expected and sha256(tmp) != expected:
            tmp.unlink()
            print(" HATA: SHA-256 uyuşmuyor (dosya değişmiş), kullanılmadı")
            return False
        tmp.replace(dest)
        print(f" {dest.stat().st_size // 1024} KB")
        return True
    except Exception as e:  # noqa: BLE001
        print(f" HATA: {e}")
        return False


def main() -> int:
    cfg = load_config()
    models = resolve_path("models")
    models.mkdir(exist_ok=True)
    ok = True

    print("Dosya modeller:")
    for name, url in URLS.items():
        ok &= download(name, url, models / name)

    print("YOLO (ultralytics kendisi indirir):")
    from ultralytics import YOLO
    coco = [m["path"] for m in cfg["weapon"]["models"] if Path(m["path"]).name.startswith(("yolo11", "yolov8"))]
    for key in (*coco, cfg["pose"]["model"]):
        try:
            YOLO(str(resolve_path(key)))
            print(f"  [tamam] {Path(key).name}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  HATA {Path(key).name}: {e}")

    print("Ses modeli (EfficientAT, ilk yüklemede iner):")
    try:
        from detectors.audio_cls import AudioDetector
        det = AudioDetector(cfg, backend="efficientat")
        print(f"  [tamam] {det.backend.name} ({cfg['audio']['efficientat']['model_name']})")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  HATA: {e}")

    print("SONUÇ:", "tüm modeller hazır, demo internetsiz çalışır." if ok else "bazı modeller inmedi, yukarıdaki hatalara bakın.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
