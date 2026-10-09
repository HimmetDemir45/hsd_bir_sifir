"""Tüm model dosyalarını models/ altına indirir (demo internetsiz çalışmalı: sunumdan ÖNCE bir kez çalıştırın).

    python -m tools.modelleri_indir

Zaten varsa tekrar indirmez. models/weapon.pt (Roboflow silah modeli) elle eklenir; yoksa sistem bıçak fallback'ine düşer.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from core.config import load_config, resolve_path

URLS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "yamnet.tflite":
        "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite",
}


def download(name: str, url: str, dest: Path) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  [var]   {name}")
        return True
    try:
        print(f"  [iniyor] {name} ...", end="", flush=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
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
    for key in (cfg["weapon"]["fallback_model"], cfg["pose"]["model"]):
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

    weapon = resolve_path(cfg["weapon"]["model"])
    print(f"Silah modeli: {'[var] ' + weapon.name if weapon.is_file() else '[yok] ' + weapon.name + ' -> sadece bıçak (fallback)'}")
    print("SONUÇ:", "tüm modeller hazır, demo internetsiz çalışır." if ok else "bazı modeller inmedi, yukarıdaki hatalara bakın.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
