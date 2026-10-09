"""Silah/bıçak YOLO11n eğitimi (GPU, ayrı eğitim ortamında: ../hsd_egitim/.venv, CUDA PyTorch).

Veri: Dangerous Items (Zenodo 16422779, CC BY 4.0), sınıflar 0 machete 1 knife 2 baseball_bat 3 rifle 4 gun
(sıra örnek görsellerle doğrulandı). data.yaml: path=<veri>/Dangerous Items, train/val/test: images/<bölüm>.
Sonuç (2026-10-09, 40 tur, RTX 4060 ~40 dk): test mAP50 0.871; çıktı best.pt -> ml/silah_v1.pt

    ../hsd_egitim/.venv/Scripts/python tools/silah_egit.py silah_v1 40 2
"""
