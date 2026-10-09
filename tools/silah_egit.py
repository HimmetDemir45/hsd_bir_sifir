"""Silah/bıçak YOLO11n eğitimi (GPU, ayrı eğitim ortamında: ../hsd_egitim/.venv, CUDA PyTorch).

Veri: Dangerous Items (Zenodo 16422779, CC BY 4.0), sınıflar 0 machete 1 knife 2 baseball_bat 3 rifle 4 gun
(sıra örnek görsellerle doğrulandı). data.yaml: path=<veri>/Dangerous Items, train/val/test: images/<bölüm>.
v1 (2026-10-09, yolo11n.pt'den 40 tur, RTX 4060 ~40 dk): test mAP50 0.871 -> ml/silah_v1.pt
v2: v1'den ince ayar + zor negatifler (tools/negatif_ornek.py; data.yaml'da train listesine eklenir).

    ../hsd_egitim/.venv/Scripts/python tools/silah_egit.py --name silah_v1 --base yolo11n.pt --epochs 40
    ../hsd_egitim/.venv/Scripts/python tools/silah_egit.py --name silah_v2 --base ml/silah_v1.pt \
        --data ../hsd_datasets/dangerous/data_hn.yaml --epochs 20

Windows: DataLoader alt süreçleri için __main__ koruması şart (yoksa "worker exited unexpectedly").
"""
import argparse

from ultralytics import YOLO

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="silah_v1")
    ap.add_argument("--base", default="yolo11n.pt", help="başlangıç ağırlıkları (COCO nano ya da önceki modelimiz)")
    ap.add_argument("--data", default="C:/Users/himmo/Desktop/e/hsd_datasets/dangerous/data.yaml")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--project", default="C:/Users/himmo/Desktop/e/hsd_egitim/runs")
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(data=args.data, epochs=args.epochs, imgsz=640, batch=16, device=0, workers=args.workers,
                patience=10, project=args.project, name=args.name, exist_ok=True, plots=True, verbose=False)
    metrics = model.val(data=args.data, split="test", plots=True,
                        project=args.project, name=f"{args.name}_test", exist_ok=True)   # ayrılmış test seti
    print("TEST mAP50:", round(float(metrics.box.map50), 3), "mAP50-95:", round(float(metrics.box.map), 3))
    for i, n in model.names.items():
        print(f"  {n}: P={metrics.box.p[i]:.3f} R={metrics.box.r[i]:.3f} mAP50={metrics.box.ap50[i]:.3f}")
