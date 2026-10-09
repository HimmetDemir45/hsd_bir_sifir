"""Sunum için veri seti demo videosu: eğitimde HİÇ görülmemiş kavga/normal klipler + silah test görüntüleri.

    python -m tools.veri_demo --fight-root ../hsd_datasets/fight --holdout ../hsd_datasets/sonuclar/demo_holdout.json \
        --weapon-root "../hsd_datasets/dangerous/Dangerous Items" --out test_media/veri_demo.mp4

Her parçanın üstünde gerçek etiket yazar (jüri doğruyu ekranda görür). Parçalar arası 1 sn siyah: model iki klibi
karıştırmasın. Silah görüntüleri veri setinin TEST bölümünden (eğitimde ve erken durdurmada kullanılmadı).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 960, 720, 15


def fit(frame: np.ndarray) -> np.ndarray:
    """Oranı koruyarak W x H içine sığdır (siyah kenarlı)."""
    h, w = frame.shape[:2]
    s = min(W / w, H / h)
    img = cv2.resize(frame, (int(w * s), int(h * s)))
    out = np.zeros((H, W, 3), np.uint8)
    y, x = (H - img.shape[0]) // 2, (W - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    return out


def caption(frame: np.ndarray, text: str, color: tuple) -> np.ndarray:
    cv2.rectangle(frame, (0, 0), (W, 44), (20, 20, 20), -1)
    cv2.putText(frame, text, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return frame


def clip_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if int(i * FPS / src_fps) != int((i - 1) * FPS / src_fps) or i == 0:
            frames.append(f)
        i += 1
    cap.release()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fight-root", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--weapon-root", required=True)
    ap.add_argument("--out", default="test_media/veri_demo.mp4")
    ap.add_argument("--image-s", type=float, default=3.0)
    args = ap.parse_args()

    held = json.load(open(args.holdout, encoding="utf-8"))
    fights = [n for n in held if n.startswith("fi")]
    normals = [n for n in held if n.startswith("nofi")]
    order = normals[:3] + fights[:3] + normals[3:] + fights[3:]
    root = Path(args.fight_root)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    black = np.zeros((H, W, 3), np.uint8)
    timeline = []
    t = 0.0

    def put(frames, text, color, label):
        nonlocal t
        start = t
        for f in frames:
            writer.write(caption(fit(f), text, color))
            t += 1 / FPS
        timeline.append((round(start, 1), round(t, 1), label))
        for _ in range(FPS):          # 1 sn siyah ayraç
            writer.write(black)
            t += 1 / FPS

    for name in order:
        sub = "fight" if name.startswith("fi") else "noFight"
        is_fight = sub == "fight"
        text = f"VERI SETI (egitimde gorulmedi) | Gercek: {'KAVGA' if is_fight else 'NORMAL'}"
        put(clip_frames(root / sub / name), text, (80, 80, 255) if is_fight else (120, 220, 120),
            ("kavga" if is_fight else "normal") + f" {name}")

    # silah: TEST bölümünden tek nesneli tabanca (4) / bıçak (1) görüntüleri
    wroot = Path(args.weapon_root)
    rng = random.Random(7)
    for cls, label, n in ((4, "TABANCA", 4), (1, "BICAK", 3)):
        cands = []
        for lf in sorted((wroot / "labels" / "test").glob("*.txt")):
            rows = [r.split() for r in open(lf) if r.strip()]
            if len(rows) == 1 and int(rows[0][0]) == cls:
                img = next((p for p in (wroot / "images" / "test").glob(lf.stem + ".*")), None)
                if img:
                    cands.append(img)
        for img in rng.sample(cands, min(n, len(cands))):
            frame = cv2.imread(str(img))
            put([frame] * int(args.image_s * FPS), f"VERI SETI TEST GORUNTUSU | Gercek: {label}", (80, 80, 255),
                f"{label.lower()} {img.name}")
    writer.release()
    Path(args.out).with_suffix(".json").write_text(json.dumps(timeline, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{args.out}: {t:.1f} sn, {len(timeline)} parça")
    for a, b, lab in timeline:
        print(f"  {a:5.1f}-{b:5.1f} sn  {lab}")


if __name__ == "__main__":
    main()
