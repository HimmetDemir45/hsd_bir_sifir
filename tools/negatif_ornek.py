"""Silah modeli için "zor negatif" (silahsız) kareler: güvenlik kamerası / gündelik videolardan, boş etiketle.

Amaç: gerçek kamerada koyu nesneleri silah sanmayı azaltmak. Ölçüm dürüst kalsın diye Surveillance Fight
noFight kliplerinin YARISI eğitime (çift sıra), diğer yarısı yanlış alarm ölçümüne (tek sıra) ayrılır;
sunum demosuna ayrılan klipler hiç kullanılmaz.

    python -m tools.negatif_ornek --fight-root ../hsd_datasets/fight --rlvs-root ../hsd_datasets/rlvs \
        --holdout ../hsd_datasets/sonuclar/demo_holdout.json --out ../hsd_datasets/negatifler
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2


def split_nofight(fight_root: Path, holdout: set[str]) -> tuple[list[Path], list[Path]]:
    """(eğitim, ölçüm) — demo klipleri ikisinde de yok."""
    clips = sorted(p for p in (fight_root / "noFight").glob("*.mp4") if p.name not in holdout)
    return clips[0::2], clips[1::2]


def grab(path: Path, n: int) -> list:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / n))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fight-root", required=True)
    ap.add_argument("--rlvs-root", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-clip", type=int, default=6, help="güvenlik kamerası klibi başına kare")
    ap.add_argument("--rlvs-n", type=int, default=600, help="RLVS NonViolence videosu sayısı (birer kare)")
    args = ap.parse_args()

    holdout = set(json.load(open(args.holdout, encoding="utf-8")))
    train, measure = split_nofight(Path(args.fight_root), holdout)
    img_dir, lbl_dir = Path(args.out) / "images" / "train", Path(args.out) / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0

    def save(frame, name):
        nonlocal n
        cv2.imwrite(str(img_dir / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (lbl_dir / f"{name}.txt").write_text("")       # boş etiket = arka plan (silah yok)
        n += 1

    for clip in train:
        for k, f in enumerate(grab(clip, args.per_clip)):
            save(f, f"cctv_{clip.stem}_{k}")
    nonviolence = next(p for p in Path(args.rlvs_root).rglob("NonViolence") if p.is_dir())
    vids = sorted(nonviolence.glob("*.mp4"))
    for v in random.Random(0).sample(vids, min(args.rlvs_n, len(vids))):
        for f in grab(v, 1):
            save(f, f"rlvs_{v.stem}")
    (Path(args.out) / "olcum_klipleri.json").write_text(json.dumps([p.name for p in measure]), encoding="utf-8")
    print(f"{n} negatif kare -> {img_dir}; eğitim klibi {len(train)}, ölçüme ayrılan klip {len(measure)}")


if __name__ == "__main__":
    main()
