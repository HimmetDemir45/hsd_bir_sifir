"""Videolardan kavga modeli için iskelet dizileri çıkarır (bir kez; sonra eğitim saniyeler sürer).

Canlı sistemle aynı işleme: kare 960 px'e küçültülür, ~14 FPS örneklenir, PoseDetector'ın takibi kullanılır
(poz her karede; eğitimde canlı hıza alt-örneklenir). Çıktı: pickle, klip başına {name, label, frames}.

    python -m tools.kavga_veri fight --root ../hsd_datasets/fight --out ../hsd_datasets/iskelet/fight.pkl
    python -m tools.kavga_veri rlvs  --root ../hsd_datasets/rlvs  --out ../hsd_datasets/iskelet/rlvs.pkl --device cuda
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

from core.config import load_config
from tools.degerlendir import frames_from_video, shrink

# veri seti -> [(alt klasör, etiket)]; 1 = kavga / şiddet
LAYOUTS = {
    "fight": [("fight", 1), ("noFight", 0)],
    "rlvs": [("Violence", 1), ("NonViolence", 0)],      # Kaggle Real Life Violence Situations
    "movies": [("fights", 1), ("noFights", 0)],         # Movies Fights (Nievas vd. 2011, "Peliculas")
}
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg")   # Movies setinin normal sahneleri .mpg


def extract(path: Path, pose, eval_fps: float, max_side: int) -> list:
    pose.reset()
    frames = []
    for ts, frame in frames_from_video(path, eval_fps):
        tracks, _ = pose.process(shrink(frame, max_side), ts)
        frames.append((round(ts, 3), [(t.id, t.person.kps.copy(), t.person.kpc.copy(), tuple(t.person.box))
                                       for t in tracks if abs(t.last_ts - ts) < 1e-6]))
    return frames


def find_videos(root: Path, sub: str) -> list[Path]:
    d = next((p for p in root.rglob(sub) if p.is_dir()), None)
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in VIDEO_EXT) if d else []


def main() -> None:
    from detectors.pose import PoseDetector

    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=[*LAYOUTS, "video"], help="video: --root tek dosya (etiket eğitimde zaman aralığıyla)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=14.0)
    ap.add_argument("--device", default=None, help="cpu | cuda (config'i ezer)")
    ap.add_argument("--limit", type=int, default=0, help="her sınıftan en fazla N klip")
    args = ap.parse_args()

    cfg = load_config()
    if args.device:
        cfg["general"]["device"] = args.device
    cfg["pose"]["every_n_frames"] = 1          # her karede poz; canlı hıza eğitimde alt-örneklenir
    pose = PoseDetector(cfg, "egitim", "egitim")
    max_side = int(cfg["general"].get("max_frame_side", 0))
    clips, t0 = [], time.time()
    if args.set == "video":
        v = Path(args.root)
        clips.append({"name": v.name, "label": -1, "frames": extract(v, pose, args.fps, max_side)})
    for sub, label in LAYOUTS.get(args.set, []):
        vids = find_videos(Path(args.root), sub)[: args.limit or None]
        print(f"{sub}: {len(vids)} video", flush=True)
        for i, v in enumerate(vids, 1):
            try:
                clips.append({"name": v.name, "label": label, "frames": extract(v, pose, args.fps, max_side)})
            except Exception as e:  # noqa: BLE001  bozuk video tüm çıkarımı durdurmasın
                print(f"  atlandı {v.name}: {e}")
            if i % 25 == 0:
                print(f"  {sub} {i}/{len(vids)} ({time.time() - t0:.0f} sn)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(clips, f)
    print(f"kaydedildi: {args.out} ({len(clips)} klip, {time.time() - t0:.0f} sn)")


if __name__ == "__main__":
    main()
