"""Detektörleri etiketli veri setlerinde değerlendirir: yakalama oranı (TPR) ve yanlış alarm oranı (FPR).

Veri setleri git'e eklenmez (lisansları ticari olmayan/araştırma); repo dışında bir klasöre indirin.

    python -m tools.degerlendir kavga --root ../hsd_datasets/fight          # fight/ + noFight/ (*.mp4)
    python -m tools.degerlendir dusme --root ../hsd_datasets/urfall         # fall-XX-cam0-rgb/ + adl-XX-cam0-rgb/ (PNG)
    python -m tools.degerlendir ses   --root ../hsd_datasets/esc50/ESC-50-master

Kareler gerçek sistemin işleme hızında (--fps, varsayılan 13) örneklenir ama zaman video saatinden alınır:
sonuç makinenin anlık yükünden bağımsız ve tekrarlanabilir. Klip başına sadece "event çıktı mı" değil,
kural sinyalinin en uzun süresi / en yüksek skoru da kaydedilir; böylece eşik taraması tek çalıştırmada yapılır.
Ham sonuçlar --out JSON dosyasına yazılır.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from core.config import load_config


# ---------------- ortak ----------------
def frames_from_video(path: Path, eval_fps: float) -> Iterator[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx, last_slot = 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = idx / fps
        slot = int(ts * eval_fps)
        if slot != last_slot:   # gerçek sistemin işleme hızına örnekle
            last_slot = slot
            yield ts, frame
        idx += 1
    cap.release()


def frames_from_images(folder: Path, eval_fps: float, src_fps: float = 30.0) -> Iterator[tuple[float, np.ndarray]]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    last_slot = -1
    for idx, p in enumerate(files):
        ts = idx / src_fps
        slot = int(ts * eval_fps)
        if slot != last_slot:
            last_slot = slot
            frame = cv2.imread(str(p))
            if frame is not None:
                yield ts, frame


def shrink(frame: np.ndarray, max_side: int) -> np.ndarray:
    """main.py ile aynı: uzun kenar max_side'dan büyükse küçült."""
    h, w = frame.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return frame
    s = max_side / max(h, w)
    return cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def rates(pos: list[float], neg: list[float], threshold: float) -> tuple[float, float]:
    tpr = sum(v >= threshold for v in pos) / len(pos) if pos else float("nan")
    fpr = sum(v >= threshold for v in neg) / len(neg) if neg else float("nan")
    return tpr, fpr


def sweep_table(title: str, pos: list[float], neg: list[float], thresholds: list[float],
                current: float, unit: str) -> str:
    lines = [f"\n**{title}** (pozitif {len(pos)}, negatif {len(neg)})\n",
             f"| eşik ({unit}) | yakalama (TPR) | yanlış alarm (FPR) |", "|---|---|---|"]
    for th in thresholds:
        tpr, fpr = rates(pos, neg, th)
        mark = " ← şu anki" if abs(th - current) < 1e-9 else ""
        lines.append(f"| {th:g}{mark} | %{tpr * 100:.0f} | %{fpr * 100:.0f} |")
    return "\n".join(lines)


def pose_clip(pose, frames: Iterator[tuple[float, np.ndarray]], max_side: int) -> dict:
    """Bir klipte poz kurallarının sinyalleri: en uzun kavga/yatay süresi, en yüksek kol/ortalama hız, event'ler."""
    pose.reset()
    fight_max = fall_max = limb_max = speed_max = 0.0
    events: list[str] = []
    n = 0
    for ts, frame in frames:
        n += 1
        tracks, evs = pose.process(shrink(frame, max_side), ts)
        events += [e.type for e in evs]
        # kuralla aynı: süre sadece kavga o karede aktifken ölçülür (gap_s içindeki boşluklarda değil)
        if pose._fight_since is not None and pose._fight_last >= ts - 1e-9:
            fight_max = max(fight_max, ts - pose._fight_since)
        require_sudden = pose.cfg["fall"].get("require_sudden", False)
        for tr in tracks:
            # kuralla aynı: yatay süre sadece (gerekiyorsa) ani düşüşle başlayan bölümde sayılır
            if tr.fallen_since is not None and tr.horizontal and (tr.fall_sudden or not require_sudden):
                fall_max = max(fall_max, ts - tr.fallen_since)
            limb_max = max(limb_max, tr.limb_speed)
            speed_max = max(speed_max, tr.mean_speed)
    return {"frames": n, "fight_max_s": round(fight_max, 2), "fall_max_s": round(fall_max, 2),
            "limb_max": round(limb_max, 2), "speed_max": round(speed_max, 2), "events": sorted(set(events))}


def make_pose(cfg):
    from detectors.pose import PoseDetector
    return PoseDetector(cfg, "eval", "eval")


# ---------------- kavga ----------------
def eval_kavga(root: Path, cfg: dict, eval_fps: float, limit: int) -> tuple[list[dict], str]:
    pose = make_pose(cfg)
    results = []
    for label, sub in (("fight", "fight"), ("normal", "noFight")):
        files = sorted((root / sub).glob("*.mp4"))[:limit or None]
        for i, f in enumerate(files, 1):
            r = pose_clip(pose, frames_from_video(f, eval_fps), cfg["general"].get("max_frame_side", 0))
            results.append({"name": f.name, "label": label, **r})
            print(f"\r  {sub}: {i}/{len(files)}", end="", flush=True)
        print()
    fc = cfg["pose"]["fight"]
    pos = [r["fight_max_s"] for r in results if r["label"] == "fight"]
    neg = [r["fight_max_s"] for r in results if r["label"] == "normal"]
    ev_pos = sum("fight" in r["events"] for r in results if r["label"] == "fight")
    ev_neg = sum("fight" in r["events"] for r in results if r["label"] == "normal")
    report = [f"### Kavga — Surveillance Camera Fight Dataset ({len(pos)} kavga + {len(neg)} normal klip, 2-3 sn)",
              f"Şu anki ayarla kavga event'i: kavga kliplerinin **%{ev_pos / len(pos) * 100:.0f}**'inde, "
              f"normal kliplerin **%{ev_neg / len(neg) * 100:.0f}**'inde (yanlış alarm).",
              sweep_table("Kavga durumunun klipteki en uzun süresi (min_duration_s taraması)", pos, neg,
                          [0.25, 0.5, 0.75, 1.0, 1.5], fc["min_duration_s"], "sn"),
              f"\nNot: klipler 2-3 sn; hız hesabı birkaç kare ısınma ister, uzun gerçek olaylarda yakalama daha yüksek beklenir."]
    return results, "\n".join(report)


# ---------------- düşme ----------------
def urfall_lying_seconds(root: Path) -> dict[str, float]:
    """UR Fall CSV'lerinden yerde yatma süresi (etiket 1 = yerde yatıyor, 30 FPS)."""
    out: dict[str, int] = defaultdict(int)
    for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        p = root / name
        if p.is_file():
            for row in csv.reader(open(p, encoding="utf-8")):
                if len(row) > 2 and row[2].strip() == "1":
                    out[row[0]] += 1
    return {k: v / 30.0 for k, v in out.items()}


def eval_dusme(root: Path, cfg: dict, eval_fps: float, limit: int) -> tuple[list[dict], str]:
    pose = make_pose(cfg)
    lying = urfall_lying_seconds(root)
    results = []
    folders = sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith("-cam0-rgb"))[:limit or None]
    for i, d in enumerate(folders, 1):
        seq = d.name.replace("-cam0-rgb", "")
        label = "fall" if seq.startswith("fall") else "adl"
        r = pose_clip(pose, frames_from_images(d, eval_fps), cfg["general"].get("max_frame_side", 0))
        results.append({"name": seq, "label": label, "gt_lying_s": round(lying.get(seq, 0.0), 2), **r})
        print(f"\r  {i}/{len(folders)}", end="", flush=True)
    print()
    fc = cfg["pose"]["fall"]
    pos = [r["fall_max_s"] for r in results if r["label"] == "fall"]
    neg = [r["fall_max_s"] for r in results if r["label"] == "adl"]
    gt = [r["gt_lying_s"] for r in results if r["label"] == "fall"]
    ev_pos = sum("fall" in r["events"] for r in results if r["label"] == "fall")
    ev_neg = sum("fall" in r["events"] for r in results if r["label"] == "adl")
    report = [f"### Düşme — UR Fall Detection ({len(pos)} düşme + {len(neg)} günlük aktivite, cam0 RGB)",
              f"Şu anki ayarla düşme event'i: düşmelerin **%{ev_pos / max(1, len(pos)) * 100:.0f}**'inde, "
              f"günlük aktivitelerin **%{ev_neg / max(1, len(neg)) * 100:.0f}**'inde (yanlış alarm).",
              f"Veri setinde düşme sonrası yerde yatma süresi (etiket): ort {np.mean(gt) if gt else 0:.1f} sn, "
              f"en kısa {min(gt) if gt else 0:.1f} sn, en uzun {max(gt) if gt else 0:.1f} sn.",
              sweep_table("Gövdenin yatay kaldığı en uzun süre (min_duration_s taraması)", pos, neg,
                          [0.5, 1.0, 1.5, 2.0, 3.0], fc["min_duration_s"], "sn")]
    return results, "\n".join(report)


# ---------------- ses ----------------
def eval_ses(root: Path, cfg: dict, limit: int) -> tuple[list[dict], str]:
    from detectors.audio_cls import EVENT_TYPES, AudioDetector
    from sources.audio import AudioSource

    det = AudioDetector(cfg)
    det.db_rule_enabled = False   # dosya seviyesi gerçek ortam gürültüsü değil
    meta = list(csv.DictReader(open(root / "meta" / "esc50.csv", encoding="utf-8")))
    if limit:   # her sınıftan eşit sayıda
        per = max(1, limit // 50)
        seen: Counter = Counter()
        meta = [m for m in meta if (seen.update([m["category"]]) or seen[m["category"]] <= per)]
    results = []
    for i, m in enumerate(meta, 1):
        det.reset()
        src = AudioSource(str(root / "audio" / m["filename"]), det.sample_rate, realtime=False)
        chunks = []
        while (c := src.read()) is not None:
            chunks.append(c)
        wav = np.concatenate(chunks)
        peaks = {t: 0.0 for t in EVENT_TYPES}
        events: list[str] = []
        for k, s in enumerate(range(0, len(wav) - det.hop + 1, det.hop)):
            events += [e.type for e in det.process_hop(wav[s:s + det.hop], 1000.0 + k * det.cfg["hop_s"])]
            for t, v in det.last_scores.items():
                peaks[t] = max(peaks[t], v)
        results.append({"name": m["filename"], "category": m["category"],
                        "peaks": {t: round(v, 3) for t, v in peaks.items()}, "events": sorted(set(events))})
        print(f"\r  {i}/{len(meta)}", end="", flush=True)
    print()

    pos_glass = [r["peaks"]["glass"] for r in results if r["category"] == "glass_breaking"]
    neg_glass = [r["peaks"]["glass"] for r in results if r["category"] != "glass_breaking"]
    th = det.min_conf
    report = [f"### Ses — ESC-50 ({len(results)} klip, 50 sınıf; db kuralı kapalı)",
              sweep_table("Cam kırılması (glass) tepe skoru", pos_glass, neg_glass,
                          [0.05, 0.08, 0.12, 0.15, 0.2, 0.3], th["glass"], "skor")]
    # ESC-50'de çığlık/silah sınıfı yok: bu tipler için tüm klipler negatif -> yanlış alarm oranı
    report.append("\n**Yanlış alarm (ESC-50'de çığlık/bağırma/silah sınıfı yok, hepsi negatif)**\n")
    report.append("| tip | eşik | yanlış alarm | en çok tetikleyen sınıflar |")
    report.append("|---|---|---|---|")
    for t in ("scream", "shout", "gunshot"):
        hits = [r for r in results if r["peaks"][t] >= th[t]]
        top = Counter(r["category"] for r in hits).most_common(4)
        report.append(f"| {t} | {th[t]} | %{len(hits) / len(results) * 100:.1f} ({len(hits)}/{len(results)}) | "
                      + ", ".join(f"{c} ({n})" for c, n in top) + " |")
    glass_fp = Counter(r["category"] for r in results
                       if r["category"] != "glass_breaking" and r["peaks"]["glass"] >= th["glass"]).most_common(4)
    report.append(f"\nCam kırılması yanlış alarmlarını en çok tetikleyen sınıflar: "
                  + (", ".join(f"{c} ({n})" for c, n in glass_fp) or "yok"))
    return results, "\n".join(report)


def main() -> None:
    ap = argparse.ArgumentParser(description="Detektörleri etiketli veri setinde değerlendir.")
    ap.add_argument("set", choices=["kavga", "dusme", "ses"])
    ap.add_argument("--root", required=True, help="veri setinin klasörü")
    ap.add_argument("--fps", type=float, default=13.0, help="görüntüde işleme hızı (gerçek sistem ~13 FPS)")
    ap.add_argument("--limit", type=int, default=0, help="klip sınırı (hızlı deneme; ses için toplam)")
    ap.add_argument("--out", default=None, help="ham sonuç JSON dosyası")
    args = ap.parse_args()

    cfg = load_config()
    root = Path(args.root)
    t0 = time.time()
    if args.set == "kavga":
        results, report = eval_kavga(root, cfg, args.fps, args.limit)
    elif args.set == "dusme":
        results, report = eval_dusme(root, cfg, args.fps, args.limit)
    else:
        results, report = eval_ses(root, cfg, args.limit)
    print(report)
    print(f"\n({time.time() - t0:.0f} sn)")
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"ham sonuç: {args.out}")


if __name__ == "__main__":
    main()
