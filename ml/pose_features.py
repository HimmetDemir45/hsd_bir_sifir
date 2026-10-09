"""Kavga modeli için iskelet penceresi -> özellik vektörü. EĞİTİM ve CANLI sistem bu tek kodu kullanır.

İLKE: yalnız iskelet keypoint'leri ve kişi kutuları (yüz/kimlik yok).

Girdi: bir zaman penceresindeki pose kareleri. Her kare: (zaman, [kişi, ...]); kişi = (takip_id, kps(17,2),
kpc(17,), kutu(x1,y1,x2,y2)). Kişiler kareler arasında takip numarasıyla eşlenir (PoseDetector'ın takibi).

Kare başına sinyaller (iki kişi etkileşimi tek kişi hızından çok daha ayırt edici):
- en yakın çiftin merkez mesafesi (boy oranı), mesafenin değişim hızı (yaklaşma), kutu çakışması (IoU)
- bileğin karşı kişinin kutusuna uzaklığı (vurma/itme temas göstergesi)
- kişi başına kol (bilek/dirsek) hızı, ortalama nokta hızı, ivme, gövde açısı
Pencere boyunca bunların ortalama / maksimum / std / eşik üstü oranı özetlenir.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12
LIMBS = (L_EL, R_EL, L_WR, R_WR)
WRISTS = (L_WR, R_WR)
KP_CONF = 0.4
FAR = 5.0          # çift yoksa mesafe yerine konan değer (boy oranı)

Person = tuple  # (track_id, kps(17,2), kpc(17,), box(4,))


def _scale(box) -> float:
    x1, y1, x2, y2 = box
    return float(max(x2 - x1, y2 - y1, 1.0))


def _center(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _point_box_dist(p, box) -> float:
    dx = max(box[0] - p[0], 0.0, p[0] - box[2])
    dy = max(box[1] - p[1], 0.0, p[1] - box[3])
    return math.hypot(dx, dy)


def _torso_angle(kps, kpc) -> float:
    if all(kpc[i] >= KP_CONF for i in (L_SH, R_SH, L_HIP, R_HIP)):
        sh = (kps[L_SH] + kps[R_SH]) / 2
        hip = (kps[L_HIP] + kps[R_HIP]) / 2
        return math.degrees(math.atan2(abs(hip[0] - sh[0]), abs(hip[1] - sh[1]) + 1e-6))
    return 0.0


def frame_signals(frames: list[tuple[float, list[Person]]]) -> list[dict]:
    """Ardışık kareler için kare başına sinyaller (ilk kare hız için referanstır)."""
    out: list[dict] = []
    prev_by_id: dict = {}
    prev_t = None
    prev_speed: dict = {}
    prev_pair_d = None
    for t, persons in frames:
        dt = (t - prev_t) if prev_t is not None else None
        limb, mean_v, acc, angles = [], [], [], []
        for tid, kps, kpc, box in persons:
            s = _scale(box)
            angles.append(_torso_angle(kps, kpc))
            if dt and dt > 1e-3 and tid in prev_by_id:
                pk, pc = prev_by_id[tid]
                valid = (kpc >= KP_CONF) & (pc >= KP_CONF)
                if valid.any():
                    disp = np.linalg.norm(kps - pk, axis=1) / s / dt
                    mv = float(disp[valid].mean())
                    lv = [disp[i] for i in LIMBS if valid[i]]
                    lmax = float(max(lv)) if lv else 0.0
                    mean_v.append(mv)
                    limb.append(lmax)
                    acc.append(abs(lmax - prev_speed.get(tid, lmax)) / dt)
                    prev_speed[tid] = lmax
        # en yakın çift (boy oranı)
        pair_d, iou, wrist_d = FAR, 0.0, FAR
        for (ia, ka, ca, ba), (ib, kb, cb, bb) in itertools.combinations(persons, 2):
            d = float(np.linalg.norm(_center(ba) - _center(bb))) / ((_scale(ba) + _scale(bb)) / 2)
            if d < pair_d:
                pair_d = d
                iou = _iou(ba, bb)
                wd = FAR
                for kp_s, kc_s, box_o, s_o in ((ka, ca, bb, _scale(bb)), (kb, cb, ba, _scale(ba))):
                    for w in WRISTS:
                        if kc_s[w] >= KP_CONF:
                            wd = min(wd, _point_box_dist(kp_s[w], box_o) / s_o)
                wrist_d = wd
        approach = 0.0
        if dt and dt > 1e-3 and prev_pair_d is not None and pair_d < FAR and prev_pair_d < FAR:
            approach = (prev_pair_d - pair_d) / dt   # pozitif: yaklaşıyorlar
        out.append({
            "n": len(persons),
            "pair_d": pair_d, "iou": iou, "wrist_d": wrist_d, "approach": approach,
            "limb_max": max(limb, default=0.0), "speed_max": max(mean_v, default=0.0),
            "speed_mean": float(np.mean(mean_v)) if mean_v else 0.0,
            "acc_max": max(acc, default=0.0), "angle_max": max(angles, default=0.0),
        })
        prev_by_id = {tid: (kps, kpc) for tid, kps, kpc, _ in persons}
        prev_t, prev_pair_d = t, pair_d
    return out


SIGNALS = ["n", "pair_d", "iou", "wrist_d", "approach", "limb_max", "speed_max", "speed_mean", "acc_max", "angle_max"]
# eşik üstü oranlar (eşikler kurallardan esinli; model ağırlığı kendi öğrenir)
FRACTIONS = [("pair_d", "lt", 0.6), ("wrist_d", "lt", 0.05), ("limb_max", "gt", 1.5), ("iou", "gt", 0.1), ("n", "ge", 2)]


def window_features(frames: list[tuple[float, list[Person]]]) -> np.ndarray:
    """Pencere -> sabit uzunlukta özellik vektörü (feature_names() sırasıyla)."""
    sig = frame_signals(frames)[1:] or frame_signals(frames)
    feats: list[float] = []
    for k in SIGNALS:
        v = np.array([s[k] for s in sig], dtype=np.float64)
        feats += [float(v.mean()), float(v.max()), float(v.min()), float(v.std())]
    for k, op, th in FRACTIONS:
        v = np.array([s[k] for s in sig], dtype=np.float64)
        frac = (v < th) if op == "lt" else (v > th) if op == "gt" else (v >= th)
        feats.append(float(frac.mean()))
    # yakın ve hızlı aynı anda (kuralın sürekli versiyonu)
    both = [s["pair_d"] < 0.6 and s["limb_max"] > 1.5 for s in sig]
    feats.append(float(np.mean(both)))
    return np.array(feats, dtype=np.float32)


def feature_names() -> list[str]:
    names = [f"{k}_{a}" for k in SIGNALS for a in ("mean", "max", "min", "std")]
    names += [f"frac_{k}_{op}_{th}" for k, op, th in FRACTIONS]
    names.append("frac_close_and_fast")
    return names
