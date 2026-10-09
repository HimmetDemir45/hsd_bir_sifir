"""Kavga modelini iskelet pencerelerinden eğitir, klip bazlı çapraz doğrulamayla ölçer, en iyisini kaydeder.

    python -m tools.kavga_egit --data ../hsd_datasets/iskelet/fight.pkl [--data ...rlvs.pkl] --out models/kavga_model.joblib

- Pencere: window_s sn, hop_s kayma; canlı poz hızına (pose_hz) alt-örneklenir (2 farklı başlangıç = çoğaltma).
- Çapraz doğrulama KLİP bazında (StratifiedGroupKFold): aynı klibin pencereleri hem eğitimde hem testte olmaz.
- Klip kararı: pencere olasılıklarının en yükseği. Metrikler klip bazında; eşik hedef yanlış alarm oranına göre seçilir.
- Gerçek dünya testi: --test-video ile canli.mp4 / Kavga.mp4 iskeletleri (eğitimde KULLANILMAZ), bölüm etiketleriyle.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.pose_features import feature_names, window_features

WINDOW_S, HOP_S, POSE_HZ = 1.5, 0.5, 7.0

# Gerçek dünya test videolarının bölüm etiketleri (sn): 1 = kavga/itişme, 0 = kavga değil
TEST_SEGMENTS = {
    "canli.pkl": [("bıçak gösterme (yakın, el hareketi)", 0, 48, 0), ("koşma", 50, 64, 0),
                  ("boğuşma", 64, 73, 1), ("itişme / vurma", 73, 147, 1), ("kendi düşme", 147, 165, 0),
                  ("iterek düşürme", 165, 186, 1)],
    "kavga_demo.pkl": [("sakin sınıf", 0, 7, 0), ("sopayla saldırı / arbede", 8, 26.5, 1)],
}


def subsample(frames: list, hz: float, phase: int) -> list:
    """Kareleri canlı poz hızına indirir (her k. kare, phase başlangıcıyla)."""
    if len(frames) < 2:
        return frames
    src_hz = (len(frames) - 1) / max(frames[-1][0] - frames[0][0], 1e-6)
    k = max(1, round(src_hz / hz))
    return frames[phase % k::k]


def windows(frames: list, window_s: float = WINDOW_S, hop_s: float = HOP_S) -> list[tuple[float, list]]:
    """(pencere başlangıcı, kareler) listesi. Kısa klipte tek pencere."""
    if not frames:
        return []
    t0, t_end = frames[0][0], frames[-1][0]
    if t_end - t0 <= window_s:
        return [(t0, frames)]
    out, s = [], t0
    while s + window_s <= t_end + 1e-6:
        w = [f for f in frames if s <= f[0] < s + window_s]
        if len(w) >= 3:
            out.append((s, w))
        s += hop_s
    return out


def build(clips: list, phases: tuple = (0, 1)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, g = [], [], []
    for ci, c in enumerate(clips):
        for ph in phases:
            for _, w in windows(subsample(c["frames"], POSE_HZ, ph)):
                X.append(window_features(w))
                y.append(c["label"])
                g.append(ci)
    return np.array(X), np.array(y), np.array(g)


def models() -> dict:
    return {
        "lojistik_regresyon": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                                n_jobs=4, random_state=0),
        "gradient_boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=300,
                                                            l2_regularization=1.0, random_state=0),
    }


def clip_scores(model, clips: list, idx: np.ndarray) -> np.ndarray:
    """Klip başına en yüksek pencere olasılığı (tek faz: canlıdaki gibi)."""
    out = []
    for ci in idx:
        ws = windows(subsample(clips[ci]["frames"], POSE_HZ, 0))
        if not ws:
            out.append(0.0)
            continue
        p = model.predict_proba(np.array([window_features(w) for _, w in ws]))[:, 1]
        out.append(float(p.max()))
    return np.array(out)


def threshold_for_fpr(neg_scores: np.ndarray, max_fpr: float) -> float:
    s = np.sort(neg_scores)[::-1]
    k = int(np.floor(max_fpr * len(s)))
    return float(s[k]) + 1e-6 if k < len(s) else 0.0


def evaluate(clips: list, n_splits: int = 5, max_fpr: float = 0.05) -> dict:
    X, y, g = build(clips)
    clip_y = np.array([c["label"] for c in clips])
    results = {}
    for name, proto in models().items():
        scores = np.zeros(len(clips))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for tr, te in cv.split(X, y, g):
            m = proto.__class__(**proto.get_params()) if not hasattr(proto, "steps") else make_pipeline(
                *[s.__class__(**s.get_params()) for _, s in proto.steps])
            m.fit(X[tr], y[tr])
            te_clips = np.unique(g[te])
            scores[te_clips] = clip_scores(m, clips, te_clips)
        th05 = 0.5
        pred = scores >= th05
        th_fpr = threshold_for_fpr(scores[clip_y == 0], max_fpr)
        results[name] = {
            "auc": round(roc_auc_score(clip_y, scores), 3),
            "acc@0.5": round(float((pred == clip_y).mean()), 3),
            "precision@0.5": round(precision_score(clip_y, pred, zero_division=0), 3),
            "recall@0.5": round(recall_score(clip_y, pred, zero_division=0), 3),
            "f1@0.5": round(f1_score(clip_y, pred, zero_division=0), 3),
            f"esik@fpr{int(max_fpr * 100)}": round(th_fpr, 3),
            f"recall@fpr{int(max_fpr * 100)}": round(float((scores[clip_y == 1] >= th_fpr).mean()), 3),
            "fpr@0.5": round(float(pred[clip_y == 0].mean()), 3),
        }
    return results


def test_video(model, pkl: Path, threshold: float) -> list[str]:
    clip = pickle.load(open(pkl, "rb"))[0]
    ws = windows(subsample(clip["frames"], POSE_HZ, 0))
    p = model.predict_proba(np.array([window_features(w) for _, w in ws]))[:, 1]
    lines = []
    for name, a, b, lab in TEST_SEGMENTS.get(pkl.name, []):
        sel = [pi for (s, _), pi in zip(ws, p) if a <= s < b]
        if not sel:
            continue
        frac = float(np.mean(np.array(sel) >= threshold))
        verdict = ("✅" if frac > 0 else "❌") if lab == 1 else ("✅" if frac == 0 else "⚠️")
        lines.append(f"| {pkl.stem} | {name} | {'kavga' if lab else 'değil'} | %{frac * 100:.0f} | {max(sel):.2f} | {verdict} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--out", default="models/kavga_model.joblib")
    ap.add_argument("--model", default=None, help="kaydedilecek model (varsayılan: en yüksek AUC)")
    ap.add_argument("--max-fpr", type=float, default=0.05)
    ap.add_argument("--test-video", action="append", default=[])
    ap.add_argument("--holdout", default=None,
                    help="JSON klip adları listesi: eğitime HİÇ girmez (sunum demosu için ayrılmış klipler)")
    args = ap.parse_args()

    clips = [c for d in args.data for c in pickle.load(open(d, "rb"))]
    if args.holdout:
        held = set(json.load(open(args.holdout, encoding="utf-8")))
        clips = [c for c in clips if c["name"] not in held]
        print(f"ayrılan (demo) klip: {len(held)} -> eğitimden çıkarıldı")
    print(f"{len(clips)} klip (kavga {sum(c['label'] == 1 for c in clips)}, normal {sum(c['label'] == 0 for c in clips)})")
    t0 = time.time()
    res = evaluate(clips, max_fpr=args.max_fpr)
    keys = list(next(iter(res.values())))
    print("\n| model | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
    for name, r in res.items():
        print(f"| {name} | " + " | ".join(str(r[k]) for k in keys) + " |")
    best = args.model or max(res, key=lambda n: res[n]["auc"])
    thr = res[best][f"esik@fpr{int(args.max_fpr * 100)}"]
    X, y, _ = build(clips)
    final = models()[best].fit(X, y)
    meta = {"model": best, "threshold": thr, "window_s": WINDOW_S, "hop_s": HOP_S, "pose_hz": POSE_HZ,
            "features": feature_names(), "cv": res[best], "n_clips": len(clips), "trained": time.strftime("%Y-%m-%d %H:%M")}
    joblib.dump({"model": final, "meta": meta}, args.out)
    print(f"\nkaydedildi: {args.out} (model={best}, eşik={thr} -> CV'de yanlış alarm ≤ %{args.max_fpr * 100:.0f}) "
          f"({time.time() - t0:.0f} sn)")
    if args.test_video:
        print("\n| video | bölüm | gerçek | kavga denen pencere | en yüksek olasılık | sonuç |\n|---|---|---|---|---|---|")
        for v in args.test_video:
            print("\n".join(test_video(final, Path(v), thr)))
    print(json.dumps(meta["cv"], ensure_ascii=False))


if __name__ == "__main__":
    main()
