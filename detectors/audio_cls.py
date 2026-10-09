"""Ses sınıflandırma (çığlık, bağırma, silah sesi, cam kırılması) + ses seviyesi (dB).

Model (config.yaml > audio.backend):
- efficientat: EfficientAT mn10_as (MIT, AudioSet mAP 47.1, ~22 ms / hop, CPU).
  Kod: detectors/third_party/efficientat (vendored). Ağırlık ilk çalıştırmada models/ altına iner.
  Model 10 sn kliplerle eğitildiği için 1 sn'lik pencere sıfırla 10 sn'ye tamamlanır (pad_to_s);
  aksi halde <3 sn girişte tüm sınıflar ~1.0 çıkıyor.
- yamnet: MediaPipe AudioClassifier + models/yamnet.tflite (AudioSet mAP ~31). Yedek.
efficientat yüklenemezse otomatik yamnet'e düşer.

Her hop_s saniyede son window_s saniye sınıflandırılır:
- AudioSet sınıfları class_map ile scream/shout/gunshot/glass'a eşlenir; skor >= min_conf ise Event.
- dB (RMS, db_offset ile kalibre) bölge tipine ve teneffüs/ders saatine göre eşiği db_min_hops
  ölçüm boyunca aşarsa "shout" Event'i (confidence 0.5–1.0, eşiğin ne kadar aşıldığına göre).
  Şemada ayrı bir "yüksek ses" tipi yok; CLAUDE.md'deki "Shout/Yell veya dB eşiği aşımı -> sarı" kuralı.

Tek başına çalıştırma:
    python -m detectors.audio_cls                         # varsayılan mikrofon
    python -m detectors.audio_cls --audio test_media/ciglik.wav
    python -m detectors.audio_cls --list-devices
"""
from __future__ import annotations

import argparse
import csv
import math
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from core.config import load_config, resolve_path
from core.schema import Event

EFFICIENTAT_DIR = Path(__file__).resolve().parent / "third_party" / "efficientat"
EVENT_TYPES = ("scream", "shout", "gunshot", "glass")


class _EfficientATBackend:
    name = "efficientat"

    def __init__(self, cfg: dict, labels_wanted: list[str], device: str) -> None:
        import warnings

        import torch

        warnings.filterwarnings("ignore", message="Don't use ConvNormActivation")
        from detectors.third_party.efficientat.mn import model as mn
        from detectors.third_party.efficientat.mn.utils import NAME_TO_WIDTH
        from detectors.third_party.efficientat.preprocess import MelSTFT

        c = cfg["audio"]["efficientat"]
        self.torch = torch
        self.sample_rate = int(c["sample_rate"])
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        mn.model_dir = str(resolve_path("models"))
        name = c["model_name"]
        self.model = mn.get_model(width_mult=NAME_TO_WIDTH(name), pretrained_name=name).to(self.device).eval()
        self.mel = MelSTFT(sr=self.sample_rate).to(self.device).eval()
        self.pad_to = int(c["pad_to_s"] * self.sample_rate)

        with open(EFFICIENTAT_DIR / "class_labels_indices.csv", encoding="utf-8") as f:
            all_labels = [row[2] for row in list(csv.reader(f))[1:]]
        missing = [lb for lb in labels_wanted if lb not in all_labels]
        if missing:
            raise RuntimeError(f"AudioSet'te olmayan sınıf adları (config audio.class_map): {missing}")
        self.index = {lb: all_labels.index(lb) for lb in labels_wanted}

    def classify(self, wav: np.ndarray) -> dict[str, float]:
        torch = self.torch
        if len(wav) < self.pad_to:
            # model kısa girişte (<3 sn) anlamsız skor üretiyor; eğitimdeki gibi 10 sn'ye sıfırla tamamla
            wav = np.pad(wav, (0, self.pad_to - len(wav)))
        with torch.no_grad():
            x = torch.from_numpy(wav).to(self.device)[None]
            logits, _ = self.model(self.mel(x).unsqueeze(0))
            probs = torch.sigmoid(logits.float()).squeeze(0).cpu().numpy()
        return {lb: float(probs[i]) for lb, i in self.index.items()}


class _YamnetBackend:
    name = "yamnet"

    def __init__(self, cfg: dict, labels_wanted: list[str], device: str) -> None:  # noqa: ARG002
        from mediapipe.tasks.python import audio
        from mediapipe.tasks.python.components.containers import audio_data
        from mediapipe.tasks.python.core.base_options import BaseOptions

        c = cfg["audio"]["yamnet"]
        model_path = resolve_path(c["model"])
        if not model_path.is_file():
            raise RuntimeError(
                f"{model_path} yok. İndirin: https://storage.googleapis.com/mediapipe-models/"
                f"audio_classifier/yamnet/float32/1/yamnet.tflite"
            )
        self.sample_rate = int(c["sample_rate"])
        self._AudioData = audio_data.AudioData
        self.labels = labels_wanted
        opts = audio.AudioClassifierOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=audio.RunningMode.AUDIO_CLIPS,
            category_allowlist=labels_wanted,
            max_results=-1,
            score_threshold=0.0,
        )
        self.clf = audio.AudioClassifier.create_from_options(opts)

    def classify(self, wav: np.ndarray) -> dict[str, float]:
        scores = {lb: 0.0 for lb in self.labels}
        # YAMNet ~0.975 sn'lik dilimlere böler; her sınıf için dilimlerin en yükseğini al
        for res in self.clf.classify(self._AudioData.create_from_array(wav, self.sample_rate)):
            for cat in res.classifications[0].categories:
                if cat.category_name in scores:
                    scores[cat.category_name] = max(scores[cat.category_name], cat.score)
        return scores


def _is_break(now: float, cfg: dict) -> bool:
    """schedule.breaks aralığında mı (HH:MM, başlangıç dahil, bitiş hariç)."""
    t = datetime.fromtimestamp(now)
    minutes = t.hour * 60 + t.minute
    for start, end in cfg.get("schedule", {}).get("breaks", []):
        sh, sm = map(int, start.split(":"))
        eh, em = map(int, end.split(":"))
        if sh * 60 + sm <= minutes < eh * 60 + em:
            return True
    return False


class AudioDetector:
    def __init__(
        self,
        cfg: dict,
        zone_id: str | None = None,
        out_queue: queue.Queue | None = None,
        device_id: str | None = None,
        backend: str | None = None,
    ) -> None:
        self.full_cfg = cfg
        self.cfg = cfg["audio"]
        self.zone_id = zone_id or self.cfg["zone_id"]
        self.device_id = device_id or self.cfg["device_id"]
        self.out_queue = out_queue

        self.class_map: dict[str, str] = self.cfg["class_map"]
        self.backend = self._load_backend(backend or self.cfg["backend"])
        self.sample_rate = self.backend.sample_rate
        self.min_conf: dict[str, float] = self.cfg["min_conf"][self.backend.name]
        self.window = int(self.cfg["window_s"] * self.sample_rate)
        self.hop = int(self.cfg["hop_s"] * self.sample_rate)

        zone_type = cfg.get("zones", {}).get(self.zone_id, {}).get("type", "default")
        thresholds = self.cfg["db_thresholds"]
        self.db_threshold = thresholds.get(zone_type, thresholds["default"])

        self._buf = np.zeros(self.window, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._loud_hops = 0
        self._last_emit: dict[str, float] = {}
        self.last_db: float = 0.0
        self.last_scores: dict[str, float] = {t: 0.0 for t in EVENT_TYPES}

    def _load_backend(self, name: str):
        labels = list(self.class_map)
        device = self.full_cfg["general"]["device"]
        if name == "efficientat":
            try:
                return _EfficientATBackend(self.full_cfg, labels, device)
            except Exception as e:  # noqa: BLE001  demo'yu riske atma: yedeğe geç
                print(f"[audio] EfficientAT yüklenemedi ({e}) -> YAMNet'e geçiliyor")
        return _YamnetBackend(self.full_cfg, labels, device)

    def current_db_threshold(self, now: float) -> float:
        return float(self.db_threshold["break" if _is_break(now, self.full_cfg) else "lesson"])

    def _emit(self, etype: str, conf: float, now: float) -> Event | None:
        if now - self._last_emit.get(etype, -1e9) < self.cfg["emit_cooldown_s"]:
            return None
        self._last_emit[etype] = now
        event = Event(source="audio", type=etype, confidence=round(conf, 3),  # type: ignore[arg-type]
                      camera_id=self.device_id, zone_id=self.zone_id, ts=now)
        if self.out_queue is not None:
            self.out_queue.put(event)
        return event

    def process_hop(self, hop: np.ndarray, now: float) -> list[Event]:
        """hop_s uzunluğunda yeni ses: tampona ekle, son window_s'i sınıflandır."""
        self._buf = np.concatenate([self._buf, hop])[-self.window:]
        events: list[Event] = []

        label_scores = self.backend.classify(self._buf)
        type_scores = {t: 0.0 for t in EVENT_TYPES}
        for label, score in label_scores.items():
            t = self.class_map[label]
            type_scores[t] = max(type_scores[t], score)
        self.last_scores = type_scores
        for t, score in type_scores.items():
            if score >= self.min_conf.get(t, 1.01):
                ev = self._emit(t, score, now)
                if ev:
                    events.append(ev)

        rms = float(np.sqrt(np.mean(hop.astype(np.float64) ** 2)))
        self.last_db = max(0.0, 20 * math.log10(rms + 1e-10) + self.cfg["db_offset"])
        threshold = self.current_db_threshold(now)
        self._loud_hops = self._loud_hops + 1 if self.last_db >= threshold else 0
        if self._loud_hops >= self.cfg["db_min_hops"]:
            conf = min(1.0, 0.5 + (self.last_db - threshold) / 20)
            ev = self._emit("shout", conf, now)
            if ev:
                events.append(ev)
        return events

    def feed(self, chunk: np.ndarray, now: float) -> list[Event]:
        """Herhangi uzunlukta ses parçası ver; her hop dolduğunda işler."""
        self._pending = np.concatenate([self._pending, chunk])
        events: list[Event] = []
        while len(self._pending) >= self.hop:
            hop, self._pending = self._pending[:self.hop], self._pending[self.hop:]
            events += self.process_hop(hop, now)
        return events

    def run(self, stop_event: threading.Event, source: str | int | None = None,
            on_hop=None) -> None:
        """Thread'de çalışır: kaynaktan okur, event'leri out_queue'ya basar. source None ise config'deki input."""
        from sources.audio import AudioSource

        src = AudioSource(self.cfg["input"] if source is None else source, self.sample_rate)
        try:
            while not stop_event.is_set():
                chunk = src.read()
                if chunk is None:
                    if src.is_file:
                        break
                    continue
                events = self.feed(chunk, src.last_ts)
                if on_hop is not None:
                    on_hop(self, events)
        finally:
            src.release()


def main() -> None:
    from sources.audio import list_devices

    cfg = load_config()
    parser = argparse.ArgumentParser(description="Ses detektörünü tek başına çalıştır.")
    parser.add_argument("--audio", default=None, help="default = mikrofon, cihaz numarası veya wav dosyası")
    parser.add_argument("--backend", default=None, choices=["efficientat", "yamnet"], help="config'i ezer")
    parser.add_argument("--list-devices", action="store_true", help="mikrofonları listele ve çık")
    args = parser.parse_args()
    if args.list_devices:
        print(list_devices())
        return

    det = AudioDetector(cfg, backend=args.backend)
    src = args.audio if args.audio is not None else cfg["audio"]["input"]
    print(f"[audio] model={det.backend.name} kaynak={src} bölge={det.zone_id} "
          f"dB eşiği={det.current_db_threshold(time.time()):.0f}  (çıkış: Ctrl+C)")

    def on_hop(d: AudioDetector, events: list[Event]) -> None:
        s = d.last_scores
        bar = "#" * max(0, min(30, int((d.last_db - 30) / 3)))
        line = (f"dB={d.last_db:5.1f} {bar:<30} | çığlık {s['scream']:.2f}  bağırma {s['shout']:.2f}  "
                f"silah {s['gunshot']:.2f}  cam {s['glass']:.2f}")
        print("\r" + line, end="", flush=True)
        for ev in events:
            print("\n" + ev.to_json(), flush=True)

    stop = threading.Event()
    try:
        det.run(stop, source=src, on_hop=on_hop)
        print("\n[audio] kaynak bitti.")
    except KeyboardInterrupt:
        stop.set()
        print()


if __name__ == "__main__":
    main()
