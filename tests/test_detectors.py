"""Kişi A modüllerinin mantık testleri (model dosyası GEREKMEZ: YOLO ve ses modelleri sahte nesneyle değiştirilir).

Kapsam: poz kuralları (kavga/düşme/koşuşma), silah 3/5 kare kuralı, ses event'i gönderme/bekleme/yardımcı model,
dB kuralı, yüz kutusu boyutu ve zamansal hafıza, klip penceresi.
"""
import time
from datetime import datetime

import numpy as np
import pytest

import detectors.pose as pose_mod
import detectors.weapon as weapon_mod
from core.config import load_config
from detectors.audio_cls import AudioDetector
from detectors.pose import Person, PoseDetector
from detectors.weapon import Detection, WeaponDetector
from privacy.blur import FaceBlurrer
from sources.clip_buffer import ClipBuffer


class DummyYOLO:
    """Ultralytics YOLO yerine: model dosyası okumaz, tahmin yapmaz."""
    names = {0: "person", 43: "knife"}

    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def pose_det(cfg, monkeypatch):
    monkeypatch.setattr(pose_mod, "YOLO", DummyYOLO)
    cfg["pose"]["every_n_frames"] = 1
    cfg["pose"]["fight_model"]["enabled"] = False   # kural testleri kuralı ölçer; model ayrı testte (FakeFightModel)
    return PoseDetector(cfg, "cam1", "kantin")


def person(cx, cy, h=200, horizontal=False, wrist=(0, 0), conf=0.9):
    """Sentetik iskelet: dik (ya da yatay) duran, bileği wrist kadar kaydırılmış bir kişi."""
    kps = np.zeros((17, 2), np.float32)
    if horizontal:
        kps[5] = kps[6] = (cx - h * 0.3, cy)
        kps[11] = kps[12] = (cx + h * 0.1, cy)
        box = (int(cx - h / 2), int(cy - h * 0.15), int(cx + h / 2), int(cy + h * 0.15))
    else:
        kps[5] = kps[6] = (cx, cy - h * 0.3)
        kps[11] = kps[12] = (cx, cy + h * 0.05)
        box = (int(cx - h * 0.2), int(cy - h / 2), int(cx + h * 0.2), int(cy + h / 2))
    kps[7] = kps[8] = (cx + h * 0.1, cy - h * 0.1)
    kps[9] = kps[10] = (cx + h * 0.15 + wrist[0], cy + wrist[1])
    kps[0] = (cx, cy - h * 0.45)
    for i in (1, 2, 3, 4, 13, 14, 15, 16):
        kps[i] = (cx, cy)
    return Person(box, conf, kps, np.ones(17, np.float32))


def run_pose(det, frames, dt=0.2):
    """frames: her adımdaki kişi listesi. Dönüş: [(zaman, tip)]."""
    out = []
    for i, persons in enumerate(frames):
        det.detect = lambda frame, p=persons: p
        _, events = det.process(None, i * dt)
        out += [(round(e.ts, 1), e.type) for e in events]
    return out


# ---------------- config ----------------
def test_config_keeps_detector_settings(cfg):
    """Birleştirmede config.yaml'ın eski sürümü alınırsa bu ayarlar sessizce kaybolabiliyor (2026-10-09'da oldu)."""
    g, a, p = cfg["general"], cfg["audio"], cfg["privacy"]
    for key in ("torch_threads", "max_frame_side", "clip_pre_ratio", "clip_max_side", "reconnect_s"):
        assert key in g, f"general.{key} eksik"
    for key in ("reemit_delta", "type_backend", "db_rule_on_files", "emit_cooldown_s"):
        assert key in a, f"audio.{key} eksik"
    assert a["type_backend"].get("gunshot") == "yamnet", "silah sesi YAMNet'te olmalı (EfficientAT silahta zayıf)"
    for key in ("face_model", "person_conf", "persist_s", "top_band", "head_scale"):
        assert key in p, f"privacy.{key} eksik"
    # dedektörün silah eşiği fusion'ın kırmızı eşiğinden yüksek olursa kırmızı uyarı hiç oluşamaz
    assert a["min_conf"]["yamnet"]["gunshot"] <= cfg["alerts"]["red"]["gunshot_min_conf"]
    # silah dedektörünün her tip eşiği fusion kırmızı eşiğinin altına inerse dedektör basar ama kırmızı oluşmaz: tutarlılık
    assert min(cfg["weapon"]["min_conf"].values()) >= cfg["alerts"]["red"]["weapon_min_conf"]


# ---------------- poz ----------------
def test_fight_close_and_fast(pose_det):
    frames = [[person(300, 300, wrist=(80 * (i % 2), 0)), person(400, 300)] for i in range(15)]
    events = run_pose(pose_det, frames)
    fights = [t for t, typ in events if typ == "fight"]
    assert fights, "yakın + hızlı kol hareketi kavga sayılmalı"
    assert fights[0] >= pose_det.cfg["fight"]["min_duration_s"], "min_duration_s dolmadan kavga basılmamalı"


def test_no_fight_when_far(pose_det):
    frames = [[person(100, 300, wrist=(80 * (i % 2), 0)), person(500, 300)] for i in range(15)]
    assert not [t for t, typ in run_pose(pose_det, frames) if typ == "fight"]


def test_fight_survives_short_dropout(pose_det):
    # Kişi bir karede kaybolsa da (takip kopması / düşük FPS) kavga sayacı sıfırlanmamalı (gap_s)
    frames = []
    for i in range(15):
        a = person(300, 300, wrist=(80 * (i % 2), 0))
        frames.append([a] if i == 4 else [a, person(400, 300)])
    fights = [t for t, typ in run_pose(pose_det, frames) if typ == "fight"]
    assert fights and fights[0] <= 1.6


def test_fall_once_after_duration(pose_det):
    # ani düşüş: dik dururken (merkez y=300) 0.2 sn içinde yerde yatay (merkez y=380, 0.4 x boy aşağı)
    frames = [[person(300, 380, horizontal=True) if i >= 2 else person(300, 300)] for i in range(25)]
    falls = [t for t, typ in run_pose(pose_det, frames) if typ == "fall"]
    assert len(falls) == 1, "her düşmede tek event"
    assert falls[0] - 0.4 >= pose_det.cfg["fall"]["min_duration_s"] - 1e-6


class FakeFightModel:
    """Öğrenilmiş kavga modeli yerine: özelliklerden bağımsız sabit olasılık döndürür."""
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p]] * len(X))


def test_fight_model_needs_consecutive_windows(pose_det):
    pose_det.fight_meta = {"window_s": 1.5, "threshold": 0.8}
    pose_det.cfg["fight_model"] = {"threshold": None, "eval_every_s": 0.5, "min_consecutive": 2}
    calm = [[person(100, 300), person(500, 300)] for _ in range(20)]
    pose_det.fight_model = FakeFightModel(0.5)          # eşik altı -> kavga yok
    assert "fight" not in [typ for _, typ in run_pose(pose_det, calm)]
    pose_det.reset()
    pose_det.fight_model = FakeFightModel(0.95)         # eşik üstü, art arda 2 değerlendirme -> kavga
    fights = [t for t, typ in run_pose(pose_det, calm) if typ == "fight"]
    assert fights and fights[0] >= 0.5, "tek değerlendirme yetmemeli (min_consecutive)"


def test_slow_lying_down_is_not_fall(pose_det):
    # yatağa uzanma gibi: yerinde (merkez inmeden) yataya geçiş -> ani düşüş yok -> düşme değil
    frames = [[person(300, 300, horizontal=i >= 2)] for i in range(25)]
    assert "fall" not in [typ for _, typ in run_pose(pose_det, frames)]


def test_head_bow_upper_body_is_not_horizontal(pose_det):
    # laptop kamerası: sadece kafa+omuz görünüyor (kalça/bacak yok), kafa eğilince kutu genişliyor -> yatay DEĞİL
    kps = np.zeros((17, 2), np.float32)
    kpc = np.zeros(17, np.float32)
    kpc[[0, 1, 2, 3, 4, 5, 6]] = 0.9
    wide_box = Person((0, 200, 400, 400), 0.9, kps, kpc)   # genişlik/yükseklik = 2.0
    assert not pose_det._is_horizontal(wide_box)


def test_fall_survives_pose_flicker(pose_det):
    # yatayken tek karelik "dik" titremesi süreyi sıfırlamamalı (gap_s)
    frames = [[person(300, 300)] if i < 2 or i == 8 else [person(300, 380, horizontal=True)] for i in range(25)]
    assert "fall" in [typ for _, typ in run_pose(pose_det, frames)]


def test_running_vs_walking(pose_det, cfg, monkeypatch):
    # 3 boy/sn (canli.mp4'teki koşu 2.2-3.5); tek kişi de yeterli (min_people 1)
    fast = [[person(100 + 120 * i, 200)] for i in range(10)]
    assert "running" in [typ for _, typ in run_pose(pose_det, fast)]
    slow_det = PoseDetector(cfg, "cam1", "kantin")
    slow = [[person(100 + 20 * i, 200), person(100 + 20 * i, 450)] for i in range(10)]
    assert not run_pose(slow_det, slow)


# ---------------- silah ----------------
def test_weapon_three_of_five_and_cooldown(cfg, monkeypatch):
    monkeypatch.setattr(weapon_mod, "YOLO", DummyYOLO)
    det = WeaponDetector(cfg, "cam1", "kantin")
    strong = Detection("knife", "knife", 0.8, (0, 0, 10, 10))
    weak = Detection("knife", "knife", 0.4, (0, 0, 10, 10))   # min_conf altı: sayılmaz
    seq = [[strong], [], [weak], [strong], [], [strong], [strong], [strong]]
    hits = [i for i, dets in enumerate(seq) if det.update(dets, now=100 + i * 0.1)]
    assert hits == [6], "son 5 karenin 3'ünde güçlü tespit olunca bir kez (bekleme süresi)"


def test_weapon_alternating_models_only_update_evaluated(cfg, monkeypatch):
    # sırayla çalıştırmada bu karede modeli çalışmayan tipin penceresine 0 eklenmemeli (pencere seyrelmesin)
    monkeypatch.setattr(weapon_mod, "YOLO", DummyYOLO)
    det = WeaponDetector(cfg, "cam1", "kantin")
    det._evaluated = {"knife"}
    det.update([Detection("knife", "knife", 0.9, (0, 0, 10, 10))], now=100.0)
    assert list(det.history["knife"]) == [0.9]
    assert all(len(h) == 0 for t, h in det.history.items() if t != "knife")


# ---------------- ses ----------------
class DummyAudioBackend:
    def __init__(self, name, sample_rate, scores=None):
        self.name, self.sample_rate, self.scores = name, sample_rate, dict(scores or {})

    def classify(self, wav):
        return dict(self.scores)


@pytest.fixture
def audio_det(cfg, monkeypatch):
    primary = DummyAudioBackend("efficientat", 32000)
    monkeypatch.setattr(AudioDetector, "_load_backend", lambda self, name: primary)
    monkeypatch.setattr(AudioDetector, "_load_aux_backend", lambda self: (None, set()))
    det = AudioDetector(cfg)
    det.db_rule_enabled = False
    return det


LESSON_TS = datetime(2026, 10, 9, 9, 0).timestamp()   # ders saati (teneffüs değil)


def test_audio_cooldown_and_reemit(audio_det):
    assert audio_det._emit("gunshot", 0.30, 100.0)
    assert audio_det._emit("gunshot", 0.33, 100.5) is None, "bekleme içinde küçük artış gönderilmez"
    assert audio_det._emit("gunshot", 0.50, 101.0), "belirgin artış (reemit_delta) gönderilir"
    assert audio_det._emit("gunshot", 0.40, 104.5), "bekleme bitince yine gönderilir"


def test_audio_thresholds(audio_det):
    hop = np.zeros(audio_det.hop, np.float32)
    audio_det.backend.scores = {"Screaming": audio_det.min_conf["scream"] - 0.01}
    assert not audio_det.process_hop(hop, LESSON_TS)
    audio_det.backend.scores = {"Screaming": audio_det.min_conf["scream"] + 0.05}
    assert [e.type for e in audio_det.process_hop(hop, LESSON_TS + 10)] == ["scream"]


def test_audio_aux_backend_owns_gunshot(cfg, monkeypatch):
    primary = DummyAudioBackend("efficientat", 32000, {"Gunshot, gunfire": 0.99})   # ana model yok sayılmalı
    aux = DummyAudioBackend("yamnet", 16000, {"Gunshot, gunfire": 0.05})
    monkeypatch.setattr(AudioDetector, "_load_backend", lambda self, name: primary)
    monkeypatch.setattr(AudioDetector, "_load_aux_backend", lambda self: (aux, {"gunshot"}))
    det = AudioDetector(cfg)
    det.db_rule_enabled = False
    hop = np.zeros(det.hop, np.float32)
    assert not det.process_hop(hop, LESSON_TS), "silah skoru yardımcı modelden gelmeli"
    aux.scores = {"Gunshot, gunfire": det.min_conf["gunshot"] + 0.1}
    assert [e.type for e in det.process_hop(hop, LESSON_TS + 10)] == ["gunshot"]


def test_db_rule_and_file_switch(audio_det):
    audio_det.db_rule_enabled = True
    loud = np.random.default_rng(0).normal(0, 0.4, audio_det.hop).astype(np.float32)   # ~92 dB (offset 100)
    events = []
    for i in range(audio_det.cfg["db_min_hops"]):
        events += audio_det.process_hop(loud, LESSON_TS + i)
    assert [e.type for e in events] == ["shout"]
    audio_det.db_rule_enabled = False   # ses dosyasında run() kapatır
    audio_det._last_emit.clear()
    assert not audio_det.process_hop(loud, LESSON_TS + 100)


# ---------------- yüz bulanıklaştırma ----------------
def face_person(conf=0.9):
    kps = np.zeros((17, 2), np.float32)
    kps[0] = (100, 100)                     # burun
    kps[1], kps[2] = (95, 95), (105, 95)    # gözler
    kps[3], kps[4] = (90, 98), (110, 98)    # kulaklar
    kps[5], kps[6] = (70, 140), (130, 140)  # omuzlar
    return Person((60, 70, 140, 300), conf, kps, np.ones(17, np.float32))


def test_head_box_covers_face_not_neck(cfg):
    x1, y1, x2, y2 = FaceBlurrer(cfg).head_box(face_person())
    assert x1 < 90 and x2 > 110 and y1 < 90, "yüz (kulaklar, gözler, alın) kapanmalı"
    assert y2 < 125, "kutu boyna/omuza (y=140) inmemeli"
    assert (x2 - x1) < 0.6 * 60, "omuz genişliği kutuyu büyütmemeli"


def test_blur_changes_only_face_and_persists(cfg):
    blur = FaceBlurrer(cfg)
    blur.face_boxes = lambda frame: []      # yüz dedektörü bu testte devre dışı
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (320, 200, 3), dtype=np.uint8)
    out = blur.blur(frame.copy(), [face_person()], now=0.0)
    assert not np.array_equal(out[85:110, 92:108], frame[85:110, 92:108]), "yüz pikselleşmeli"
    assert np.array_equal(out[250:320], frame[250:320]), "vücudun altı değişmemeli"
    # kişi bir kare kaybolsa da (persist_s içinde) yüz açılmamalı; süre geçince açılmalı
    later = blur.blur(frame.copy(), [], now=0.2)
    assert not np.array_equal(later[85:110, 92:108], frame[85:110, 92:108])
    much_later = blur.blur(frame.copy(), [], now=0.2 + cfg["privacy"]["persist_s"] + 0.5)
    assert np.array_equal(much_later, frame)


# ---------------- kamera ----------------
class FakeCapture:
    """cv2.VideoCapture yerine: ilk açılış N kare verip kopar; yeniden açılınca tekrar kare verir."""
    opened = 0

    def __init__(self, *args, **kwargs):
        FakeCapture.opened += 1
        self.left = 3 if FakeCapture.opened == 1 else 10 ** 6

    def isOpened(self):
        return True

    def read(self):
        if self.left <= 0:
            return False, None
        self.left -= 1
        time.sleep(0.01)
        return True, np.full((4, 4, 3), FakeCapture.opened, np.uint8)

    def release(self):
        pass


def test_camera_reconnects_after_drop(monkeypatch):
    import sources.camera as cam_mod
    FakeCapture.opened = 0
    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", FakeCapture)
    cam = cam_mod.Camera(0, reconnect_s=0.2)
    try:
        deadline = time.time() + 5
        frame = None
        while time.time() < deadline:
            frame = cam.read(timeout=0.5)
            if frame is not None and frame[0, 0, 0] == 2:   # 2. açılıştan gelen kare
                break
        assert FakeCapture.opened >= 2, "kopan kamera yeniden açılmalı"
        assert frame is not None and frame[0, 0, 0] == 2, "yeniden açıldıktan sonra kare gelmeli"
    finally:
        cam.release()


# ---------------- klip ----------------
def test_clip_window_before_and_after(tmp_path):
    buf = ClipBuffer(10, pre_ratio=0.5, max_side=160)
    for i in range(200):   # 15 FPS, 1000.0 -> 1013.3 sn
        buf.add(np.full((120, 160, 3), i % 255, np.uint8), 1000 + i / 15)
    out = buf.save(str(tmp_path / "klip.mp4"), event_ts=1005.0)
    import cv2
    cap = cv2.VideoCapture(out)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert 140 <= n <= 152, "olayın 5 sn öncesi + 5 sn sonrası"
    assert 13 <= fps <= 17, "gerçek kare hızında"


def test_clip_without_frames_in_window_raises(tmp_path):
    buf = ClipBuffer(10, pre_ratio=0.5, max_side=160)
    buf.add(np.zeros((120, 160, 3), np.uint8), 2000.0)   # olaydan çok sonra: pencerede kare yok
    with pytest.raises(RuntimeError):
        buf.save(str(tmp_path / "bos.mp4"), event_ts=1000.0)
