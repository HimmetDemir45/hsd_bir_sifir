from datetime import datetime

import pytest

from core.config import load_config
from core.schema import Event
from fusion.engine import FusionEngine
from fusion.rules import classify, is_break


@pytest.fixture
def cfg():
    return load_config()


def make_engine(cfg):
    got = []
    return FusionEngine(cfg, None, got.append), got  # type: ignore[arg-type]


def ev(type_, conf=0.8, ts=1000.0, zone="kantin", source="pose"):
    return Event(source=source, type=type_, confidence=conf, camera_id="cam1", zone_id=zone, ts=ts)


def test_red_weapon_threshold(cfg):
    assert classify(ev("knife", 0.82, source="weapon"), cfg)[0] == "red"
    assert classify(ev("knife", 0.59, source="weapon"), cfg) is None


def test_red_gunshot_threshold(cfg):
    th = cfg["alerts"]["red"]["gunshot_min_conf"]  # config'den: eşik ayarlanınca test bozulmasın
    assert classify(ev("gunshot", th, source="audio"), cfg)[0] == "red"
    assert classify(ev("gunshot", round(th - 0.01, 2), source="audio"), cfg) is None


def test_scream_depends_on_crowd(cfg):
    assert classify(ev("scream", source="audio"), cfg, crowd_size=3)[0] == "yellow"
    assert classify(ev("scream", source="audio"), cfg, crowd_size=4)[0] == "orange"


def test_reason_text_is_turkish(cfg):
    assert classify(ev("knife", 0.82, source="weapon"), cfg)[1] == "Bıçak tespit edildi (%82)"


def test_is_break(cfg):
    assert is_break(datetime(2026, 10, 9, 10, 15), cfg)
    assert not is_break(datetime(2026, 10, 9, 10, 20), cfg)
    assert not is_break(datetime(2026, 10, 9, 9, 0), cfg)


def test_dedup_merges_same_zone_and_type(cfg):
    eng, got = make_engine(cfg)
    a = eng.process(ev("fight", 0.7, ts=1000))
    b = eng.process(ev("fight", 0.9, ts=1010))  # aynı sebep metni -> değişiklik yok
    assert len(a) == 1 and b == []
    assert len(got) == 1


def test_dedup_window_expires(cfg):
    eng, got = make_engine(cfg)
    eng.process(ev("fight", ts=1000))
    eng.process(ev("fight", ts=1000 + cfg["alerts"]["dedup_window_s"] + 1))
    assert len(got) == 2 and got[0].id != got[1].id


def test_dedup_upgrades_level_and_adds_reason(cfg):
    eng, got = make_engine(cfg)
    eng.process(ev("scream", source="audio", ts=1000), crowd_size=0)  # sarı
    out = eng.process(ev("scream", source="audio", ts=1005), crowd_size=5)  # turuncu
    assert out[0].id == got[0].id  # aynı uyarı güncellendi
    assert out[0].level == "orange" and len(out[0].reasons) == 2


def test_different_zone_not_deduped(cfg):
    eng, got = make_engine(cfg)
    eng.process(ev("fight", ts=1000, zone="kantin"))
    eng.process(ev("fight", ts=1001, zone="bahce"))
    assert len(got) == 2


def test_escalation_three_yellows_to_orange(cfg):
    eng, got = make_engine(cfg)
    eng.process(ev("shout", 0.5, ts=1000, source="audio"))
    eng.process(ev("running", ts=1030))
    out = eng.process(ev("glass", ts=1060, source="audio"))
    assert [a.level for a in out] == ["yellow", "orange"]


def test_no_escalation_outside_window(cfg):
    eng, got = make_engine(cfg)
    eng.process(ev("shout", 0.5, ts=1000, source="audio"))
    eng.process(ev("running", ts=1050))
    eng.process(ev("glass", ts=1000 + cfg["alerts"]["escalation"]["window_s"] + 10, source="audio"))
    assert all(a.level == "yellow" for a in got)


def test_no_escalation_across_zones(cfg):
    eng, got = make_engine(cfg)
    eng.process(ev("shout", 0.5, ts=1000, zone="kantin", source="audio"))
    eng.process(ev("running", ts=1010, zone="bahce"))
    eng.process(ev("glass", ts=1020, zone="koridor_1", source="audio"))
    assert all(a.level == "yellow" for a in got)
