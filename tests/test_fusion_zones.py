import struct

from core.config import ROOT, load_config
from core.schema import Event
from fusion.demo import SCENARIO
from fusion.engine import FusionEngine


def plan_size() -> tuple[int, int]:
    """web/floorplan.png boyutu (PNG başlığından; ek paket gerekmez). Isı haritası tuvali bununla aynı olmalı."""
    with open(ROOT / "web" / "floorplan.png", "rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def bbox(poly):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def test_every_zone_is_well_formed_and_inside_the_plan():
    w, h = plan_size()
    for zid, z in load_config()["zones"].items():
        assert z.get("name") and z.get("type"), zid
        assert len(z["polygon"]) >= 3, zid
        x1, y1, x2, y2 = bbox(z["polygon"])
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, f"{zid} planın dışında ({w}x{h})"


def test_zones_do_not_overlap():
    boxes = {zid: bbox(z["polygon"]) for zid, z in load_config()["zones"].items()}
    ids = list(boxes)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ax1, ay1, ax2, ay2 = boxes[a]
            bx1, by1, bx2, by2 = boxes[b]
            overlap = min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)
            assert not overlap, f"{a} ve {b} üst üste biniyor"


def test_cameras_and_demo_use_known_zones():
    cfg = load_config()
    zones = set(cfg["zones"])
    for cam in cfg["cameras"]:
        assert cam["zone_id"] in zones, cam
    for *_, zone in SCENARIO:
        assert zone in zones, f"demo senaryosu bilinmeyen bölge kullanıyor: {zone}"


def test_demo_story_is_unchanged_by_extra_zone_events():
    """Başka bölgelerdeki küçük sarı olaylar eskalasyon/kırmızı üretmemeli; kırmızı yalnızca Kantin'de."""
    cfg = load_config()
    got = []
    eng = FusionEngine(cfg, None, got.append)  # type: ignore[arg-type]
    ts = 1000.0
    for delay, source, etype, conf, cam, zone in SCENARIO:
        ts += delay
        eng.process(Event(source=source, type=etype, confidence=conf,  # type: ignore[arg-type]
                          camera_id=cam, zone_id=zone, ts=ts), crowd_size=0)
    by_zone: dict[str, set[str]] = {}
    for a in got:
        by_zone.setdefault(a.zone_id, set()).add(a.level)
    assert by_zone["kantin"] == {"yellow", "orange", "red"}
    for zone, levels in by_zone.items():
        if zone != "kantin":
            assert levels == {"yellow"}, f"{zone}: beklenmeyen seviye {levels}"
