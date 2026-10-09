import queue
import threading

from core.config import load_config
from core.schema import Alert, Event
from fusion.demo import SCENARIO, DemoController
from fusion.engine import FusionEngine
from fusion.escalation import RedTimeoutWatcher
from notify.telegram import Notifier
from storage.db import Storage


def ev(type_="fight", ts=1000.0, zone="kantin"):
    return Event(source="pose", type=type_, confidence=0.8, camera_id="cam1", zone_id=zone, ts=ts)


def test_engine_reset_lets_same_event_alert_again():
    got = []
    eng = FusionEngine(load_config(), None, got.append)  # type: ignore[arg-type]
    eng.process(ev(ts=1000))
    eng.process(ev(ts=1005))      # dedup: yeni uyarı yok
    assert len(got) == 1
    eng.reset()
    eng.process(ev(ts=1010))      # reset sonrası aynı bölge+tip yeniden uyarı üretir
    assert len(got) == 2 and got[0].id != got[1].id


def test_engine_reset_clears_escalation_counter():
    got = []
    eng = FusionEngine(load_config(), None, got.append)  # type: ignore[arg-type]
    eng.process(ev("shout", ts=1000), crowd_size=0)
    eng.reset()
    eng.process(ev("running", ts=1010))
    out = eng.process(Event(source="audio", type="glass", confidence=0.6, camera_id="c",
                            zone_id="kantin", ts=1020), crowd_size=0)
    assert [a.level for a in out] == ["yellow"]   # reset öncesi sarı sayılmaz: 3. değil 2. sarı


def test_storage_clear_removes_alerts_and_events():
    db = Storage(":memory:")
    db.save_alert(Alert(level="red", reasons=["x"], zone_id="kantin", created_at=1.0))
    db.save_event(ev())
    db.clear()
    assert db.list_alerts() == [] and db.heatmap() == {}


def test_red_watcher_and_notifier_reset():
    cfg = load_config()
    cfg["alerts"]["auto_escalate_red"] = {"enabled": True, "timeout_s": 10}
    w = RedTimeoutWatcher(cfg)
    w.track("a", "red", now=0)
    assert w.due(now=20, status_of=lambda i: "pending") == ["a"]
    w.reset()
    w.track("a", "red", now=100)   # reset sonrası aynı id yeniden izlenebilir
    assert w.due(now=111, status_of=lambda i: "pending") == ["a"]

    n = Notifier(cfg)
    n._sent.add(("a", "red"))
    n.reset()
    assert n._sent == set()


def test_reset_endpoint_is_closed_outside_demo_mode():
    """Gerçek kamera/ses çalışırken kayıtlar ASLA silinmemeli: demo dışında 404."""
    import pytest
    pytest.importorskip("httpx")  # TestClient için (requirements'ta yok; geliştirme ortamında genelde kurulu)
    from fastapi.testclient import TestClient

    from api.server import FrameStore, Hub, create_app

    cfg = load_config()
    live = TestClient(create_app(cfg, Storage(":memory:"), FrameStore(), Hub()))
    assert live.get("/api/status").json() == {"demo": False}
    assert live.post("/api/demo/reset").status_code == 404

    called = []
    demo = TestClient(create_app(cfg, Storage(":memory:"), FrameStore(), Hub(),
                                 demo_reset=lambda: called.append(1)))
    assert demo.get("/api/status").json() == {"demo": True}
    assert demo.post("/api/demo/reset").status_code == 200 and called == [1]


def test_demo_controller_restart_replays_scenario_from_start():
    q: "queue.Queue[Event]" = queue.Queue()
    ctl = DemoController(q, speed=10_000)   # çok hızlı: bekleme süreleri ~0
    ctl.start()
    ctl._thread.join(timeout=5)
    first = q.qsize()
    assert first == len(SCENARIO)
    ctl.start()                              # yeniden başlat: senaryo baştan oynar
    ctl._thread.join(timeout=5)
    assert q.qsize() == 2 * first
    ctl.stop()
    assert threading.active_count() >= 1
