import time

import pytest

from core.config import load_config
from core.schema import Alert
from storage.db import Storage


def test_false_alarm_stats_counts_only_decided():
    db = Storage(":memory:")
    now = time.time()
    ids = {}
    for name, status in [("a", "dismissed"), ("b", "dismissed"), ("c", "confirmed"), ("d", "pending")]:
        al = Alert(level="orange", reasons=["x"], zone_id="kantin", created_at=now)
        db.save_alert(al)
        db.update_alert_status(al.id, status)
        ids[name] = al.id
    assert db.false_alarm_stats() == {"dismissed": 2, "decided": 3, "total": 4}
    assert db.false_alarm_stats(since_ts=now + 100) == {"dismissed": 0, "decided": 0, "total": 0}


class FakeNotifier:
    def __init__(self):
        self.confirmations = []

    def send_confirmation(self, alert):
        self.confirmations.append(alert["id"])


def make_client():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from api.server import FrameStore, Hub, create_app

    db = Storage(":memory:")
    notifier = FakeNotifier()
    client = TestClient(create_app(load_config(), db, FrameStore(), Hub(), notifier))
    return client, db, notifier


def test_only_red_confirmation_sends_112_message():
    client, db, notifier = make_client()
    orange = Alert(level="orange", reasons=["Kavga şüphesi"], zone_id="kantin")
    red = Alert(level="red", reasons=["Bıçak tespit edildi (%82)"], zone_id="kantin")
    db.save_alert(orange)
    db.save_alert(red)

    r = client.post(f"/api/alerts/{orange.id}/confirm")   # "Gördüm"
    assert r.status_code == 200 and r.json()["status"] == "confirmed"
    assert notifier.confirmations == []                    # turuncu: 112 mesajı GİTMEZ

    client.post(f"/api/alerts/{red.id}/confirm")
    assert notifier.confirmations == [red.id]              # kırmızı: gider


def test_level_endpoint_unavailable_without_audio_and_returns_values_with_it():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from api.server import FrameStore, Hub, create_app

    cfg = load_config()
    none = TestClient(create_app(cfg, Storage(":memory:"), FrameStore(), Hub()))
    assert none.get("/api/level").json() == {"available": False}

    off = TestClient(create_app(cfg, Storage(":memory:"), FrameStore(), Hub(), level_source=lambda: None))
    assert off.get("/api/level").json() == {"available": False}   # ses dedektörü henüz yok

    data = {"db": 61.5, "threshold": 75.0, "scores": {"scream": 0.1}}
    on = TestClient(create_app(cfg, Storage(":memory:"), FrameStore(), Hub(), level_source=lambda: data))
    assert on.get("/api/level").json() == {"available": True, **data}


def test_false_alarm_endpoint():
    client, db, _ = make_client()
    al = Alert(level="yellow", reasons=["x"], zone_id="kantin")
    db.save_alert(al)
    client.post(f"/api/alerts/{al.id}/dismiss")
    assert client.get("/api/summary/false-alarms").json() == {"dismissed": 1, "decided": 1, "total": 1}
