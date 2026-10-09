from core.schema import Alert, Event
from storage.db import Storage


def make():
    return Storage(":memory:")


def test_alert_roundtrip_and_update_keeps_status():
    db = make()
    a = Alert(level="yellow", reasons=["Bağırma sesi (%55)"], zone_id="kantin", created_at=10.0)
    db.save_alert(a)
    db.update_alert_status(a.id, "confirmed")
    a.level, a.reasons = "orange", a.reasons + ["Kavga şüphesi"]  # dedup güncellemesi, aynı id
    db.save_alert(a)
    got = db.get_alert(a.id)
    assert got["level"] == "orange" and got["reasons"] == a.reasons
    assert got["status"] == "confirmed"  # güncelleme status'u ezmez
    assert len(db.list_alerts()) == 1


def test_list_alerts_newest_first_and_limit():
    db = make()
    for i in range(5):
        db.save_alert(Alert(level="yellow", reasons=["x"], zone_id="z", created_at=float(i)))
    out = db.list_alerts(3)
    assert [a["created_at"] for a in out] == [4.0, 3.0, 2.0]


def test_unknown_alert_status_update_returns_none():
    assert make().update_alert_status("yok", "dismissed") is None


def test_weekly_summary_and_heatmap():
    db = make()
    for lvl, zone in [("yellow", "kantin"), ("yellow", "kantin"), ("red", "kantin"), ("orange", "bahce")]:
        db.save_alert(Alert(level=lvl, reasons=["x"], zone_id=zone, created_at=100.0))
    db.save_alert(Alert(level="red", reasons=["eski"], zone_id="bahce", created_at=1.0))
    s = db.weekly_summary(since_ts=50.0)
    assert s["kantin"] == {"yellow": 2, "orange": 0, "red": 1}
    assert s["bahce"] == {"yellow": 0, "orange": 1, "red": 0}
    for z in ["kantin", "kantin", "bahce"]:
        db.save_event(Event(source="pose", type="fight", confidence=0.8, camera_id="cam1", zone_id=z))
    assert db.heatmap() == {"kantin": 2, "bahce": 1}
