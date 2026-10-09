import copy

from core.config import load_config
from core.schema import Alert
from fusion.escalation import RedTimeoutWatcher
from notify.telegram import Notifier
from storage.db import Storage


def cfg_with(enabled: bool, timeout_s: float = 120):
    cfg = copy.deepcopy(load_config())
    cfg["alerts"]["auto_escalate_red"] = {"enabled": enabled, "timeout_s": timeout_s}
    return cfg


def test_default_config_is_disabled():
    # İLKE: polis bildirimi otomatik değil; varsayılan KAPALI kalmalı
    assert load_config()["alerts"]["auto_escalate_red"]["enabled"] is False


def test_disabled_never_fires():
    w = RedTimeoutWatcher(cfg_with(False))
    w.track("a1", "red", now=0)
    assert w.due(now=10_000, status_of=lambda i: "pending") == []


def test_only_red_is_tracked():
    w = RedTimeoutWatcher(cfg_with(True, 60))
    w.track("y", "yellow", now=0)
    w.track("o", "orange", now=0)
    assert w.due(now=1000, status_of=lambda i: "pending") == []


def test_fires_after_timeout_when_still_pending_and_only_once():
    w = RedTimeoutWatcher(cfg_with(True, 60))
    w.track("a1", "red", now=100)
    assert w.due(now=159, status_of=lambda i: "pending") == []   # süre dolmadı
    assert w.due(now=160, status_of=lambda i: "pending") == ["a1"]
    assert w.due(now=500, status_of=lambda i: "pending") == []   # tekrar tetiklenmez
    w.track("a1", "red", now=600)                                # dedup güncellemesi yeniden izlemez
    assert w.due(now=1000, status_of=lambda i: "pending") == []


def test_confirmed_or_dismissed_alert_is_not_escalated():
    for decided in ("confirmed", "dismissed"):
        w = RedTimeoutWatcher(cfg_with(True, 60))
        w.track("a1", "red", now=0)
        assert w.due(now=1000, status_of=lambda i, s=decided: s) == []


def test_unknown_alert_dropped():
    w = RedTimeoutWatcher(cfg_with(True, 60))
    w.track("gone", "red", now=0)
    assert w.due(now=1000, status_of=lambda i: None) == []


def test_append_reason_persists_and_is_idempotent():
    db = Storage(":memory:")
    a = Alert(level="red", reasons=["Bıçak tespit edildi (%82)"], zone_id="kantin", created_at=1.0)
    db.save_alert(a)
    db.append_reason(a.id, "Onay bekleniyor: 60 sn geçti (ek bildirim gönderildi)")
    db.append_reason(a.id, "Onay bekleniyor: 60 sn geçti (ek bildirim gönderildi)")
    got = db.get_alert(a.id)
    assert len(got["reasons"]) == 2 and got["status"] == "pending"
    assert db.append_reason("yok", "x") is None


def test_unconfirmed_message_is_notification_only():
    n = Notifier(cfg_with(False))
    text = n.format_unconfirmed({"zone_id": "kantin"}, 120)
    assert "ONAYLANMADI" in text and "Kantin" in text
    assert "yapılmaz" in text  # otomatik arama / polis bildirimi yok
