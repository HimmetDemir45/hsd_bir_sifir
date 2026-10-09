import requests

from core.config import load_config
from notify.telegram import Notifier

ALERT = {"id": "a1", "level": "red", "reasons": ["Bıçak tespit edildi (%82)"],
         "zone_id": "kantin", "snapshot": None}


def cfg_with(enabled: bool):
    cfg = load_config()
    cfg["telegram"] = {"enabled": enabled, "bot_token": "T", "chat_id": "42"}
    return cfg


def test_disabled_never_calls_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("ağ çağrısı olmamalı")
    monkeypatch.setattr(requests, "post", boom)
    n = Notifier(cfg_with(False), async_send=False)
    n.send_alert(ALERT)
    n.send_confirmation(ALERT)


def test_enabled_sends_message_with_roles_and_dedups(monkeypatch):
    calls = []

    class R:
        ok = True

    monkeypatch.setattr(requests, "post", lambda url, **k: calls.append((url, k)) or R())
    n = Notifier(cfg_with(True), async_send=False)
    n.send_alert(ALERT)
    n.send_alert(ALERT)  # aynı id + seviye: tekrar gitmez
    assert len(calls) == 1
    url, kw = calls[0]
    assert url.endswith("/botT/sendMessage")
    assert "Kantin" in kw["data"]["text"] and "Okul müdürü" in kw["data"]["text"]
    n.send_alert({**ALERT, "level": "red", "id": "a2"})
    assert len(calls) == 2


def test_network_error_does_not_raise(monkeypatch):
    def fail(*a, **k):
        raise requests.ConnectionError("internet yok")
    monkeypatch.setattr(requests, "post", fail)
    n = Notifier(cfg_with(True), async_send=False)
    n.send_alert(ALERT)  # istisna fırlatmamalı
    n.send_confirmation(ALERT)


def test_token_never_appears_in_logs(monkeypatch, capsys):
    secret = "123456:SECRET-TOKEN-VALUE"

    def fail(url, **k):  # requests, hata metnine URL'yi (token dahil) koyar
        raise requests.ConnectionError(f"HTTPSConnectionPool: url: {url}")

    class Bad:
        ok = False
        status_code = 401
        text = f"unauthorized for {secret}"

    cfg = cfg_with(True)
    cfg["telegram"]["bot_token"] = secret
    n = Notifier(cfg, async_send=False)
    monkeypatch.setattr(requests, "post", fail)
    n.send_alert(ALERT)
    monkeypatch.setattr(requests, "post", lambda url, **k: Bad())
    n.send_alert({**ALERT, "id": "a9"})
    out = capsys.readouterr().out
    assert secret not in out and "<token>" in out


def test_confirmation_text_says_simulation():
    n = Notifier(cfg_with(False))
    t = n.format_confirmation(ALERT)
    assert "112 arandı (simülasyon)" in t and "sınıfta kalın" in t
