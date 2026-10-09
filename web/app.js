"use strict";
// Tek sayfa, framework yok, CDN yok (demo internetsiz çalışmalı).

const LEVEL_TR = { yellow: "Sarı", orange: "Turuncu", red: "Kırmızı" };
const STATUS_TR = { pending: "Bekliyor", confirmed: "Onaylandı", dismissed: "Yanlış alarm" };

const alerts = new Map();   // id -> alert
const listEl = document.getElementById("alerts");
const connEl = document.getElementById("conn");
const countEl = document.getElementById("count");
let zones = {};

function zoneName(id) { return (zones[id] && zones[id].name) || id; }

function fmtTime(ts) {
  return new Date(ts * 1000).toLocaleTimeString("tr-TR");
}

function render(freshId) {
  const sorted = [...alerts.values()].sort((a, b) => b.created_at - a.created_at);
  countEl.textContent = "(" + sorted.length + ")";
  listEl.replaceChildren(...sorted.map((a) => {
    const li = document.createElement("li");
    li.className = "alert " + a.level + (a.id === freshId ? " fresh" : "");

    const top = document.createElement("div");
    top.className = "top";
    const lvl = document.createElement("span");
    lvl.className = "lvl";
    lvl.textContent = LEVEL_TR[a.level] + " · " + zoneName(a.zone_id);
    const time = document.createElement("span");
    time.textContent = fmtTime(a.created_at);
    top.append(lvl, time);

    const reasons = document.createElement("ul");
    for (const r of a.reasons) {
      const item = document.createElement("li");
      item.textContent = r;   // textContent: sebep metni asla HTML olarak yorumlanmaz
      reasons.append(item);
    }

    const status = document.createElement("div");
    status.className = "status";
    status.textContent = STATUS_TR[a.status] || a.status;

    li.append(top, reasons, status);
    return li;
  }));
}

function upsert(alert, fresh) {
  alerts.set(alert.id, alert);
  render(fresh ? alert.id : null);
  updateModal();
}

// ---------- Kırmızı uyarı modalı + sesli alarm ----------
const modal = document.getElementById("redModal");
const alarm = document.getElementById("alarm");
const soundBtn = document.getElementById("soundBtn");
let soundOn = false;
let resultFor = null;   // onaylandıktan sonra sonuç ekranı gösterilen uyarı id'si

soundBtn.addEventListener("click", () => {
  // Tarayıcılar sesi ancak kullanıcı tıklamasından sonra çalmaya izin verir: bu buton o tıklama.
  soundOn = !soundOn;
  soundBtn.className = "sound " + (soundOn ? "on" : "off");
  soundBtn.textContent = soundOn ? "Alarm sesi açık" : "Alarm sesi kapalı (açmak için tıkla)";
  updateModal();
});

function pendingReds() {
  return [...alerts.values()]
    .filter((a) => a.level === "red" && a.status === "pending")
    .sort((a, b) => a.created_at - b.created_at);
}

function setAlarm(play) {
  if (play && soundOn) { alarm.play().catch(() => {}); }
  else { alarm.pause(); alarm.currentTime = 0; }
}

function showModalFor(a) {
  document.getElementById("redZone").textContent = zoneName(a.zone_id);
  const snap = document.getElementById("redSnap");
  // Snapshot yoksa canlı yayının görüntüsü gösterilir
  snap.src = a.snapshot ? "/snapshots/" + a.snapshot.split(/[\\/]/).pop() : "/video/cam1";
  document.getElementById("redReasons").replaceChildren(...a.reasons.map((r) => {
    const li = document.createElement("li");
    li.textContent = r;
    return li;
  }));
  document.getElementById("redActions").hidden = false;
  document.getElementById("redResult").hidden = true;
  document.getElementById("confirmBtn").disabled = false;
  document.getElementById("dismissBtn").disabled = false;
}

function updateModal() {
  if (resultFor) return;   // sonuç ekranı açıkken dokunma
  const reds = pendingReds();
  if (reds.length === 0) {
    modal.hidden = true;
    setAlarm(false);
    return;
  }
  const current = reds[0];
  if (modal.hidden || modal.dataset.id !== current.id) {
    modal.dataset.id = current.id;
    showModalFor(current);
  } else {   // aynı uyarı güncellendi (yeni sebep eklendi)
    document.getElementById("redReasons").replaceChildren(...current.reasons.map((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      return li;
    }));
  }
  modal.hidden = false;
  setAlarm(true);
}

async function act(kind) {
  const id = modal.dataset.id;
  document.getElementById("confirmBtn").disabled = true;
  document.getElementById("dismissBtn").disabled = true;
  // WebSocket güncellemesi POST yanıtından önce gelebilir; modal kapanmasın diye önceden işaretle
  if (kind === "confirm") resultFor = id;
  try {
    const res = await fetch("/api/alerts/" + id + "/" + kind, { method: "POST" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const updated = await res.json();
    alerts.set(updated.id, updated);
    render(null);
    if (kind === "confirm") {
      setAlarm(false);
      modal.hidden = false;
      document.getElementById("redActions").hidden = true;
      document.getElementById("redResult").hidden = false;
    } else {
      updateModal();
    }
  } catch (e) {
    resultFor = null;
    console.error("işlem başarısız", e);
    document.getElementById("confirmBtn").disabled = false;
    document.getElementById("dismissBtn").disabled = false;
  }
}

document.getElementById("confirmBtn").addEventListener("click", () => act("confirm"));
document.getElementById("dismissBtn").addEventListener("click", () => act("dismiss"));
document.getElementById("closeBtn").addEventListener("click", () => { resultFor = null; updateModal(); });

async function loadInitial() {
  try {
    zones = await (await fetch("/api/zones")).json();
    const list = await (await fetch("/api/alerts")).json();
    for (const a of list) alerts.set(a.id, a);
    render(null);
    updateModal();
  } catch (e) {
    console.error("ilk yükleme başarısız", e);
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(proto + "//" + location.host + "/ws");
  ws.onopen = () => { connEl.textContent = "bağlı"; connEl.className = "conn on"; };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "alert") upsert(msg.alert, true);
  };
  ws.onclose = () => {
    connEl.textContent = "bağlantı yok"; connEl.className = "conn off";
    setTimeout(connect, 1500);   // sunucu yeniden başlarsa kendiliğinden bağlanır
  };
}

loadInitial().then(connect);
