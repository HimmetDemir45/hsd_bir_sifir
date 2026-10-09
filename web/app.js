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
}

async function loadInitial() {
  try {
    zones = await (await fetch("/api/zones")).json();
    const list = await (await fetch("/api/alerts")).json();
    for (const a of list) alerts.set(a.id, a);
    render(null);
  } catch (e) {
    console.error("ilk yükleme başarısız", e);
  }
}

// Canlı görüntü (MJPEG) sunucu yeniden başlarsa kendiliğinden bağlanmaz: yeniden bağlan
const camEl = document.getElementById("cam");
function reconnectVideo() {
  camEl.src = "/video/cam1?t=" + Date.now();   // önbellek kırıcı: tarayıcı yeni bağlantı açar
}
camEl.onerror = () => setTimeout(reconnectVideo, 1500);

let wasDisconnected = false;
function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(proto + "//" + location.host + "/ws");
  ws.onopen = () => {
    connEl.textContent = "bağlı"; connEl.className = "conn on";
    if (wasDisconnected) { reconnectVideo(); wasDisconnected = false; }
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "alert") upsert(msg.alert, true);
  };
  ws.onclose = () => {
    connEl.textContent = "bağlantı yok"; connEl.className = "conn off";
    wasDisconnected = true;
    setTimeout(connect, 1500);   // sunucu yeniden başlarsa kendiliğinden bağlanır
  };
}

loadInitial().then(connect);
