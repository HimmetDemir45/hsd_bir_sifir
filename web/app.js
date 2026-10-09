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

function clipUrl(a) {
  return a.clip ? "/clips/" + encodeURIComponent(a.clip.split(/[\\/]/).pop()) : null;
}

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

    li.append(top, reasons);
    const url = clipUrl(a);
    if (url) {   // klip turuncu/kırmızıda, kaydedilince (birkaç sn sonra) gelir
      const link = document.createElement("a");
      link.className = "clip-link";
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "▶ Klibi izle";
      li.append(link);
    }
    li.append(status);
    return li;
  }));
}

function upsert(alert, fresh) {
  alerts.set(alert.id, alert);
  render(fresh ? alert.id : null);
  updateModal();
  scheduleStats();
}

// ---------- Kat planı ısı haritası + haftalık özet ----------
const planCanvas = document.getElementById("plan");
const planCtx = planCanvas.getContext("2d");
const floorplan = new Image();
let floorplanReady = false;
floorplan.onload = () => { floorplanReady = true; refreshStats(); };
floorplan.src = "/floorplan.png";
let statsTimer = null;

function scheduleStats() {   // uyarı yağmurunda her seferinde değil, en fazla ~1 sn'de bir
  if (statsTimer) return;
  statsTimer = setTimeout(() => { statsTimer = null; refreshStats(); }, 1000);
}

function drawHeatmap(counts) {
  planCtx.clearRect(0, 0, planCanvas.width, planCanvas.height);
  if (floorplanReady) planCtx.drawImage(floorplan, 0, 0, planCanvas.width, planCanvas.height);
  const max = Math.max(1, ...Object.values(counts));
  for (const [id, z] of Object.entries(zones)) {
    const n = counts[id] || 0;
    const t = n / max;                       // 0..1 yoğunluk
    planCtx.beginPath();
    z.polygon.forEach(([x, y], i) => (i ? planCtx.lineTo(x, y) : planCtx.moveTo(x, y)));
    planCtx.closePath();
    if (n > 0) {
      // sarıdan kırmızıya: yoğunluk arttıkça hem renk hem opaklık artar
      const hue = 55 - 55 * t;
      planCtx.fillStyle = "hsla(" + hue + ", 95%, 50%, " + (0.25 + 0.55 * t) + ")";
      planCtx.fill();
    }
    planCtx.lineWidth = 2;
    planCtx.strokeStyle = "#555";
    planCtx.stroke();
    const xs = z.polygon.map((p) => p[0]), ys = z.polygon.map((p) => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    planCtx.fillStyle = "#222";
    planCtx.textAlign = "center";
    planCtx.font = "bold 15px system-ui, sans-serif";
    planCtx.fillText(z.name, cx, cy - 4);
    planCtx.font = "13px system-ui, sans-serif";
    planCtx.fillText(n + " olay", cx, cy + 14);
  }
}

function renderWeekly(summary) {
  const body = document.querySelector("#weekly tbody");
  const rows = Object.entries(summary);
  if (rows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4; td.className = "empty"; td.textContent = "Henüz uyarı yok";
    tr.append(td);
    body.replaceChildren(tr);
    return;
  }
  body.replaceChildren(...rows.map(([zone, c]) => {
    const tr = document.createElement("tr");
    for (const text of [zoneName(zone), c.yellow, c.orange, c.red]) {
      const td = document.createElement("td");
      td.textContent = String(text);
      tr.append(td);
    }
    return tr;
  }));
}

async function refreshStats() {
  try {
    const [counts, summary] = await Promise.all([
      fetch("/api/heatmap").then((r) => r.json()),
      fetch("/api/summary/weekly").then((r) => r.json()),
    ]);
    drawHeatmap(counts);
    renderWeekly(summary);
  } catch (e) {
    console.error("istatistikler yüklenemedi", e);
  }
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

function setModalClip(a) {
  const link = document.getElementById("redClip");
  const url = clipUrl(a);
  link.hidden = !url;
  if (url) link.href = url;
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
  setModalClip(a);
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
  } else {   // aynı uyarı güncellendi (yeni sebep eklendi / klip hazır oldu)
    setModalClip(current);
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
    refreshStats();
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
