# OkulKalkan (geçici ad) — Okul Güvenliği Erken Uyarı Sistemi

## Bağlam
- KTÜ HSD Eğitim Hackathonu, Tema 05: Erken Uyarı ve Takip Sistemleri.
- Süre: ~17 saat. Final sunumu 10 Ekim 09.30. Hedef: **çalışan, demo edilebilir prototip**, mükemmel ürün değil.
- Donanım: öğrenci laptopları (çoğunlukla CPU). Performans her zaman öncelikli.
- Önce basit ve çalışan çözüm; akıllıca ama riskli çözümlerden kaçın.

## Ürün özeti
Mevcut okul kameralarına (RTSP) yazılımla entegre olan, görüntü + ses analizini birleştirip **kademeli uyarı** üreten sistem:
- Silah/bıçak tespiti (görüntü)
- Kavga, itişme, yere düşme, ani koşuşma tespiti (iskelet/poz analizi)
- Çığlık, bağırma, silah sesi, cam kırılması tespiti (ses sınıflandırma + dB)
- Uyarılar şiddete göre Sarı / Turuncu / Kırmızı; kırmızıda insan onayıyla tek tuş 112 (demoda simülasyon)
- Olaylar okul kat planında ısı haritasına işlenir (zorbalık/olay sıcak noktaları)

## Değişmez ilkeler (bunlara aykırı kod yazma)
1. **Yüz tanıma ve duygu/mimik tanıma YOK.** Kimlik tespiti yapılmaz. Davranış analizi sadece iskelet keypoint'leri üzerinden.
2. **Polis bildirimi otomatik değil.** Kırmızı uyarı bir insanın ekranına düşer, o onaylar. (Opsiyonel zaman aşımı eskalasyonu config'de, varsayılan KAPALI.)
3. **Gizlilik:** Dashboard yayınında ve kaydedilen snapshot/kliplerde yüzler bulanıklaştırılır (yüz *tespiti* sadece bulanıklaştırma için). Klip yalnızca turuncu/kırmızı uyarıda kaydedilir.
4. Tüm eşikler ve kurallar `config.yaml`'da; kod içinde sabit eşik yok. Gizli bilgiler `.env`'de.

## Teknoloji
- Python 3.11, type hints
- Görüntü: OpenCV (webcam / RTSP / video dosyası), Ultralytics YOLO (nano modeller)
  - Silah/bıçak: `models/weapon.pt` (Roboflow Universe hazır weapon modeli). **Dosya yoksa fallback:** COCO `yolo11n.pt` ile sadece `knife` sınıfı.
  - Poz: `yolo11n-pose.pt`
- Ses: YAMNet — tercihen MediaPipe Tasks `AudioClassifier` + `yamnet.tflite` (TensorFlow kurulumu ağır). Sınıflar: Screaming, Shout, Yell, Gunshot/gunfire, Glass/Shatter. Ek olarak RMS → dB.
- Backend: FastAPI + WebSocket
- Frontend: tek sayfa, vanilla HTML/JS/CSS (framework yok)
- Depolama: SQLite (`storage/events.db`)
- Bildirim: Telegram Bot API (token `.env`)

## Klasör yapısı
```
config.yaml
.env.example
requirements.txt
main.py                 # tüm pipeline'ı başlatır (--source, --audio, --demo)
sources/camera.py       # webcam / RTSP / video dosyası
sources/audio.py        # mikrofon / wav dosyası
detectors/weapon.py
detectors/pose.py       # kavga, düşme, koşuşma kuralları
detectors/audio_cls.py
fusion/engine.py        # event → alert, eskalasyon, dedup
fusion/rules.py
privacy/blur.py
api/server.py           # REST + WebSocket
notify/telegram.py
storage/db.py
web/index.html, app.js, style.css, floorplan.png
models/                 # ağırlıklar (git'e eklenmez)
test_media/             # demo video ve ses dosyaları
```
Her detektör `if __name__ == "__main__":` ile tek başına webcam/mikrofon üzerinde çalıştırılabilir ve konsola event basar.

## Veri şemaları
**Event** (detektörlerden fusion'a, `queue.Queue` üzerinden):
```json
{"source": "weapon|pose|audio", "type": "gun|knife|fight|fall|running|scream|shout|gunshot|glass",
 "confidence": 0.0, "camera_id": "cam1", "zone_id": "kantin", "ts": 0.0, "snapshot": "path|null"}
```
**Alert** (fusion'dan API/DB/Telegram'a):
```json
{"id": "uuid", "level": "yellow|orange|red", "reasons": ["..."], "zone_id": "kantin",
 "created_at": 0.0, "status": "pending|confirmed|dismissed", "snapshot": "path", "clip": "path|null"}
```

## Uyarı kuralları (varsayılan değerler, config'den ayarlanır)
**KIRMIZI**
- Silah/bıçak: conf ≥ 0.6, son 5 karenin en az 3'ünde
- Ses: gunshot conf ≥ 0.5
- Aksiyon: dashboard'da tam ekran modal + snapshot → [Onayla: 112'yi ara (simülasyon) + tüm öğretmenlere "sınıfta kalın" mesajı] / [Yanlış alarm]

**TURUNCU**
- Kavga şüphesi: iki iskelet yakın (merkezler arası < kişi boyunun ~0.6'sı) + bilek/dirsek hızı yüksek, ≥ 1 sn sürerse
- Yere düşme: gövde yatay / kalça keypoint'leri zemin seviyesinde, ≥ 3 sn
- Çığlık + aynı bölgede ≥ 4 kişinin kümelenmesi
- Aksiyon: nöbetçi + müdür yrd. + rehber öğretmene bildirim, 10 sn klip

**SARI**
- Shout/Yell sınıfı veya dB eşiği aşımı (eşik bölge tipine ve saate göre: teneffüs/spor salonu yüksek, ders saati düşük)
- Ani koşuşma: ortalama keypoint hızı eşik üstü
- Aksiyon: nöbetçi öğretmene bildirim + ısı haritasına kayıt

**Genel**
- Eskalasyon: aynı bölgede 120 sn içinde 3 sarı → turuncu
- Dedup: aynı bölge + aynı tip 30 sn içinde tek uyarıda birleşir
- Zaman bağlamı: `config.yaml` içinde ders/teneffüs çizelgesi; kamera → bölge eşlemesi ve bölge tipi

## Dashboard gereksinimleri
- Canlı kamera görüntüsü (yüzler bulanık), algılanan kutular/iskeletler overlay
- Renkli, zaman sıralı uyarı akışı (WebSocket ile anlık)
- Kırmızı uyarı modalı (onay / yanlış alarm), sesli alarm
- Kat planı üzerinde bölge poligonları (`config.yaml`) + olay yoğunluğu ısı haritası
- Basit haftalık özet: bölgelere göre sarı/turuncu/kırmızı sayıları

## Performans
- Nano modeller, `imgsz` 480–640, işleme 5–10 FPS
- Poz analizi her 2–3 karede bir; silah tespiti her karede
- Config'de `device: cpu|cuda`
- Kamera, ses, fusion ve API ayrı thread'lerde; aralarında kuyruk

## Demo modu
- `python main.py --source test_media/kavga.mp4 --audio test_media/ciglik.wav`
- `--demo`: hazır senaryoyu sırayla oynatır (sarı → turuncu → kırmızı)
- 112 araması asla gerçek değil; modalda "112 arandı (simülasyon)" gösterilir
- Demo internetsiz çalışmalı (modeller yerelde)

## Kilometre taşları
- **M1 (17.00):** Proje iskeleti, requirements, config. Üç detektör de tek başına webcam/mikrofonda çalışıp event basıyor.
- **M2 (22.00):** Fusion motoru + FastAPI/WebSocket + canlı uyarı akışı olan dashboard.
- **M3 (02.00):** Kırmızı onay akışı, Telegram, yüz bulanıklaştırma, ısı haritası, eskalasyon/dedup.
- **M4 (06.30):** Eşik ayarı, test videolarıyla demo senaryosu, README. 06.30'da code freeze.

## Kapsam dışı
Yüz/kimlik tanıma, duygu tanıma, gerçek 112/polis entegrasyonu, bulut dağıtımı, kullanıcı girişi/hesap sistemi, mobil uygulama.

## Çalışma şekli
- Her adımdan sonra nasıl test edileceğini kısa yaz.
- Bağımlılıkları minimum tut; yeni paket eklemeden önce gerekçesini söyle.
- Bir şey çalışmıyorsa önce basit fallback'e geç, demoyu riske atma.
