# Kişi B: Füzyon + Backend + Dashboard görev dosyası

> **Claude Code için:** Önce `CLAUDE.md`'yi, sonra bu dosyayı baştan sona oku. Sen **Kişi B**'nin asistanısın.
> Sadece aşağıdaki "Sorumluluk alanı"ndaki dosyalara dokun. Her adımdan sonra kullanıcıya nasıl test
> edeceğini kısa yaz. M1'den başla.

## 0. Kurulum (bir kez)

Windows, Python 3.11 gerekli (`py -0` ile kontrol et; yoksa python.org'dan 3.11 kur).

```bash
git clone https://github.com/HimmetDemir45/hsd_bir_sifir.git
cd hsd_bir_sifir
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
git checkout -b fuzyon-web
```

Kurulum testi:
```bash
.venv\Scripts\python -c "from core.schema import Event, Alert; from core.config import load_config; print(load_config()['alerts'])"
```
Hata vermeden `alerts` ayarlarını basmalı.

Kişi A'nın detektörlerini denemek istersen (opsiyonel, webcam açar):
```bash
.venv\Scripts\python -m detectors.weapon
.venv\Scripts\python -m detectors.pose
```
İlk çalıştırmada `models/yolo11n.pt` ve `models/yolo11n-pose.pt` otomatik iner (internet gerekir).

## 1. Sorumluluk alanı

| Kişi B (sen) dokunur | Kişi A dokunur | Ortak (değiştirmeden önce haber ver) |
|---|---|---|
| `fusion/`, `api/`, `storage/`, `notify/`, `web/`, `main.py`, `tests/test_fusion*.py` | `sources/`, `detectors/`, `privacy/`, `models/`, `test_media/` | `core/schema.py` (**DONDURULDU**), `config.yaml`, `requirements.txt`, `README.md`, `CLAUDE.md` |

`config.yaml`'da B'nin bölümleri: `alerts`, `recipients`, `zones`, `schedule`, `api`, `telegram`, `privacy.record_clips_for`.
A'nın bölümleri: `general`, `weapon`, `pose`, `audio`, `cameras`. Başkasının bölümünü düzenleme.

Yeni paket eklemeden önce gerekçesini söyle ve A'ya haber ver (`requirements.txt` ortak).

## 2. A ile arayüz (sözleşme)

### Event (A → B), `core/schema.py`
```python
from core.schema import Event, Alert
# Event(source="weapon|pose|audio", type="gun|knife|fight|fall|running|scream|shout|gunshot|glass",
#       confidence=float, camera_id="cam1", zone_id="kantin", ts=time.time(), snapshot=None)
```
- Detektörler event'leri verilen `queue.Queue`'ya `put` eder. B bu kuyruktan `get` eder.
- Detektör aynı tipi kısa sürede tekrar basmaz (weapon 2 sn, pose 3 sn). Asıl dedup (30 sn) ve eskalasyon **fusion'ın işi**.
- `snapshot` M3'e kadar `None`. M3'te A, yüzü bulanıklaştırılmış snapshot yolunu dolduracak.
- `ts`: saniye, `time.time()` ekseni (video dosyasında da aynı eksen).

### A'nın verdiği sınıflar (main.py'de kullanacaksın)
```python
from sources.camera import Camera                 # Camera(source).read() -> frame | None ; .last_ts ; .release()
from detectors.weapon import WeaponDetector       # WeaponDetector(cfg, camera_id, zone_id, out_queue)
from detectors.pose import PoseDetector           # PoseDetector(cfg, camera_id, zone_id, out_queue)
# ikisi de: dets, events = det.process(frame, camera.last_ts); det.draw(frame, dets)  -> overlay
# PoseDetector.status.crowd_size -> en kalabalık kümedeki kişi sayısı
```
A'nın yapacakları (henüz yok, gelene kadar sahte veri kullan):
- `detectors/audio_cls.py`: `AudioDetector(cfg, zone_id, out_queue)`, `.run(stop_event: threading.Event)` (thread'de çalışır), `.last_db: float` (dashboard'da ses seviyesi için).
- `privacy/blur.py`: `blur_faces(frame) -> frame`. Dashboard yayınından ve kayıttan **önce** çağır.
- `sources/clip_buffer.py`: `ClipBuffer(seconds)`, `.add(frame, ts)`, `.save(path) -> str` (son N sn'yi mp4 yazar, kareler zaten bulanık). Turuncu/kırmızı uyarıda sen çağırırsın.

### Açık konu: kalabalık (A ile karar verin)
"Çığlık + aynı bölgede ≥4 kişi" kuralı için kalabalık bilgisi lazım ama Event tiplerinde `crowd` yok.
Seçenekler: (a) şemaya `crowd` tipi eklenir, pose kalabalıkta event basar; (b) fusion `PoseDetector.status.crowd_size`'ı doğrudan okur.
Karar verilene kadar fusion'da bunu bir fonksiyon parametresi (`crowd_size_by_zone: dict[str, int]`) olarak al.

## 3. Değişmez ilkeler (CLAUDE.md'den, bunlara aykırı kod yazma)
- **112 / polis bildirimi otomatik değil.** Kırmızı uyarı ekrana düşer, insan onaylar. Onayda sadece "112 arandı (simülasyon)" gösterilir; gerçek arama **asla** yok.
- `alerts.auto_escalate_red.enabled` varsayılan **false**.
- Tüm eşikler `config.yaml`'da; kodda sabit eşik yok. Telegram token `.env`'de.
- Dashboard'da yüzler bulanık (A'nın `blur_faces`'i). Klip sadece turuncu/kırmızıda.
- **Demo internetsiz çalışmalı:** dashboard'da CDN kullanma (font, JS kütüphanesi vb. dahil); her şey `web/` içinde.
- Frontend: tek sayfa, vanilla HTML/JS/CSS, framework yok.

## 4. Görevler

### M1 (17.00): Fusion motoru + sahte event üreteci
1. **`fusion/rules.py`**: Saf fonksiyonlar (I/O yok, test edilebilir). Event'ten seviye üretimi:
   - KIRMIZI: `gun`/`knife` conf ≥ `alerts.red.weapon_min_conf`; `gunshot` conf ≥ `alerts.red.gunshot_min_conf`
   - TURUNCU: `fight`, `fall`; `scream` + aynı bölgede kalabalık ≥ `alerts.orange.scream_crowd_min_people`
   - SARI: `shout`, `running`, `scream` (kalabalık yoksa), `glass` (A ile netleştirin; şimdilik turuncu da olabilir)
   - Zaman bağlamı: `schedule.breaks` ile "teneffüs mü ders mi" fonksiyonu (dB eşiği seçimi için A'ya da lazım olacak, `is_break(now, cfg)`).
2. **`fusion/engine.py`**: `FusionEngine(cfg, in_queue, on_alert: Callable[[Alert], None])`, `.run(stop_event)`.
   - Dedup: aynı bölge + aynı tip `alerts.dedup_window_s` içinde tek uyarı (yeni event'in sebebini mevcut uyarının `reasons`'ına ekle, seviye yükseldiyse yükselt).
   - Eskalasyon: aynı bölgede `alerts.escalation.window_s` içinde `yellow_count` sarı → turuncu.
   - `reasons` kullanıcıya gösterilecek Türkçe metin: `"Bıçak tespit edildi (%82)"`, `"Kavga şüphesi"` vb.
3. **`fusion/demo.py`**: Sahte event üreteci. `--demo` senaryosu: sarı → turuncu → kırmızı sırasıyla, gerçekçi aralıklarla (ör. 3 bağırma → eskalasyonla turuncu → kavga → bıçak). Gerçek detektörler kırılsa bile demo bununla çalışmalı.
4. **`tests/test_fusion.py`**: dedup, eskalasyon ve kırmızı kuralı için birkaç test (zamanı parametre olarak ver, `time.sleep` kullanma).

Test: `python -m fusion.demo` (konsola Alert JSON'ları basmalı), `python -m pytest tests -q`
(pytest sadece geliştirme için; yoksa `pip install pytest`, requirements'a ekleme).

### M2 (22.00): Backend + canlı dashboard
1. **`storage/db.py`**: SQLite `storage/events.db`. Tablolar: `events`, `alerts` (şemadaki alanlar, `reasons` JSON metin). Fonksiyonlar: `save_event`, `save_alert`, `update_alert_status`, `list_alerts(limit)`, `weekly_summary()` (bölge × seviye sayıları), `heatmap()` (bölge → olay sayısı). Thread'ler arası kullanım için `check_same_thread=False` + kilit.
2. **`api/server.py`**: FastAPI.
   - `GET /` → `web/index.html`, statik dosyalar `web/`
   - `GET /api/alerts`, `POST /api/alerts/{id}/confirm`, `POST /api/alerts/{id}/dismiss`
   - `GET /api/summary/weekly`, `GET /api/heatmap`, `GET /api/zones` (config'deki poligonlar)
   - `WS /ws`: yeni/güncellenen alert'leri anlık iter
   - `GET /video/{camera_id}`: MJPEG yayın (`multipart/x-mixed-replace`), main.py'nin tuttuğu son overlay'li kareden
3. **`main.py`**: Her şeyi thread'lerde başlatır. `--source`, `--audio`, `--demo` argümanları.
   - Kamera thread'i: `Camera` → `WeaponDetector.process` (her kare) + `PoseDetector.process` → `draw` → (M3: `blur_faces`) → son JPEG'i paylaşılan değişkende tut.
   - Fusion thread'i, API (uvicorn) thread'i. Thread'ler arası sadece kuyruk / kilitli değişken.
   - `--demo`: kamera yerine `fusion/demo.py`'den event besler.
4. **`web/index.html`, `app.js`, `style.css`**: canlı görüntü (`<img src="/video/cam1">`), renkli zaman sıralı uyarı akışı (WebSocket).

Test: `python main.py --demo` → tarayıcıda `http://localhost:8000` → uyarılar sırayla akmalı. `python main.py` → webcam görüntüsü + kutular/iskeletler görünmeli.

### M3 (02.00): Kırmızı akış, Telegram, ısı haritası
1. Kırmızı uyarıda tam ekran modal + snapshot + sesli alarm (ses dosyası `web/` içinde, internetsiz).
   - **[Onayla]** → status `confirmed`, modalda "112 arandı (simülasyon)" + "Tüm öğretmenlere 'sınıfta kalın' mesajı gönderildi" + Telegram'a mesaj.
   - **[Yanlış alarm]** → status `dismissed`.
2. **`notify/telegram.py`**: `requests` ile Bot API `sendMessage` / `sendPhoto`. Token ve chat id `.env`'den; `telegram.enabled: false` ise sadece konsola yaz. İnternet yoksa hata verip pipeline'ı **durdurmasın** (try/except + log).
   Seviyeye göre alıcılar `recipients`'tan (demoda hepsi aynı chat'e, mesajda rol adı yazsın).
3. Kat planı: `web/floorplan.png` (basit çizim yeterli) üzerinde `zones` poligonları + olay sayısına göre renk yoğunluğu (canvas).
4. Turuncu/kırmızıda `ClipBuffer.save(...)` ile 10 sn klip, alert'in `clip` alanına yolu yaz.
5. Haftalık özet tablosu (bölge × sarı/turuncu/kırmızı).

Test: `--demo` ile kırmızıya kadar git → modal + alarm sesi → Onayla → Telegram mesajı (veya konsol logu) + DB'de status `confirmed`.

### M4 (06.30): Cila
- Demo senaryosunu A'nın test videolarıyla birlikte prova et, README'ye "Demo nasıl çalıştırılır" bölümü.
- 06.30 **code freeze**: sonrasında sadece kritik hata düzeltmesi.

## 5. Git akışı

- Kendi dalında çalış: `fuzyon-web`. A `goruntu-ses` dalında.
- Küçük ve sık commit at (her çalışan adımda bir):
  ```bash
  git add fusion/ tests/
  git commit -m "fusion: dedup ve eskalasyon kuralları"
  git push -u origin fuzyon-web      # ilk seferde -u, sonra sadece git push
  ```
  `git add .` kullanma; sadece kendi klasörlerini ekle (yanlışlıkla `.env`, model, db eklenmesin; `.gitignore` bunları zaten dışlar ama dikkat).
- Her kilometre taşında main'e birleştirme (önce A ile konuşun, sırayla yapın):
  ```bash
  git fetch origin
  git merge origin/main              # önce main'deki yenilikleri al, çakışma varsa çöz, test et
  git push
  git checkout main
  git pull
  git merge fuzyon-web
  git push
  git checkout fuzyon-web
  ```
  Çakışma sadece ortak dosyalarda (`config.yaml`, `requirements.txt`, `README.md`) çıkmalı. `config.yaml` çakışmasında iki tarafın bölümlerini de koru.
- A'nın son kodunu (ör. yeni ses detektörü) almak için: `git fetch origin && git merge origin/main`.

## 6. Çalışma şekli
- Önce basit ve çalışan çözüm. Bir şey çalışmıyorsa basit fallback'e geç, demoyu riske atma.
- Her adımdan sonra kullanıcıya nasıl test edileceğini kısa yaz.
- `core/schema.py`'yi değiştirmen gerekirse **önce A'ya sor**.
