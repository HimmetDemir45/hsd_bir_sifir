# OkulKalkan — Teknik Özet (sunum için)

Mevcut okul kameralarına ve mikrofonuna **yazılımla** bağlanan, görüntü + sesi birlikte analiz edip
**Sarı / Turuncu / Kırmızı** kademeli uyarı üreten erken uyarı sistemi. Yüz tanıma yok, polise otomatik bildirim yok:
son karar her zaman bir insanda.

Ayrıntılı ölçümler: [DEGERLENDIRME.md](DEGERLENDIRME.md) · Mimari: [MIMARI.md](MIMARI.md) · Demo akışı: [DEMO.md](DEMO.md)

---

## 1. Teknoloji yığını

| Katman | Teknoloji | Neden |
|---|---|---|
| Dil | Python 3.11 (type hints) | Hızlı geliştirme, ML ekosistemi |
| Görüntü okuma | OpenCV 4.14 | Webcam, RTSP ve video dosyası tek arayüzde |
| Nesne tespiti (silah/bıçak) | Ultralytics **YOLO11n**, kendi eğittiğimiz `ml/silah_v2.pt` | Nano model: öğrenci laptopunda CPU'da gerçek zamanlı |
| İskelet (poz) | Ultralytics **YOLO11n-pose** (17 keypoint) | Davranışı yüzsüz, yalnız iskeletten çıkarmak için |
| Kavga sınıflandırıcı | **scikit-learn** HistGradientBoosting (`ml/kavga_model.joblib`) | Küçük veri, hızlı eğitim, CPU'da <1 ms tahmin |
| Ses sınıflandırma | **EfficientAT mn10_as** (PyTorch, AudioSet mAP 0.47) + **YAMNet** (MediaPipe Tasks, TFLite) | TensorFlow kurmadan; silah sesinde YAMNet daha iyi |
| Ses okuma | sounddevice, soundfile | Mikrofon ve wav |
| Yüz bulanıklaştırma | Poz keypoint'leri + OpenCV **YuNet** (ONNX), yedek Haar cascade | Yalnız yüzün yerini bulup pikselleştirmek için |
| Backend | **FastAPI** + Uvicorn, **WebSocket**, MJPEG yayın | Anlık uyarı akışı, tek süreç |
| Depolama | **SQLite** (`storage/events.db`) | Kurulumsuz, dosya tabanlı |
| Bildirim | **Telegram Bot API** (requests) | Rol bazlı, token `.env`'de |
| Frontend | Vanilla HTML/JS/CSS (framework yok, CDN yok) | İnternetsiz çalışır |
| Eğitim | PyTorch CUDA, RTX 4060 (ayrı ortam) | Silah modeli GPU'da ~40 dk |
| Test | pytest (70 test, model dosyası gerektirmez) | Kurallar sahte model/iskeletle test edilir |

Tüm eşikler ve kurallar `config.yaml`'da (kodda sabit eşik yok), gizli bilgiler `.env`'de.

---

## 2. Veri akışı ve mimari

```
Kamera/RTSP/video ─► kare küçültme ─► Silah/bıçak (YOLO11n, her kare)      ┐
  sources/camera.py                   Poz (YOLO11n-pose, 2 karede bir)     ├─► Event ─► queue ─► FUSION ─► Alert
                                       ├ kavga: öğrenilmiş model + temas     │                      fusion/       │
                                       ├ düşme / koşuşma / kalabalık (kural) │                                    │
Mikrofon/wav ─► ses sınıflandırma (EfficientAT + YAMNet) + dB ─► Event ─────┘                                    ▼
                                                                         SQLite · WebSocket · Telegram · Dashboard
Yüz bulanıklaştırma ─► canlı yayın (MJPEG), snapshot, klip  (bulanıklaştırılamayan kare yayına girmez)
```

- **4 ayrı thread**: kamera, ses, fusion, API. Aralarında yalnız `queue.Queue` ya da kilitli değişken; biri çökerse
  diğerleri çalışır.
- **Ortak sözleşme** `core/schema.py`:
  - `Event` (detektör → fusion): `source, type, confidence, camera_id, zone_id, ts`
  - `Alert` (fusion → API/DB/Telegram): `id, level, reasons, zone_id, status, snapshot, clip`

---

## 3. Modüller (kod tarafında ne yaptık)

### 3.1 Kaynaklar (`sources/`)
- `camera.py`: webcam / RTSP / video dosyası.
  - Canlı kaynakta okuma ayrı thread'de, hep son kare tutulur (gecikme birikmez).
  - Kopan RTSP'ye `reconnect_s` ile yeniden bağlanır.
  - Video dosyası **gerçek zamanlı** oynatılır (CPU yetişemezse kare atlanır): canlı kamerayla aynı davranış.
- `audio.py`: mikrofon / wav, 1 sn pencere, 0.5 sn kaydırma.
- `clip_buffer.py`: olay klibi.
  - Son 10 sn **JPEG halka tamponda** tutulur (ham kareyle ~1 GB, JPEG ile ~20 MB).
  - Uyarıda olay öncesi + sonrası yazılır (VP8 `.webm`, yedek H.264/mp4v).
- `zone_timeline.py`: demo videosunda parça başına bölge (her parça farklı kamera/bölge gibi).

### 3.2 Silah / bıçak (`detectors/weapon.py`)
- Kendi eğittiğimiz YOLO11n.
  - Sınıflar: tabanca, tüfek, bıçak, pala, beyzbol sopası.
  - `class_map` ile `gun` / `knife` event tiplerine eşlenir.
- **Zamansal doğrulama**: son 5 karenin en az 3'ünde, tip başına güven ≥ 0.6. Tek karelik yanlış tespit uyarı üretmez.
- Birden fazla model listesi desteklenir (sırayla çalıştırma); dosyası olmayan model uyarıyla atlanır.

### 3.3 Poz ve davranış (`detectors/pose.py`, `ml/pose_features.py`)
- **Takip**: kişiler kareler arasında merkez mesafesi / boy oranıyla açgözlü eşleştirilir. Hızlar EMA ile yumuşatılır.
- **Ölçek bağımsızlık**: tüm mesafe ve hızlar "kişi boyu" cinsinden (kameraya yakın/uzak fark etmesin).
- **Kavga = öğrenilmiş model + "hızlı temas" kuralı** (ikisinden biri yeterli):
  - **Model girdisi**: son 1.5 sn'lik iskelet penceresi.
  - **Kare başına sinyaller** (10 adet):
    - en yakın çift mesafesi, yaklaşma hızı, kutu çakışması (IoU)
    - **bileğin karşı kişinin kutusuna uzaklığı** (temas)
    - kol hızı, ortalama nokta hızı, ivme, gövde açısı, kişi sayısı
  - **Pencere özeti**: her sinyalin ortalama / max / min / std'si + eşik üstü oranlar → **46 özellik**.
  - **Aynı kod**: eğitim ve canlı sistem aynı fonksiyonu (`window_features`) kullanır, eğitim/canlı uyuşmazlığı yok.
  - **Temas kuralı**: aynı karede bilek karşı kişide (≤ 0.1 boy) VE kol hızı ≥ 1.2 boy/sn; 3 sn içinde 2 pencerede
    görülürse kavga. Model uzak güvenlik kamerasıyla eğitildiği için yakın kamerada yumruk/itmeyi kaçırıyordu; bu kural
    ekibin kendi kaydından ölçülerek eklendi.
  - Model dosyası yoksa / scikit-learn yoksa / sürüm uyumsuzsa sistem **çökmez**, elle yazılmış kurala döner.
- **Düşme**: gövde yatay (omuz + kalça keypoint'leri) ya da bacaklar görünürken kutu en/boy ≥ 1.2. 2 sn sürerse turuncu.
  - Duruş titremesine tolerans var (0.5 sn).
  - "Kafa eğme" yanlış alarmı engellendi: sadece üst gövde görünüyorsa kutu oranına bakılmaz.
- **Koşuşma**: ortalama keypoint hızı ≥ 2 boy/sn, 0.3 sn → sarı.
- **Kalabalık**: 1.5 boy yarıçapında en büyük küme. Çığlık + ≥ 4 kişi → turuncu.

### 3.4 Ses (`detectors/audio_cls.py`)
- **Ana model** EfficientAT mn10_as: 32 kHz, 1 sn pencere sıfırla 10 sn'ye tamamlanır. Model 10 sn kliple eğitildi; kısa
  girişte tüm sınıflar ~1.0 çıkıyordu, bunu bulup düzelttik.
- **Silah sesi** için tip bazında model seçimi: YAMNet. CC0 silah kayıtlarında YAMNet 0.26–0.85, EfficientAT 0.01–0.10 verdi.
- **AudioSet sınıf eşlemesi**:

  | AudioSet sınıfı | Bizim tip |
  |---|---|
  | Screaming | scream |
  | Shout, Yell, Bellow | shout |
  | Gunshot/gunfire, Machine gun, Fusillade | gunshot |
  | Shatter | glass |

  Bilerek dışarıda bırakılanlar:
  - "Children shouting": teneffüste sürekli duyulur.
  - "Glass": kantindeki bardak sesi.
- **dB kuralı**: RMS → dB. Eşik bölge tipine ve saate göre değişir (ders/teneffüs çizelgesi `config.yaml`'da):

  | Bölge tipi | Ders | Teneffüs |
  |---|---|---|
  | Kantin | 75 dB | 90 dB |
  | Spor salonu | 85 dB | 95 dB |

  1 sn aşılırsa sarı.

### 3.5 Gizlilik (`privacy/blur.py`)
- **Yüz tanıma yok**: yüzün yalnız yeri bulunur ve pikselleştirilir; kimlik/özellik çıkarılmaz, saklanmaz.
- İki kaynağın **birleşimi** bulanıklaştırılır (kaçırmak, fazla bulanıklaştırmaktan kötü):
  1. Poz keypoint'lerinden kafa bölgesi: burun/göz/kulak; yoksa omuzların üstü. Yandan/tepeden açılarda yüz dedektöründen iyi.
  2. YuNet yüz dedektörü (pozun kaçırdığı yarım yüzler için).
- Yalnız yüz pikselleştirilir; boyun/gövde açık kalır (ilk sürümde boyun da gidiyordu, kafa kutusu daraltıldı).
- **Fail-closed**: bulanıklaştırma hata verirse kare yayına, snapshot'a ve klibe **hiç girmez**.
- Klip yalnız turuncu/kırmızıda kaydedilir.

### 3.6 Füzyon motoru (`fusion/`)
- `rules.py`: saf fonksiyonlar, event → seviye.

  | Seviye | Tetikleyen |
  |---|---|
  | **Kırmızı** | silah/bıçak ≥ 0.6, silah sesi ≥ 0.4 |
  | **Turuncu** | kavga, düşme, çığlık + kalabalık |
  | **Sarı** | bağırma, yüksek ses, koşuşma, cam kırılması, kalabalıksız çığlık |

- **Dedup**: aynı bölge + aynı tip 30 sn içinde tek kartta birleşir (yeni sebep eklenir).
- **Eskalasyon**: aynı bölgede 120 sn içinde 3 sarı → turuncu.
- **Kırmızı zaman aşımı** (`escalation.py`): varsayılan KAPALI. Açıksa onaylanmayan kırmızı için yalnız *ek bildirim*;
  gerçek arama asla yok.
- Zaman `event.ts`'ten gelir: kurallar saatten bağımsız, test edilebilir.

### 3.7 API, depolama, bildirim
- **FastAPI**:
  - `GET /api/alerts`, `POST /api/alerts/{id}/confirm`, `POST /api/alerts/{id}/dismiss`
  - `/api/summary/weekly`, `/api/summary/false-alarms`, `/api/heatmap`, `/api/zones`, `/api/level` (ses seviyesi)
  - `/ws` (WebSocket ile anlık uyarı), `/video/{kamera}` (MJPEG canlı yayın)
  - `/snapshots`, `/clips`
- **SQLite** iki tablo:
  - `events`: ham detektör olayları.
  - `alerts`: seviye, sebepler, bölge, durum, snapshot, klip.
  - Haftalık özet, ısı haritası ve yanlış alarm istatistikleri SQL ile hesaplanır.
- **Telegram** (`notify/telegram.py`):
  - Rol bazlı alıcılar:

    | Seviye | Alıcılar |
    |---|---|
    | Sarı | nöbetçi |
    | Turuncu | + müdür yardımcısı, rehber öğretmen |
    | Kırmızı | + müdür |

  - Snapshot ile birlikte gönderilir.
  - Asenkron gönderim: internet yoksa sistem durmaz.
  - Token loglarda maskelenir.

### 3.8 Dashboard (`web/`)
- Canlı görüntü:
  - yüzler bulanık
  - kutu/iskelet overlay'i
  - `kavga_p` olasılığı ve `TEMAS` göstergesi
- Renkli, zaman sıralı uyarı akışı (WebSocket), uyarı kartında klip oynatma.
- **Kırmızı modal**: tam ekran + alarm sesi.
  - **[Onayla]**: "112 arandı (simülasyon)" + öğretmenlere "sınıfta kalın" mesajı.
  - **[Yanlış alarm]**.
- **Kat planı**: bölge poligonları + olay yoğunluğu ısı haritası.
- Haftalık özet (bölge × seviye), yanlış alarm oranı, canlı ses seviyesi göstergesi.

---

## 4. Veri setleri

| Veri seti | İçerik | Lisans | Kullanım |
|---|---|---|---|
| **Surveillance Camera Fight** | 150 kavga + 150 normal güvenlik kamerası klibi | MIT | Kavga eğitimi/testi, silah yanlış alarm ölçümü, demo |
| **Real Life Violence Situations** (Kaggle) | 1000 şiddet + 1000 normal video | Kaggle | Kavga eğitimi, silah için zor negatif |
| **Movies Fight Detection** | 100 kavga + 101 normal film sahnesi | — | Kavga eğitimi |
| **Dangerous Items** (Zenodo 16422779) | 5934 eğitim / 1272 doğrulama / 1272 test, 5 sınıf | CC BY 4.0 | Silah modeli eğitimi ve testi |
| **UR Fall Detection** | 15 düşme + 14 günlük aktivite | Araştırma | Düşme kuralı ölçümü |
| **ESC-50** | 2000 çevresel ses, 50 sınıf | CC BY-NC | Ses yanlış alarm ve cam kırılması ölçümü |
| OpenGameArt Gunshots | Silah sesi kayıtları | CC0 | Silah sesi testi |
| Wikimedia fotoğrafları | 11 tabanca, 7 bıçak | Serbest | Dağılım dışı silah testi |
| **Ekip kayıtları** | `canli.mp4` (bıçak, koşma, boğuşma, itişme, düşme), canlı denemelerin klipleri | Kendi | Gerçek dünya testi, temas kuralı ayarı |

Veri setleri lisans gereği repoda yok (`../hsd_datasets/`). Toplam **2489 kavga klibi** iskelete çevrildi.

---

## 5. Model eğitimi

### 5.1 Kavga modeli
1. **İskelet çıkarma** (`tools/kavga_veri.py`): her video YOLO11n-pose ile ~14 FPS'te işlenip iskelet önbelleğine (`.pkl`)
   yazıldı. GPU'da çalıştırıldı.
2. **Pencereleme** (`tools/kavga_egit.py`):
   - 1.5 sn pencere, 0.5 sn kaydırma, canlıdaki poz hızına (7/sn) indirilmiş örnekleme.
   - 2 farklı faz ile veri artırma.
3. **Model karşılaştırma**: lojistik regresyon, random forest, gradient boosting.
4. **Doğrulama**: **klip bazlı StratifiedGroupKFold (5 kat)**. Aynı klibin pencereleri hem eğitimde hem testte olamaz,
   yani veri sızıntısı yok. Klip skoru = en yüksek pencere olasılığı.
5. **Eşik**: normal kliplerde yanlış alarm ≤ %5 olacak şekilde CV'den seçildi (0.932). Model dosyasına eşik, pencere
   ayarları ve CV sonuçları birlikte kaydedildi.
6. **12 klip demo için ayrıldı**: eğitimde hiç kullanılmadı (`demo_holdout.json`).

| Model | Veri | AUC | %5 yanlış alarmda yakalama |
|---|---|---|---|
| Eski kural (yakınlık + kol hızı) | — | — | %12 (@%3 yanlış alarm) |
| Lojistik regresyon | 300 klip | 0.81 | %40 |
| Lojistik regresyon | 2489 klip | 0.87 | %36 |
| Random forest | 2489 klip | 0.91 | %62 |
| **Gradient boosting (seçilen)** | 2489 klip | **0.91** | **%61** |

### 5.2 Silah modeli
- **v1**: COCO önceden eğitilmiş YOLO11n'den Dangerous Items ile 40 tur (RTX 4060, ~40 dk, imgsz 640, batch 16).
  Ayrılmış test seti mAP50 **0.87**. COCO'nun hazır "knife" sınıfı aynı test setinde yalnız %5 yakalıyordu.
- **Sorun**: gerçek güvenlik kamerasında koyu nesneleri silah sanıyordu.
- **v2 = zor negatif madenciliği** (`tools/negatif_ornek.py`):
  - Silahsız güvenlik kamerası ve gündelik videolardan **1032 kare**, boş etiketle eğitime eklendi.
  - Ölçüm dürüst kalsın diye klipler ikiye bölündü: yarısı eğitime, diğer 72'si yalnız ölçüme.
  - v1'den ince ayar, erken durdurma 11. turda.

| Eşik 0.6 | Test seti yakalama | 72 görülmemiş kamera klibinde yanlış kırmızı |
|---|---|---|
| v1 tabanca / bıçak | %81 / %75 | 3 / 2 |
| **v2 tabanca / bıçak** | %74 / %71 | **0 / 0** |

Okulda yanlış kırmızı alarm pahalı olduğu için v2 seçildi (test mAP50 0.84).

---

## 6. Doğruluğu artırmak için yaptıklarımız (yineleme geçmişi)

1. **Kural → öğrenilmiş model** (kavga): yakalama %12'den %61'e (aynı yanlış alarm seviyesinde).
2. **Daha çok veri**: 300 → 2489 klip, AUC 0.81 → 0.91.
3. **Etkileşim özellikleri**: tek kişi hızı yerine iki kişi ilişkisi (mesafe, yaklaşma, bilek-karşı kişi teması).
4. **Zor negatifler** (silah): kamera yanlış kırmızıları 5/72 → 0/72.
5. **Zamansal doğrulama**: silahta 5 karede 3, temas kuralında 3 sn'de 2 pencere, düşmede 2 sn.
6. **Eşik kalibrasyonu**: her eşik ölçümle seçildi (yakalama / yanlış alarm taraması). Seçim gerekçesi config yorumunda.
7. **Canlı denemeden geri besleme**:
   - Kafa eğmenin düşme sanılması → gövde keypoint şartı.
   - Yakın kamerada yumruk kaçması → temas kuralı.
   - Düşük FPS'te itişme kaçması → kol hızı eşiği 2.0 → 1.2 (canlı kliplerden ölçülerek).
   - Yavaş yatmanın elenmesi → ani düşüş şartı kapatıldı.
8. **Ses**: kısa pencere hatası bulundu (10 sn'ye tamamlama); silah sesinde tip bazında model seçimi; cam eşiği 0.12 → 0.05
   (ESC-50'de yakalama %20 → %45, yanlış alarm %0).

---

## 7. Sonuçlar (özet)

| Detektör | Ölçüm | Sonuç |
|---|---|---|
| Kavga modeli | 2489 klip, 5 katlı CV | AUC 0.91, %5 yanlış alarmda %61 yakalama |
| Kavga (model + temas) | Ekip kaydı | itişme/vurma %57 pencere; bıçak gösterme, koşma, kendi düşme %0 |
| Silah v2 | Ayrılmış test seti | mAP50 0.84; tabanca %74, bıçak %71 (eşik 0.6) |
| Silah v2 | 72 görülmemiş kamera klibi | 0 yanlış kırmızı |
| Cam kırılması | ESC-50 | %45 yakalama, %0 yanlış alarm |
| Silah sesi | 4 CC0 kayıt + ESC-50 | 4/4, %2.4 yanlış alarm |
| Sunum demosu | 6 normal + 6 kavga görülmemiş klip | normal 6/6 alarmsız, kavga 6/6 (tam hatta CPU'da 4/6) |

---

## 8. Değişmez ilkeler kodda nasıl uygulandı

| İlke | Kodda |
|---|---|
| Yüz/duygu tanıma yok | Davranış yalnız iskeletten (`ml/pose_features.py`); yüz sadece konum için, pikselleştirilir |
| Polise otomatik bildirim yok | Kırmızı yalnız ekrana düşer; Onayla = simülasyon; zaman aşımı yalnız ek bildirim, varsayılan kapalı |
| Gizlilik | Bulanıklaştırma yayından/kayıttan önce, fail-closed; klip yalnız turuncu/kırmızıda |
| Eşikler config'de | `config.yaml` + config tutarlılık testleri (ör. detektör eşiği fusion eşiğinin altında olamaz) |

---

## 9. Kalite ve performans

- **70 otomatik test** (pytest): poz kuralları sentetik iskeletle, silah 3/5 kuralı, ses event/bekleme, dB, yüz kutusu,
  klip, fusion kuralları, dedup, eskalasyon, API, depolama, Telegram, bölge zaman çizelgesi, model yüklenemezse yedeğe dönüş.
  Testler model dosyası gerektirmez.
- **Performans**:
  - nano modeller, imgsz 640, kareler 960 px'e küçültülür
  - poz 2 karede bir, silah her karede
  - CPU'da ~14 FPS (video dosyası), PyTorch thread sınırı
  - `device: cpu|cuda` config'den
- **Demo internetsiz**: modeller yerelde, dashboard CDN kullanmaz.
- **Ekip çalışması**: 3 kişi, 3 dal (`goruntu-ses`, `fuzyon-web`, `tasarim`), ortak şema `core/schema.py`, 115+ commit.

---

## 10. Bilinen sınırlar ve sonraki adımlar

- **Dağılım farkı**: modeller kamu veri setleriyle eğitildi. Okulun kendi kamera açılarından toplanmış veri en büyük
  iyileştirme olur.
- **Kalabalık sahne**: RLVS'deki kalabalık spor/sarılma sahnelerinde temas kuralı %13 yanlış alarm.
- **Silah**: dağılım dışı fotoğraflarda düşüş (Wikimedia tabanca 4/11); maket bıçak tanınmıyor.
- **Düşük FPS**: karanlık ortamda webcam ~7 FPS veriyor, hızlı hareket ölçümü zayıflıyor.
- **Sonraki adımlar**:
  - okul verisiyle ince ayar
  - daha büyük model (YOLO11s) için GPU
  - çok kameralı kurulum (bölge başına kamera config'de hazır)
  - çığlık/silah sesi için daha çok pozitif örnek (MIVIA)

---

## 11. Jüriden gelebilecek sorular

- **"Yüz tanıyor musunuz?"** Hayır. Yüzün yalnız yerini bulup pikselleştiriyoruz. Davranışı 17 iskelet noktasından çıkarıyoruz.
- **"Yanlış alarm ne olacak?"**
  - Kademeli uyarı + insan onayı.
  - Zamansal doğrulama (tek kare yetmez).
  - Zor negatiflerle eğitim (kamera kliplerinde 0 yanlış kırmızı).
  - Dashboard yanlış alarm oranını da gösteriyor.
- **"Modeli siz mi eğittiniz?"** Evet:
  - Silah: Dangerous Items + kendi topladığımız negatiflerle YOLO11n, GPU'da.
  - Kavga: 2489 klipten çıkardığımız iskelet özellikleriyle gradient boosting.
  - Ölçümler ayrılmış test setinde ve sızıntısız çapraz doğrulamada.
- **"Okula kurulum?"** Mevcut RTSP kameralar `config.yaml`'a eklenir (kamera → bölge). Ek donanım gerekmez. Laptop CPU'sunda
  çalışıyor.
- **"Neden 112'yi otomatik aramıyor?"** Yanlış alarmın maliyeti çok yüksek; karar yetkisi insanda kalmalı (etik ilke).
