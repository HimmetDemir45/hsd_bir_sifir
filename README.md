# hsd_bir_sifir — OkulKalkan

Okul güvenliği erken uyarı sistemi (KTÜ HSD Eğitim Hackathonu, Tema 05). Ayrıntılar: [CLAUDE.md](CLAUDE.md)

## Kurulum (Windows, Python 3.11)

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m tools.modelleri_indir
```

**Son komut zorunlu:** tüm modelleri `models/` altına indirir (git'e eklenmez, internet gerekir). Demo internetsiz
çalışacağı için sunumdan önce mutlaka çalıştırın; sonunda "tüm modeller hazır" yazmalı. İndirilenler:
- `face_detection_yunet_2023mar.onnx`: yüz bulanıklaştırma (YuNet). **Yoksa bulanıklaştırma zayıf yedeğe düşer**, açılışta uyarı basılır.
- `yolo11n.pt`, `yolo11n-pose.pt`: bıçak ve poz. `mn10_as_mAP_471.pt`: ses (EfficientAT). `yamnet.tflite`: ses yedeği.
- `weapon.pt` (Roboflow Universe silah modeli) elle eklenir; yoksa sistem `yolo11n.pt` ile sadece bıçak tespitine düşer.

OpenCV `4.14`'e sabittir: 5.0 pip paketinde Haar cascade dosyaları yok. Eski kurulumu güncellemek için tekrar
`pip install -r requirements.txt` çalıştırın (`opencv-python` ve `opencv-contrib-python` aynı sürümde olmalı).

## Detektörleri tek başına çalıştırma

```bash
python -m detectors.weapon                              # webcam
python -m detectors.weapon --source test_media/x.mp4    # video dosyası
python -m detectors.pose                                # kavga / düşme / koşuşma
python -m detectors.audio_cls                           # mikrofon: çığlık / bağırma / silah sesi / cam + dB
python -m detectors.audio_cls --audio test_media/x.wav  # ses dosyası
python -m detectors.audio_cls --list-devices            # mikrofon numaraları (config: audio.input)
```

Ses modeli: [EfficientAT](https://github.com/fschmid56/EfficientAT) `mn10_as` (MIT, `detectors/third_party/efficientat/`),
yedek olarak MediaPipe + YAMNet (`models/yamnet.tflite`, `tools.modelleri_indir` indirir).

```bash
python -m privacy.blur                                  # webcam: yüz bulanıklaştırmayı dene
python -m privacy.blur --source test_media/Kavga.mp4
```

Çıkmak için pencerede `q`. Tespit edilen olaylar konsola JSON olarak basılır.
Video dosyaları gerçek zamanlı oynatılır (CPU yetişemezse kare atlanır), böylece canlı kamerayla aynı davranır.

## Demo nasıl çalıştırılır

Dashboard: `http://localhost:8000` (başka bir uygulama 8000'i tutuyorsa `--port 8001`).
Her çalıştırmada `--fresh` ile eski kayıtları silip temiz başlayın.

**1. Sahte senaryo (kamera/model gerekmez, en güvenli):**
```bash
python main.py --demo --fresh
```
Yaklaşık 40 sn içinde: önce koridorda ve Sınıf 1-A'da birer küçük sarı olay (ısı haritasında başka bölgeler de görünsün), sonra Kantin'de sırayla 3 sarı (bağırma, koşuşma, cam) → eskalasyonla turuncu → kavga (turuncu) → bıçak (kırmızı).
Hızlı deneme için `--speed 5`.

**2. Gerçek kamera ve ses:**
```bash
python main.py --fresh                                               # webcam
python main.py --fresh --source test_media/kavga.mp4                 # video dosyası
python main.py --fresh --source test_media/kavga.mp4 --audio test_media/ciglik.wav
python main.py --fresh --audio                                       # mikrofon
```

**Dashboard'da gösterilecekler**
1. Sol üst: canlı görüntü (kutular / iskeletler). Sağ: renkli, zaman sıralı uyarı akışı.
2. Sağ üstteki **"Alarm sesi kapalı"** butonuna bir kez tıklayın (tarayıcı sesi ancak tıklamadan sonra çalar).
3. Kırmızı uyarıda tam ekran modal + alarm → **Onayla** (ekranda "112 arandı (simülasyon)"; gerçek arama yoktur) veya **Yanlış alarm**.
4. Aşağıda: kat planında olay yoğunluğu ve haftalık özet tablosu.

**Telegram (isteğe bağlı):** `.env` içine `TELEGRAM_BOT_TOKEN` ve `TELEGRAM_CHAT_ID` yazın, `config.yaml`'da `telegram.enabled: true` yapın.
Kapalıyken veya internet yokken mesajlar sadece konsola yazılır; sistem çalışmaya devam eder.

**Gerçek detektörlerle demo senaryosu, canlı demo kontrolleri ve bilinen sınırlar:** [docs/DEMO.md](docs/DEMO.md)

**Sunum öncesi kontrol listesi**
- [ ] `python -m tools.modelleri_indir` "tüm modeller hazır" diyor (internet varken)
- [ ] `main.py` açılışında `[privacy] UYARI` satırı YOK (varsa YuNet inmemiş, yüzler kaçabilir)
- [ ] `python -m pytest tests -q` hepsi geçiyor (pytest geliştirme bağımlılığı: `pip install pytest`; testler model dosyası istemez)
- [ ] `python main.py --demo --fresh` ile kırmızıya kadar gidip Onayla denendi
- [ ] Alarm sesi butonu tıklandı, hoparlör açık
- [ ] Dashboard internetsiz açılıyor (CDN yok, her şey `web/` içinde)

**Sorun giderme**
- *Sayfa eski/boş görünüyor:* 8000 portu başka süreçte olabilir; `--port 8001` ile deneyin.
- *Kamera açılmıyor:* `--demo` ile devam edin; senaryo gerçek detektörlerden bağımsız çalışır.
- *Alarm sesi gelmiyor:* "Alarm sesi" butonuna tıklayın.
- *Kırmızı uyarıda görüntü yok:* yüz bulanıklaştırma (`privacy/blur.py`) hazır değilse gerçek kamerada snapshot alınmaz; modal canlı yayını gösterir.

## Görev dağılımı

| Kişi A: Algılama | Kişi B: Füzyon + Backend + Dashboard mantığı | Kişi C: Frontend tasarımı |
|---|---|---|
| `sources/`, `detectors/`, `privacy/`, `tools/`, `models/`, `test_media/` | `fusion/`, `api/`, `storage/`, `notify/`, `web/app.js`, `main.py` | `web/index.html`, `web/style.css`, `web/assets/` |
| dal: `goruntu-ses` | dal: `fuzyon-web` · [docs/KISI_B.md](docs/KISI_B.md) | dal: `tasarim` · [docs/KISI_C.md](docs/KISI_C.md) |

Ortak sözleşme: `core/schema.py` (Event, Alert). Değiştirmeden önce haber verin.
`config.yaml`'da herkes kendi bölümünü düzenler. C, `app.js`'in kullandığı id/sınıfları korur (bkz. KISI_C.md).
