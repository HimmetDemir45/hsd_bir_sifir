# Demo rehberi (sunum: 10 Ekim 09.30)

Üç seçenek, güvenilirden etkileyiciye. Sunumda **önce 1'i** gösterin; zaman ve koşullar uygunsa 3'ü ekleyin.
2 her zaman yedekte açık dursun (bir şey bozulursa ona geçin).

## 1. Gerçek detektörlerle hazır senaryo (önerilen ana demo)

```bash
python main.py --fresh --source test_media/Kavga.mp4 --audio test_media/demo_ses.wav
```
Tarayıcı: `http://localhost:8000`, "Alarm sesi" düğmesine **önceden** bir kez tıklayın.

Gerçek modeller çalışır (sahte olay yok). Yaklaşık 30 sn'de şu akış oluşur (test edildi):

| ~zaman | Kaynak | Uyarı |
|---|---|---|
| 3. sn | ses: cam kırılması | 🟡 SARI: Cam kırılma sesi |
| 8-10. sn | görüntü: sınıfta kavga (sopa ile saldırı) | 🟠 TURUNCU: Kavga şüphesi (+10 sn bulanık klip) |
| 11. sn | ses: çığlık (+ o anda görüntüde ≥4 kişi varsa) | 🟠 TURUNCU: Çığlık ve kalabalık, ya da 🟡 SARI: Çığlık duyuldu |
| 23. sn | ses: silah sesi | 🔴 KIRMIZI: tam ekran modal + alarm → **Onayla** → "112 arandı (simülasyon)" |

Çığlık satırı iki şekilde de çıkabilir (testlerde ikisi de oldu): çığlık anında videoda tam 4 kişinin
görülmesine bağlı; sınırda. İkisi de doğru davranış: "kalabalıkla birlikte çığlık daha ciddi" diye anlatılabilir.

Anlatırken gösterilecekler: yüzler bulanık (canlı yayın, snapshot ve klip), iskelet çizimi (yüz/kimlik yok),
"Klibi izle" bağlantısı, kat planı ısı haritası, kırmızıda insan onayı (otomatik polis araması yok).

**Medya dosyaları git'te yok** (`test_media/` git'e eklenmez). Sunum bilgisayarına elle kopyalayın:
- `test_media/Kavga.mp4`: sınıf kavgası (güvenlik kamerası haberi, alt kısımdaki röportaj kesildi)
- `test_media/demo_ses.wav` (31 sn): 3. sn cam kırılması ([ESC-50](https://github.com/karolpiczak/ESC-50), CC BY-NC),
  10.5. sn çığlık (`ciglik.mp4`'ten), 22.5. sn silah sesi ([OpenGameArt "Gunshots"](https://opengameart.org/content/gunshots), CC0)

## 2. Yedek: sahte olaylarla demo (model/kamera gerekmez)

```bash
python main.py --demo --fresh
```
Detektörler kırılsa da dashboard, fusion, eskalasyon, kırmızı modal ve onay akışı gösterilebilir.

## 3. Canlı: webcam + mikrofon

```bash
python main.py --fresh --audio
```
Önceden **mutlaka** aşağıdaki kontrolleri yapın; canlı demo ortama (ışık, gürültü, mesafe) çok bağlıdır.

## Sunumdan önce kontroller (Kişi A)

1. **Modeller:** `python -m tools.modelleri_indir` → "tüm modeller hazır". `main.py` açılışında `[privacy] UYARI` olmamalı.
2. **Mikrofon seçimi:** `python -m detectors.audio_cls --list-devices` → doğru mikrofonu `config.yaml` `audio.input`'a yaz
   (varsayılan "WO Mic" telefon uygulaması olabilir).
3. **dB kalibrasyonu:** `python -m detectors.audio_cls` → normal konuşmada `dB=` 55-65 olmalı; değilse `audio.db_offset`.
4. **Silah sesi (mikrofondan):** telefondan bir silah sesi efekti çalıp mikrofona tutun → `silah` skoru 0.25 üstüne çıkmalı.
   Fusion kırmızı için `alerts.red.gunshot_min_conf: 0.4` ister; testlerde silah sesleri 0.41-0.80 aldı (hepsi kırmızı).
5. **Bıçak (webcam):** `python -m detectors.weapon` → mutfak bıçağını kameraya yan tutun. Kutu kırmızı/turuncu ve
   güven ≥ 0.6 olmalı, konsolda `"type": "knife"`. Düşükse `weapon.min_conf` ve `alerts.red.weapon_min_conf` birlikte düşürülür
   (yanlış alarm riskini artırır). Tabanca için `models/weapon.pt` gerekir (yoksa sadece bıçak).
6. **Yanlış alarm:** kamera önünde 2 dk normal oturun/konuşun/yürüyün → kavga/düşme uyarısı gelmemeli.
7. **Bulanıklaştırma:** `python -m privacy.blur` → yüz kapanmalı, boyun açık kalmalı (`privacy.head_scale` ile ayar).

## Bilinen sınırlar (jüri sorarsa dürüstçe)

- Sopa/beyzbol sopası nesne olarak tanınmıyor; saldırı **kavga** (davranış) olarak yakalanıyor.
- Hareket bulanıklığı ve arkası dönük kişilerde birkaç karelik yüz bulanıklaştırma kaçağı olabilir (kafanın arkası).
- Ses modelleri hazır (AudioSet) modeller; stüdyo kaydı tek atışlık bazı silah seslerini düşük skorlayabiliyor
  (test: 4 kaydın 3'ü yakalandı). Silah sesi için YAMNet, diğer sesler için EfficientAT kullanılıyor.
- Laptop CPU'da ~12-15 FPS işleniyor. Dayanıklılık testi: 10 dk ve 6 dk kesintisiz, çökme/hata yok, bellek ~700 MB'de
  sabit (sızıntı yok). CPU kullanımı `general.torch_threads: 4` ile ~8 çekirdekten ~6'ya indirildi.
