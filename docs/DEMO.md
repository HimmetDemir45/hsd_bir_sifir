# Demo rehberi (sunum: 10 Ekim 09.30)

Sunumda **önce 0 (veri seti, eğitilmiş modeller)**, sonra 1; zaman ve koşullar uygunsa 3.
2 her zaman yedekte açık dursun (bir şey bozulursa ona geçin).

## 0. Veri seti demosu — eğitilmiş modeller (ANA DEMO)

```bash
python main.py --fresh --source test_media/veri_demo.mp4
```
`test_media/veri_demo.mp4` (69 sn, `python -m tools.veri_demo` ile üretilir, git'te yok -> kopyalayın):
eğitimde **hiç görülmemiş** 6 normal + 6 kavga güvenlik kamerası klibi, sonra silah veri setinin TEST
bölümünden 4 tabanca + 3 bıçak görüntüsü. Her parçanın üstünde gerçek etiket yazar.
Beklenen: normal kliplerde uyarı yok, kavgada 🟠 "Kavga şüphesi" (ekranda `kavga_p` olasılığı), tabanca/bıçakta
🔴 kırmızı modal. Aynı tip olaylar 30 sn içinde tek kartta birleşir (fusion dedup) — kart sayısı değil, kartın
gelmesi ve `kavga_p` önemli. Anlatırken `docs/DEGERLENDIRME.md` ÖZET tablosunu ve `docs/gorseller/` grafiklerini
gösterin: "2489 klipte çapraz doğrulama AUC 0.91; silah modeli ayrılmış testte mAP50 0.84, zor negatiflerle eğitildikten
sonra 72 görülmemiş kamera klibinde 0 yanlış kırmızı".
Dürüst not: her klip yakalanmıyor (kavga ~%61 yakalama @%5 yanlış alarm; tabanca %74, bıçak %71 yakalama) — tablo bunu söylüyor.

## 1. Gerçek detektörlerle hazır senaryo

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
5. **Bıçak / tabanca (webcam):** `python -m detectors.weapon` (pose komutu silaha bakmaz). Gerçek **mutfak bıçağını**
   kameraya yan tutun; **maket bıçağı tanınmıyor** (canlı kayıtta güven 0.00). Kendi modelimiz (`ml/silah_v2.pt`,
   eşik 0.6) net görüntüde çalışıyor; fotoğraf testinde 4/11. Silahı en güvenilir şekilde **silah sesiyle** gösterin.
6. **Kavga / düşme / koşma:** canli.mp4 kaydıyla ayarlandı (docs/DEGERLENDIRME.md). Kavga için kollar hızlı ve
   yakın olun; düşmede kendinizi **birden** bırakıp en az **2 sn** yerde kalın (yavaşça uzanmak düşme sayılmaz).
7. **Yanlış alarm:** kamera önünde 2 dk normal oturun/konuşun/yürüyün → kavga/düşme uyarısı gelmemeli.
7. **Bulanıklaştırma:** `python -m privacy.blur` → yüz kapanmalı, boyun açık kalmalı (`privacy.head_scale` ile ayar).

## Bilinen sınırlar (jüri sorarsa dürüstçe)

- Sopa/beyzbol sopası ve maket bıçağı nesne olarak tanınmıyor; saldırı **kavga** (davranış) olarak yakalanıyor.
- Kısa itişmeler (0.5 sn'den kısa) ve kadrajdan çok hızlı geçen koşucu kaçabilir (canlı kayıt: kavga 6 uyarı, koşma 2/2).
- Hareket bulanıklığı ve arkası dönük kişilerde birkaç karelik yüz bulanıklaştırma kaçağı olabilir (kafanın arkası).
- Ses modelleri hazır (AudioSet) modeller; stüdyo kaydı tek atışlık bazı silah seslerini düşük skorlayabiliyor
  (test: 4 kaydın 3'ü yakalandı). Silah sesi için YAMNet, diğer sesler için EfficientAT kullanılıyor.
- Laptop CPU'da ~12-15 FPS işleniyor. Dayanıklılık testi: 10 dk ve 6 dk kesintisiz, çökme/hata yok, bellek ~700 MB'de
  sabit (sızıntı yok). CPU kullanımı `general.torch_threads: 4` ile ~8 çekirdekten ~6'ya indirildi.
