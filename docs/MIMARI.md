# Mimari (tek sayfa)

OkulKalkan, okul kameralarından ve mikrofonundan gelen görüntü + sesi analiz edip **kademeli uyarı** üretir
(Sarı / Turuncu / Kırmızı). Kararı her zaman bir insan verir; sistem polisi/112'yi otomatik aramaz.

## Veri akışı

```
 Kamera / video ──► Kare küçültme ──► Silah/bıçak (YOLO11n, kendi modelimiz) ┐
 (sources/camera)   (max_frame_side)  Poz → kavga (öğrenilmiş model) /      ├─► Event ──► queue.Queue ──► FUSION ──► Alert
                                      düşme / koşuşma (kural) / kalabalık ──┘                            (fusion/)     │
 Mikrofon / wav ──► Ses sınıflandırma (EfficientAT + YAMNet) + dB ─► Event ──────────────┘             │
                                                                                                        ▼
      yüz bulanıklaştırma (privacy/blur) ──► canlı yayın (MJPEG), snapshot, klip       SQLite (storage/)  ·  WebSocket  ·  Telegram (notify/)
                                                                                                        │
                                                                                          Dashboard (web/): uyarı akışı, kırmızı modal,
                                                                                          kat planı ısı haritası, haftalık özet, ses seviyesi
```

- **Detektörler** (`detectors/`, `ml/`): **Silah/bıçak** için kendi eğittiğimiz YOLO11n (`ml/silah_v2.pt`, Dangerous Items seti,
  CC BY 4.0; ince ayar için silahsız güvenlik kamerası kareleri). **Poz**: YOLO11n-pose, 17 keypoint. **Kavga**:
  iskelet pencerelerinden 46 özellik (`ml/pose_features.py`) + gradient boosting sınıflandırıcı (`ml/kavga_model.joblib`),
  `pose.fight_model.enabled` açıkken kararı model verir; model dosyası yoksa elle yazılmış kural (yakınlık + kol hızı) devreye girer.
  **Düşme, koşuşma, kalabalık** kural tabanlı. **Ses**: EfficientAT + YAMNet + dB.
- **Event** (detektör → fusion): `source`, `type` (gun/knife/fight/fall/running/scream/shout/gunshot/glass), `confidence`,
  `camera_id`, `zone_id`, `ts`. **Alert** (fusion → API/DB/Telegram): `level`, `reasons` (Türkçe metin), `zone_id`,
  `status` (pending/confirmed/dismissed), `snapshot`, `clip`. Şema `core/schema.py`'de donduruldu.
- **Thread'ler** (aralarında yalnızca kuyruk veya kilitli değişken): kamera, ses, fusion, API (uvicorn); isteğe bağlı
  kırmızı zaman aşımı izleyicisi ve demo senaryosu. Bir thread çökerse diğerleri çalışmaya devam eder.

## Fusion (`fusion/`)
- **Kurallar** (`rules.py`, saf fonksiyonlar): tabanca/bıçak ve silah sesi eşiği aşarsa **kırmızı**; kavga, düşme veya çığlık +
  kalabalık **turuncu**; bağırma, koşuşma, cam kırılması, çığlık (kalabalıksız) **sarı**.
- **Dedup**: aynı bölge + aynı tip 30 sn içinde tek uyarıda birleşir (yeni sebep eklenir, seviye yükselirse yükselir).
- **Eskalasyon**: aynı bölgede 120 sn içinde 3 sarı uyarı → turuncu.
- **Tüm eşikler `config.yaml`'da**; kodda sabit eşik yok. Zaman, `event.ts`'ten gelir, bu yüzden kurallar test edilebilir.

## İnsan onayı ve 112
- Kırmızı uyarı dashboard'da **tam ekran modal + alarm** açar. **[Onayla]** yalnızca durumu `confirmed` yapar ve
  "112 arandı (simülasyon)" gösterir; **gerçek arama yoktur**. **[Yanlış alarm]** durumu `dismissed` yapar.
- Turuncu/sarıda kartın üzerindeki **[Gördüm]** yalnızca "Görüldü" işaretler; 112 mesajı **yalnızca kırmızıda** gönderilir.
- `alerts.auto_escalate_red` varsayılan **kapalı**. Açılırsa, süre içinde onaylanmayan kırmızı için **sadece ek bildirim ve ekranda
  uyarı** üretir (karar yine insanda).

## Gizlilik
- Yüz **tanıma yok**; yüzün yeri yalnızca bulanıklaştırmak için bulunur. Davranış, iskelet keypoint'lerinden çıkarılır.
- Bulanıklaştırma, canlı yayından, snapshot'tan ve klipten **önce** yapılır. Bulanıklaştırma hata verirse kare **yayına/kayda
  alınmaz** (fail-closed).
- Klip yalnızca turuncu/kırmızıda kaydedilir. Telegram token'ı `.env`'de tutulur ve hata loglarında maskelenir.

## Arayüzler
- **REST**: `/api/alerts`, `/api/alerts/{id}/confirm|dismiss`, `/api/summary/weekly`, `/api/summary/false-alarms`,
  `/api/heatmap`, `/api/zones`, `/api/level` (canlı ses seviyesi), `/api/status`, `/clips/…`, `/snapshots/…`.
- **WebSocket** `/ws`: uyarı güncellemeleri (ve demo yeniden başlatma). **MJPEG** `/video/{kamera}`.
- Dashboard tek sayfa, vanilla HTML/JS/CSS; font ve ses dosyası dahil her şey `web/` içinde, **internetsiz çalışır**.
- Sunucu varsayılan olarak yalnızca `127.0.0.1`'de dinler.

## Dayanıklılık
- İnternet yoksa Telegram hatası loglanır, pipeline durmaz. Kamera kopunca yeniden açılır.
- `python main.py --demo` sahte olay senaryosunu oynatır; model veya kamera gerekmez (yedek demo).

## Sınırlar ve gelecek iş
**Şu an:** Kavga tespiti **öğrenilmiş bir modelle**, ama yüz/görüntü değil, yalnızca **iskelet keypoint'leri** üzerinden (ilke 1).
Düşme, koşuşma ve kalabalık hâlâ **kural tabanlı** (hız, süre, yakınlık eşikleri; her uyarının gerekçesi gösterilebiliyor, eşikler
`config.yaml`'dan ayarlanıyor). Silah/bıçak için kendi eğittiğimiz YOLO11n kullanılıyor.

**Ölçülen sonuçlar** ([DEGERLENDIRME.md](DEGERLENDIRME.md); dış veri setleri, okul ortamı değil, bu yüzden mutlak doğruluk değil
gösterge):
- **Kavga:** 2489 klipte klip bazlı 5 katlı çapraz doğrulama, AUC 0,91; **%5 yanlış alarmda %61 yakalama** (eski kural: %12).
- **Silah/bıçak (v2, eşik 0,6):** ayrılmış test setinde tabanca %74, bıçak %71 yakalama, test mAP50 0,84; 72 görülmemiş silahsız
  güvenlik kamerası klibinde yanlış kırmızı 0/72.

**Bilinen sınırlar (jüriye açıkça söylenebilir):**
- Kavga rakamları **klip başına** ölçüldü. Canlı akışta model her 0,5 sn'de bakıyor; **saat başına yanlış alarm ölçülmedi**.
- Gerçek bir sınıf kaydında (kameraya çok yakın, sahnelenmiş) genelleme zayıf kaldı: boğuşma pencerelerinin %17'si, itişmenin %2'si.
- Silah modeli veri setine dışarıdan bakınca zayıflıyor: dış fotoğraflarda tabanca 4/11, bıçak 2/7.
- Eğitim verisi okul kamerası açısından değil; okulda çekilmiş örnekler en büyük iyileştirme olurdu.

**Kurulum notu:** Kavga modeli `scikit-learn` ister (`requirements.txt`'te sabit sürümle). Yeni bir makinede
`pip install -r requirements.txt` çalıştırılmalı ve açılışta `[pose] kavga modeli: ...` satırı görülmelidir.

**Gelecek iş (yapılmadı):**
1. Okul kamerası açısından ek eğitim verisi ve sürekli akışta saat başına yanlış alarm ölçümü; çok pencereli doğrulama.
2. Çoklu kamera (kamera başına bulanıklaştırma durumu ve klip tamponu gerekir).
3. Telegram'dan onay (satır içi butonlar, yalnızca yetkili sohbet), okul bilgi sistemiyle entegrasyon.
4. Daha fazla ortamda kalibrasyon ve ölçüm (eşikler gerçek bir okulda ayarlanmalı).

## Ayrıca bakın
[DEMO.md](DEMO.md) (demo rehberi ve bilinen sınırlar) · [DEGERLENDIRME.md](DEGERLENDIRME.md) (veri setiyle yakalama / yanlış
alarm oranları) · [../CLAUDE.md](../CLAUDE.md) (ilkeler ve kurallar).
