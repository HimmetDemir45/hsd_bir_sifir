# Veri setiyle değerlendirme (2026-10-09)

## ÖZET: Öğrenilmiş modeller (sunumda gösterilecek)

### Kavga modeli (iskelet tabanlı; yüz/görüntü kullanılmaz)
- **Veri:** 2489 klip (1244 kavga / 1245 normal) — Surveillance Camera Fight 300 (güvenlik kamerası),
  Movies Fights 201, Real Life Violence Situations 2000 (Kaggle). 12 klip sunum demosu için eğitimden ayrıldı.
- **Yöntem:** YOLO11n-pose iskeletleri, 1.5 sn pencerelerde 46 etkileşim özelliği (en yakın çift mesafesi ve
  yaklaşma hızı, bileğin karşı kişiye uzaklığı, kol hızı/ivmesi, kutu çakışması...) `ml/pose_features.py`;
  eğitim (`tools/kavga_egit.py`) ve canlı sistem AYNI kodu kullanır.
- **Ölçüm:** klip bazlı 5 katlı çapraz doğrulama (aynı klibin pencereleri hem eğitim hem testte olamaz).

| model | veri | AUC | doğruluk | yakalama@0.5 | **%5 yanlış alarmda yakalama** |
|---|---|---|---|---|---|
| eski kural (yakınlık + kol hızı) | — | — | — | %12 (yanlış %3) | — |
| lojistik regresyon | 300 klip | 0.81 | %71 | %73 | %40 |
| lojistik regresyon | 2489 klip | 0.87 | %77 | %92 | %36 |
| random forest | 2489 klip | 0.91 | %78 | %96 | %62 |
| **gradient boosting (seçilen)** | 2489 klip | **0.91** | %78 | %96 | **%61** |

Gerçek dünya (eğitimde hiç kullanılmadı): Kavga.mp4 arbede pencerelerinin %74'ü kavga, sakin bölüm %0.
Ekip kaydı canli.mp4 (kameraya çok yakın, sahnelenmiş): boğuşma %17, itişme %2 pencere — veri setinden
çok farklı ortamda genelleme zayıf (bilinen sınır).

### Modele ek "hızlı temas" kuralı (yumruk/itme)
Model uzak güvenlik kamerasıyla eğitildi; yakın kamerada (ekip kaydı canli.mp4) yumruğu kaçırıyordu. Kaçan
pencerelerde iki sinyal belirgindi: bilek karşı kişinin kutusunda (`wrist_d` ~0.04 boy) ve kol hızlı (~2.5 boy/sn).
Kural: ikisi AYNI karede, 3 sn içinde 2 pencerede -> kavga (model VEYA kural). `config.yaml pose.fight_model.strike`.

| ölçüm | yalnız model | model + temas |
|---|---|---|
| canli.mp4 itişme/vurma (kavga işaretli kare) | 8 | **287** |
| canli.mp4 iterek düşürme | 0 | **44** |
| canli.mp4 bıçak gösterme / kendi düşme (normal) | 0 / 0 | 0 / 0 |
| veri_demo kavga klipleri (6) | 2-3 | **6** (CPU'da tam hat: 4) |
| veri_demo normal klipler (6) | 0 | 0 |

Görülmemiş normal kliplerde kuralın tek başına yanlış alarmı: Surveillance %2, Movies %0, RLVS %18 (kalabalık
spor/sarılma sahneleri — bilinen sınır). Ayar taraması ve gerekçe config yorumunda.

### Silah modeli (YOLO11n, kendi eğitimimiz)
- **Veri:** Dangerous Items (Zenodo 16422779, CC BY 4.0): 5934 eğitim / 1272 doğrulama / 1272 test görüntüsü,
  5 sınıf. GPU (RTX 4060) 40 tur, ~40 dk. `tools/silah_egit.py` -> v1.
- **v2 = zor negatiflerle ince ayar** (`ml/silah_v2.pt`, sistemde bu kullanılıyor): v1'in hatası gerçek güvenlik
  kamerasında koyu nesneleri silah sanmasıydı. `tools/negatif_ornek.py` 1032 silahsız kare çıkardı (Surveillance
  Fight noFight kliplerinin yarısı + 600 RLVS NonViolence), boş etiketle eğitime eklendi; diğer 72 klip yalnızca
  ölçüm için ayrıldı (eğitimde hiç görülmedi). 11 turda erken durdu.
- **Ayrılmış test seti:** v2 mAP50 0.84 (v1 0.87). Grafikler (v2): `docs/gorseller/silah_test_karisiklik_matrisi.png`,
  `silah_test_pr_egrisi.png`.
- **v1 vs v2** (test seti yakalama | 72 görülmemiş silahsız kamera klibinde yanlış KIRMIZI, 3/5 kare kuralı):

| eşik | v1 tabanca | **v2 tabanca** | v1 bıçak | **v2 bıçak** |
|---|---|---|---|---|
| 0.5 | %87 \| 5/72 | %80 \| **0/72** | %82 \| 4/72 | %79 \| **0/72** |
| 0.6 | %81 \| 3/72 | **%74 \| 0/72** (seçilen) | %75 \| 2/72 | **%71 \| 0/72** (seçilen) |
| 0.7 | %71 \| 0/72 | %63 \| 0/72 | %66 \| 2/72 | %59 \| 0/72 |

  Önceki ayar (v1, tabanca 0.65 / bıçak 0.7): %79 / 3 yanlış, %66 / 2 yanlış. v2 @0.6 yanlış kırmızıyı sıfırladı,
  bıçakta yakalamayı da artırdı; bedeli tabancada ~5 puan yakalama. Okulda yanlış kırmızı alarm daha pahalı.
- **Doğruluğu artırmak için yapılabilecekler:** daha çok gerçek okul kamerası negatifi, okul açısından çekilmiş
  pozitif örnek (dağılım farkı en büyük sorun), daha büyük model (YOLO11s, GPU varsa), çoklu kare doğrulama.
- **Dış test** (Wikimedia fotoğrafları, eğitim kaynağından farklı): v2 tabanca 4/11, bıçak 2/7 (@0.5) — dağılım
  dışı görüntülerde düşüş var.

### Sunum demosu (`test_media/veri_demo.mp4`, `python -m tools.veri_demo`)
Eğitimde görülmemiş 12 güvenlik kamerası klibi + silah veri setinin TEST görüntüleri, üstünde gerçek etiket.
Klip bazında (v2): kavga 2-3/6, normal 6/6 (yanlış alarm yok), tabanca 1/4, bıçak 3/3 — ölçülen başarıyla tutarlı.
Arka arkaya kavga klipleri detektörün 3 sn bekleme süresine takılabiliyor (p ≥ eşik ama yeni event yok).

---

Detektörler, hiç ayar yapılmamış **dış veri setlerinde** ölçüldü: yakalama oranı (TPR: gerçek olayların yüzde kaçı
bulundu) ve yanlış alarm oranı (FPR: normal sahnelerin yüzde kaçında boşuna uyarı). Tekrar üretmek için:

```bash
python -m tools.degerlendir kavga --root ../hsd_datasets/fight
python -m tools.degerlendir dusme --root ../hsd_datasets/urfall
python -m tools.degerlendir ses   --root ../hsd_datasets/esc50/ESC-50-master
```
Görüntü gerçek sistemin hızında (~13 FPS) örneklenir, zaman video saatinden alınır (tekrarlanabilir).
Veri setleri lisans gereği git'te yok (repo dışında `../hsd_datasets/`).

## Özet

| Detektör | Veri seti | Şu anki ayar: yakalama | Şu anki ayar: yanlış alarm | Yorum |
|---|---|---|---|---|
| Cam kırılması | ESC-50 (40 cam / 1960 diğer) | **%45** (eski eşik 0.12: %20) | **%0** | Eşik 0.12 → 0.05 yapıldı |
| Çığlık | ESC-50 (negatif) | — | %0.9 | Çoğu bebek ağlaması (anlamca yakın) |
| Bağırma | ESC-50 (negatif) | — | %0.1 | |
| Silah sesi (kırmızı, ≥0.4) | ESC-50 (negatif) + 4 CC0 silah kaydı | 4/4 | %2.4 | Konserve açma (17), havai fişek (11), cam kırılması (6) |
| Kavga | Surveillance Camera Fight (150 kavga / 150 normal, **2-3 sn** klip) | %7 | %2 | Sinyal var ama 1 sn süreklilik kısa klipte nadiren sağlanıyor (aşağıda) |
| Düşme | UR Fall (15 düşme / 14 günlük aktivite) | %0 | %0 | Klipler düşmeden ~1.3 sn sonra bitiyor; 3 sn kuralı bu sette ölçülemez. "Ani düşüş" şartı eklendi (aşağıda) |

## Kavga
Klipte en az bir kez görülen koşullar (38 + 38 klip örneklem):

| Koşul | Kavga klipleri | Normal klipler |
|---|---|---|
| 2+ kişi algılandı | %87 | %53 |
| Yakın çift (merkez mesafesi < 0.6 x boy) | %71 | %34 |
| Hızlı kol hareketi (≥ 1.5 boy/sn) | %63 | %24 |
| **Yakın + hızlı aynı anda** | **%50** | **%18** |

Süreklilik eşiği (`pose.fight.min_duration_s`) taraması: 0.25 sn → %18 / %7, 0.5 sn → %12 / %3,
**1.0 sn (şu anki) → %7 / %2**, 1.5 sn → %2 / %1. Kural uzun süren kavgada çalışıyor (Kavga.mp4, 26 sn: 3/3), kısa
itişmeleri kaçırıyor. Süreyi kısaltmak yakalamayı az artırıp yanlış alarmı artırıyor; asıl iyileştirme öğrenilmiş bir
kavga sınıflandırıcısı olurdu (hackathon süresine sığmaz).

## Düşme
Duruş tespiti etiketle uyumlu (ölçülen "yatay" süre ≈ etiketli "yerde yatma" süresi; ör. fall-06: 1.8 / 2.0 sn).
Ama veri setinde kişi düşmeden sonra ortalama **1.3 sn** (en fazla 2.0 sn) görünüyor; bizim kural 3 sn yatmayı bekliyor.
Eşik taraması: 0.5 sn → %80 / %29, **1.0 sn → %53 / %21**, 1.5 sn → %13 / %7, 3 sn → %0 / %0.
Yanlış alarmlar "yatağa uzanma" gibi bilinçli yatma hareketlerinden: kural düşmeyi yavaşça uzanmaktan ayırmıyordu.

**Eklenen "ani düşüş" şartı** (`pose.fall.require_sudden`): yataya geçmeden en fazla `transition_s` (1 sn) önce yakında
dik duran biri olmalı ve gövde merkezi en az `drop_ratio` x boy aşağı inmiş olmalı. Ayrıca yatay duruş kısa süre
(`gap_s` 0.5 sn) kesilirse süre artık sıfırlanmıyor. Tarama (yakalama / yanlış alarm):

| ayar | 0.5 sn | 1.0 sn | 1.5 sn |
|---|---|---|---|
| ani düşüş yok | %93 / %36 | %67 / %36 | %40 / %7 |
| **ani düşüş, drop 0.15 (seçilen)** | **%67 / %14** | %40 / %7 | %20 / %0 |
| ani düşüş, drop 0.2 | %60 / %14 | %33 / %7 | %13 / %0 |
| ani düşüş, drop 0.3 | %27 / %7 | %20 / %0 | %0 / %0 |

Aynı yakalamada (%67) yanlış alarm %36 → %14. Süre CLAUDE.md gereği **3 sn** kaldı (veri seti bunu ölçemiyor:
kişi en fazla 2 sn yerde görünüyor); şart yalnızca yavaşça uzanmanın alarm vermesini engelliyor.

## Ses
- Cam kırılması: tüm eşiklerde yanlış alarm %0 → eşik 0.12'den **0.05**'e indirildi (yakalama %20 → %45, 18/40;
  yanlış alarm 0/1960). Demo senaryosu değişmedi.
- Silah sesi kırmızı eşiği taraması (ESC-50 yanlış alarm): 0.25 → %3.6, **0.4 → %2.4**, 0.5 → %1.6, 0.6 → %0.9.
  Kendi silah kayıtlarımız 0.41 / 0.50 / 0.59 / 0.80: 0.5'te 3/4, 0.4'te 4/4. Kırmızıda insan onayı olduğundan
  kaçırmamak öncelikli: 0.4 korunuyor. Bilinen risk: kantinde konserve açma, havai fişek.

## Canlı kamera kaydı (ekip, test_media/canli.mp4, 185 sn, 1080p)
Ekibin kendi kamerasıyla çektiği kayıt: maket bıçağı (0-47 sn), koşma (52-61), boğuşma (66-70), itişme/vurma
(75-145), düşme (149-165), iterek düşürme (166-181). Canlı sistem hızında (14 FPS) kare kare ölçüldü
(`../hsd_datasets/canli_analiz.py`, `canli_tarama.py`).

| Bölüm | Önceki ayar | Yeni ayar | Neden kaçıyordu |
|---|---|---|---|
| Maket bıçağı | ❌ | ❌ | COCO "knife" güveni 0.00; maket bıçağı modellerin bıçak sınıfına benzemiyor (tabanca modeli 0.62 ile "tabanca" sandı, eşik 0.8 olduğu için alarm yok) |
| Koşma (2 geçiş) | 0 | **2** | kural en az 2 kişi istiyordu, kadrajda tek kişi ~0.7 sn görünüyor -> tek kişi, hız ≥2.0, ≥0.3 sn |
| Boğuşma / itişme | 0 / 0 | **1 / 4** uyarı | kol hızı ve yakınlık sağlanıyor ama itişme kesik kesik (en uzun 0.7 sn) -> min süre 1.0 -> 0.5 sn |
| Düşme | ✅ | ✅ | |
| İterek düşürme | kavga | **kavga + düşme** | kişi yerde 2.4 sn kalıp kalktı -> düşme süresi 3 -> 2 sn |

Yeni ayarın bedeli (kavga veri seti, event bazlı): yakalama %7 -> %12, yanlış kavga %2 -> %3, yanlış koşma %0.
Kayıtta bıçak/koşma bölümlerinde yanlış kavga/düşme yok; Kavga.mp4 demo senaryosu değişmedi.
Ayrıca canlı test şunları gösterdi ve düzeltildi: canlı kamera 8 FPS'te işleniyordu (-> 15), laptop önünde kafa
eğmek "düşme" üretiyordu (kutu oranı yedeği artık yalnız bacaklar görünüyorsa).

## Silah / bıçak (görüntü)
Tabanca için [JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8](https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8)
modeli (MIT, YOLOv8n) eklendi; bıçak COCO'da kaldı (Joao'nun bıçak sınıfı fotoğraf testinde 1/7). Model dosyası
pickle olduğu için yüklemeden önce statik tarandı (yalnız torch/ultralytics/collections) ve SHA-256 ile sabitlendi.
Test: Wikimedia Commons serbest lisanslı 11 tabanca + 7 bıçak fotoğrafı; yanlış alarm için silahsız güvenlik
kamerası klipleri (Surveillance Fight) ve demo videoları. 5 karenin 3'ü kuralıyla:

| tabanca eşiği | tabanca fotoğrafı | yanlış kırmızı: normal klip | yanlış kırmızı: kavga klibi | Kavga.mp4 (sopa) |
|---|---|---|---|---|
| 0.6 | 9/11 | 26/150 | 40/150 | silah sanıldı |
| 0.7 | 7/11 | 13/150 | 19/150 | silah sanıldı |
| **0.8 (seçilen)** | **4/11** | **4/150** | **4/150** | yok |

Bıçak (COCO, 0.6): fotoğrafların 2/7'si. İki model birlikte kare başına ~95 ms sürüp FPS'i 13 → 7-11'e indirdi ve
demo videosunda kavga tespiti kayboldu; modeller karelere sırayla dağıtılınca ~27 ms, FPS 13-16, kavga geri geldi.

## Sınırlar
- Veri setleri okul ortamı değil; sonuçlar mutlak doğruluk değil, **gösterge**.
- Kavga klipleri çok kısa (2-3 sn); düşme klipleri düşmeden hemen sonra bitiyor. Her ikisi de süreye dayalı
  kurallarımızı gerçek hayattan daha zor bir durumda ölçüyor.
- Çığlık ve silah için pozitif örnek az (ESC-50'de yok); MIVIA Audio Events (kayıt gerekli) ile genişletilebilir.

Kaynaklar: [Surveillance Camera Fight Dataset](https://github.com/sayibet/fight-detection-surv-dataset) (MIT),
[UR Fall Detection](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) (CC BY-NC-SA 4.0),
[ESC-50](https://github.com/karolpiczak/ESC-50) (CC BY-NC), [OpenGameArt Gunshots](https://opengameart.org/content/gunshots) (CC0).
