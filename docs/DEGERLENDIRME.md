# Veri setiyle değerlendirme (2026-10-09)

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
