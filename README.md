# hsd_bir_sifir — OkulKalkan

Okul güvenliği erken uyarı sistemi (KTÜ HSD Eğitim Hackathonu, Tema 05). Ayrıntılar: [CLAUDE.md](CLAUDE.md)

## Kurulum (Windows, Python 3.11)

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Model dosyaları `models/` klasörüne konur (git'e eklenmez):
- `models/weapon.pt`: Roboflow Universe weapon modeli. Yoksa sistem otomatik olarak `yolo11n.pt` ile sadece bıçak tespitine düşer.
- `models/yolo11n.pt`, `models/yolo11n-pose.pt`: ilk çalıştırmada otomatik iner (internet gerekir; demodan önce bir kez çalıştırın).

## Detektörleri tek başına çalıştırma

```bash
python -m detectors.weapon                              # webcam
python -m detectors.weapon --source test_media/x.mp4    # video dosyası
python -m detectors.pose                                # kavga / düşme / koşuşma
```

Çıkmak için pencerede `q`. Tespit edilen olaylar konsola JSON olarak basılır.
Video dosyaları gerçek zamanlı oynatılır (CPU yetişemezse kare atlanır), böylece canlı kamerayla aynı davranır.

## Görev dağılımı

| Kişi A: Algılama | Kişi B: Füzyon + Backend + Dashboard |
|---|---|
| `sources/`, `detectors/`, `privacy/`, `models/`, `test_media/` | `fusion/`, `api/`, `storage/`, `notify/`, `web/`, `main.py` |

Ortak sözleşme: `core/schema.py` (Event, Alert). Değiştirmeden önce haber verin.
`config.yaml`'da herkes kendi bölümünü düzenler.
