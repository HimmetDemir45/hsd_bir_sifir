# EfficientAT (vendored)

- Kaynak: https://github.com/fschmid56/EfficientAT
- Commit: `a425fdce92572e602a1d5634799bd9f1f2efa806` (2024-11-20)
- Lisans: MIT (bkz. `LICENSE`, Copyright (c) 2022 Florian Schmid)
- Makale: Schmid vd., "Efficient Large-Scale Audio Tagging via Transformer-to-CNN Knowledge Distillation", ICASSP 2023

AudioSet'te eğitilmiş MobileNetV3 ses sınıflandırıcıları (527 sınıf). Projede `mn10_as` (4.9M parametre, mAP 47.1) kullanılıyor.
Ağırlıklar ilk çalıştırmada GitHub Releases'tan `models/` klasörüne iner.

## Alınan dosyalar
- `mn/model.py`, `mn/block_types.py`, `mn/attention_pooling.py`, `mn/utils.py` ← `models/mn/`
- `class_labels_indices.csv` ← `metadata/`
- `preprocess.py` ← `models/preprocess.py` (yeniden yazıldı, aşağıya bakın)

## Yapılan değişiklikler
- `from models.mn.x` → `from .x` (paket içi göreli import)
- `helpers/utils.py`'deki `NAME_TO_WIDTH` → `mn/utils.py` sonuna taşındı
- `mn/model.py`: `model_dir` varsayılanı `models`; `get_model` içindeki `print(m)` kapatıldı
- `preprocess.py`: torchaudio bağımlılığı kaldırıldı (Kaldi mel filtre bankası saf torch ile yazıldı), eğitim augmentasyonları çıkarıldı

Bu klasördeki dosyaları elle düzenlemeyin; proje kodu `detectors/audio_cls.py` içinde.
