"""EfficientAT AugmentMelSTFT'nin yalnızca çıkarım (eval) sürümü, torchaudio bağımlılığı olmadan.

Orijinal: https://github.com/fschmid56/EfficientAT/blob/main/models/preprocess.py (MIT)
OkulKalkan değişiklikleri:
- torchaudio kaldırıldı (pip'teki son torchaudio, kurulu torch sürümüyle uyumsuz; kurmak torch'u düşürür).
  `torchaudio.compliance.kaldi.get_mel_banks` aşağıda saf torch ile birebir yeniden yazıldı
  (vtln_warp_factor=1.0 durumu; orijinal kod da bunu kullanıyor).
- Eğitim augmentasyonları (frekans/zaman maskeleme, fmin/fmax kaydırma) çıkarıldı.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _mel_scale(freq: torch.Tensor) -> torch.Tensor:
    return 1127.0 * (1.0 + freq / 700.0).log()


def kaldi_mel_banks(num_bins: int, window_length_padded: int, sample_freq: float,
                    low_freq: float, high_freq: float) -> torch.Tensor:
    """torchaudio.compliance.kaldi.get_mel_banks(..., vtln_warp_factor=1.0)[0] ile aynı.
    Dönüş: (num_bins, window_length_padded // 2)"""
    num_fft_bins = window_length_padded // 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist
    fft_bin_width = sample_freq / window_length_padded
    mel_low = 1127.0 * math.log(1.0 + low_freq / 700.0)
    mel_high = 1127.0 * math.log(1.0 + high_freq / 700.0)
    mel_delta = (mel_high - mel_low) / (num_bins + 1)

    b = torch.arange(num_bins).unsqueeze(1)
    left_mel = mel_low + b * mel_delta
    center_mel = mel_low + (b + 1.0) * mel_delta
    right_mel = mel_low + (b + 2.0) * mel_delta

    mel = _mel_scale(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    return torch.clamp(torch.min(up_slope, down_slope), min=0.0)


class MelSTFT(nn.Module):
    def __init__(self, n_mels: int = 128, sr: int = 32000, win_length: int = 800, hopsize: int = 320,
                 n_fft: int = 1024, fmin: float = 0.0, fmax: float | None = None,
                 fmax_aug_range: int = 2000) -> None:
        super().__init__()
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2  # orijinaldeki varsayılan (15000 Hz @ 32 kHz)
        self.n_fft = n_fft
        self.hopsize = hopsize
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length, periodic=False), persistent=False)
        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)
        mel_basis = kaldi_mel_banks(n_mels, n_fft, sr, fmin, fmax)
        mel_basis = nn.functional.pad(mel_basis, (0, 1), mode="constant", value=0)
        self.register_buffer("mel_basis", mel_basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=True)
        x = x.real ** 2 + x.imag ** 2  # güç spektrumu (orijinalde (x**2).sum(-1), aynı sonuç)
        melspec = torch.matmul(self.mel_basis, x)
        melspec = (melspec + 0.00001).log()
        return (melspec + 4.5) / 5.0  # hızlı normalizasyon
