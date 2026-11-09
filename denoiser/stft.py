"""STFT helpers matching the challenge baseline notebook."""

import numpy as np
from scipy import signal


def as_waveform(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty, finite mono waveform.")
    return x


def stft(x, sample_rate, n_fft, hop):
    x = as_waveform(x)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    return signal.stft(x, fs=sample_rate, window="hann", nperseg=n_fft,
                       noverlap=n_fft - hop, nfft=n_fft,
                       boundary="zeros", padded=True)


def istft(z, length, sample_rate, n_fft, hop):
    _, y = signal.istft(z, fs=sample_rate, window="hann", nperseg=n_fft,
                        noverlap=n_fft - hop, nfft=n_fft,
                        input_onesided=True, boundary=True)
    if len(y) < length:
        raise ValueError("Inverse STFT produced insufficient samples.")
    return as_waveform(y[:length])
