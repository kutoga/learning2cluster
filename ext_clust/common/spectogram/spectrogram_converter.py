"""
This file opens, converts and compresses the wav file into a usable spectrogram for our application.

Based on work of Lukic and Vogt.
"""
import librosa
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal


def spectrogram(wav_file):
    (rate, sig) = wav.read(wav_file)

    nperseg = 20 * rate / 1000
    for i in range(0, 12):
        n = 2 ** i
        if n >= nperseg:
            nfft = n
            break

    f, t, Sxx = signal.spectrogram(sig, fs=rate, window='hamming', nperseg=nperseg, noverlap=nperseg / 2,
                                   nfft=nfft, detrend=None, scaling='spectrum', return_onesided=True)

    for i in range(Sxx.shape[0]):
        for j in range(Sxx.shape[1]):
            Sxx[i, j] = hr_to_mel_spect(Sxx[i, j])

    for i in range(Sxx.shape[0]):
        for j in range(Sxx.shape[1]):
            Sxx[i, j] = dyn_range_compression(Sxx[i, j])

    return Sxx


def mel_spectrogram(wav_file):
    # Read out audio range and sample rate of wav file
    audio_range, sample_rate = librosa.load(path=wav_file, sr=None)
    nperseg = int(10 * sample_rate / 1000)

    # NOTE: nperseg MUST be an int before handing it over to liberosa's function
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_range, sr=sample_rate, n_fft=1024, hop_length=nperseg)

    # Compress the mel spectrogram to the human dynamic range
    for i in range(mel_spectrogram.shape[0]):
        for j in range(mel_spectrogram.shape[1]):
            mel_spectrogram[i, j] = dyn_range_compression(mel_spectrogram[i, j])

    return mel_spectrogram


def dyn_range_compression(x):
    return np.log10(1 + 10000 * x)


def hr_to_mel_spect(f):
    """Convert an array of frequency in Hz into mel."""
    return 1127.01048 * np.log10(f / 700 + 1)
