import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


# Constants for spectrogram
N_MELS = 64         # Number of mel bands
HOP_LENGTH = 512    # Controls time resolution
N_FFT = 1024        # FFT window size

def waveform_to_logmel(waveform, sample_rate):
    """
    Take 1D waveform arrays and convert them into mel-scaled spectrograms
    """
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec