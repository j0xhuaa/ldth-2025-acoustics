import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st

# Constants
DATA_DIR = 'data'
SAMPLE_RATE = 16000
MAX_DURATION = 5.0
N_SAMPLES_DISPLAY = 5  # Limit to avoid crashing on large sets

# Title
st.title("Audio Sample Explorer with Log-Mel Spectrograms")

# Load audio data
@st.cache_data
def load_data():
    X, y, labels_map = [], [], {}
    for idx, label_name in enumerate(sorted(os.listdir(DATA_DIR))):
        label_path = os.path.join(DATA_DIR, label_name)
        if not os.path.isdir(label_path):
            continue

        labels_map[idx] = label_name

        for file in os.listdir(label_path):
            if not file.endswith(('.wav', '.mp3')):
                continue

            file_path = os.path.join(label_path, file)
            try:
                waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
                target_length = int(SAMPLE_RATE * MAX_DURATION)
                if len(waveform) > target_length:
                    waveform = waveform[:target_length]
                else:
                    waveform = np.pad(waveform, (0, target_length - len(waveform)))
                X.append(waveform)
                y.append(idx)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return np.array(X), np.array(y), labels_map

X, y, labels = load_data()

# UI: Sample selection
sample_idx = st.slider("Choose a sample index", 0, min(len(X)-1, N_SAMPLES_DISPLAY - 1), 0)

# Display info
label_name = labels[y[sample_idx]]
st.markdown(f"**Class:** `{label_name}`")

# Waveform plot
st.markdown("### Waveform")
fig_waveform, ax_waveform = plt.subplots()
librosa.display.waveshow(X[sample_idx], sr=SAMPLE_RATE, ax=ax_waveform)
ax_waveform.set(title=f"Waveform ({label_name})")
st.pyplot(fig_waveform)

# Log-Mel Spectrogram plot
st.markdown("### Log-Mel Spectrogram")
logmel = waveform_to_logmel(X[sample_idx], SAMPLE_RATE)
fig_spec, ax_spec = plt.subplots()
img = librosa.display.specshow(logmel, sr=SAMPLE_RATE, hop_length=512, x_axis='time', y_axis='mel', ax=ax_spec)
ax_spec.set(title=f"Log-Mel Spectrogram ({label_name})")
fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
st.pyplot(fig_spec)