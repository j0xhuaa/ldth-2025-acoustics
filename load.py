import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Constants
DATA_DIR = 'data'
SAMPLE_RATE = 16000  # Target sample rate (Hz)
MAX_DURATION = 5.0   # Max length in seconds to truncate/pad audio
N_SAMPLES_DISPLAY = 5


# Prepare containers
X = []  # list of waveforms
y = []  # list of labels
labels = {}  # label to index mapping

# create a path to each directory that is holiding different sound sample types
for idx, label_name in enumerate(sorted(os.listdir(DATA_DIR))):
    label_path = os.path.join(DATA_DIR, label_name)
    if not os.path.isdir(label_path):
        continue

    labels[label_name] = idx

    # get the file
    for file in os.listdir(label_path):
        if not file.endswith(('.wav', '.mp3')):
            continue
    
    file_path = os.path.join(label_path, file)
    
    try:
        waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        # truncate or pad to fixed length
        target_length = int(SAMPLE_RATE * MAX_DURATION)
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))

        X.append(waveform)
        y.append(idx)

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} audio samples.")
print("Class label mapping:", labels)

logmel_specs = [waveform_to_logmel(waveform, SAMPLE_RATE) for waveform in X]
logmel_specs = np.array(logmel_specs)

print(f"Shape of one spectrogram: {logmel_specs[0].shape}")
print(f"Shape of all spectrograms: {logmel_specs.shape}")

print(logmel_specs[1])