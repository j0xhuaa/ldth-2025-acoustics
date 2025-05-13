import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers, models, utils


# parameters
DATA_DIR = 'data/raw/train'
SAMPLE_RATE = 16000
DURATION = 2.0
N_MELS = 64

def load_data_to_waveform(data_dir):
    """
    Walks through all class folders in 'data_dir', loads the WAV files,
    and returns preprocessed spectrograms (X), labels (y), and label mappings.
    """
    X, y = [], []
    labels = {}
    label_idx = 0

    # creates path to data files DATA_DIR + folder label names, and skips files, keeping directorys
    for label_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(class_dir):
            continue

        # creates a label dict k:v (dir:idx)
        if label_name not in labels:
            labels[label_name] = label_idx
            label_idx += 1

        # iterate through each .wav file in the class directorys
        for file in os.listdir(class_dir):
            if not file.endswith('.wav'):
                continue

            # create a complete path to the .wav files
            path = os.path.join(class_dir, file)

            # turn the audio samples into waveform arrays
            waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

            # pad to consistent length
            waveform = pad_or_truncate(waveform, SAMPLE_RATE, DURATION)

            # store processed data and label
            X.append(waveform)
            y.append(labels[label_name])

    return np.array(X), np.array(y), labels




def pad_or_truncate(waveform, sr, duration):
    """
    Pads or cuts waveform to the specified duration,
    to ensure samples are all the same lenght.
    """

    target_len = int(sr * duration)
    if len(waveform) < target_len:
        return np.pad(waveform, (0, target_len - len(waveform)))
    return waveform[:target_len]


def plot_waveforms(waveforms, sample_rate, labels=None, max_plots=5):
    """
    Generate matplotlib figures of waveform plots.

    Args:
        waveforms (List or np.ndarray): List of 1D waveform arrays.
        sample_rate (int): Sampling rate used for the waveforms.
        labels (List[str], optional): Labels for each waveform.
        max_plots (int): Maximum number of waveform plots to return.

    Returns:
        List[matplotlib.figure.Figure]: List of waveform plot figures.
    """
    #TO DO identify what waveform we are looking at, and maybe even just write one for each sample type so we can compare on the page.
    num_plots = min(len(waveforms), max_plots)
    figs = []

    for i in range(num_plots):
        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(waveforms[i], sr=sample_rate, ax=ax)

        title = f"Waveform {i + 1}"
        ax.set_title(title)
        figs.append(fig)

    return figs



if __name__ == "__main__":
    X, y, labels = load_data_to_waveform(DATA_DIR)
    print(labels)
   
