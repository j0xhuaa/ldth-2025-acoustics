import streamlit as st
from data_loader import load_data_to_waveform, waveform_to_logmel, SAMPLE_RATE, plot_log_mel_spectrograms

st.title("🎛️ Log-Mel Spectrogram Viewer")
st.info("This page is under construction. Visualizations coming soon!")

DATA_DIR = 'data/raw/train'
SAMPLE_RATE = 16000


waveforms, y, labels = load_data_to_waveform(DATA_DIR)
print(f'after: {type(waveforms)}')
log_mels = waveform_to_logmel(waveforms, SAMPLE_RATE)

spectrogram_figs = plot_log_mel_spectrograms(log_mels, SAMPLE_RATE, labels)

for fig in spectrogram_figs:
    st.pyplot(fig)
