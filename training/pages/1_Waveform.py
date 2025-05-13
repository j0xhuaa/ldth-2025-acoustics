import streamlit as st
import matplotlib.pyplot as plt
import librosa.display
from data_loader import load_data_to_waveform, SAMPLE_RATE, plot_waveforms

DATA_DIR = 'data/raw/train'
SAMPLE_RATE = 16000

st.title("📈 Waveform Viewer")

# Load waveforms and labels
waveforms, y, labels = load_data_to_waveform(DATA_DIR)

# Generate waveform figures
waveform_figs = plot_waveforms(waveforms, SAMPLE_RATE, labels, max_plots=5)

# Render each figure in Streamlit
for fig in waveform_figs:
    st.pyplot(fig)