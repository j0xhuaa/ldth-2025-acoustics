import streamlit as st

st.set_page_config(page_title="Home", layout="centered")
st.title("🏠 Home Page")
st.markdown("""
Welcome to your **Acoustic Audio Dashboard**!

Use the left-hand navigation to view:
- 📈 Raw Waveforms
- 🎛️ Log-Mel Spectrograms
""")