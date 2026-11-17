import os
import re
import json
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Hate Speech Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Simple styling
# -------------------------------------------------
st.markdown("""
<style>
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .hate {
        background-color: #FFE5E5;
        color: #C0392B;
        border: 2px solid #E74C3C;
    }
    .non-hate {
        background-color: #E8F8F5;
        color: #1E8449;
        border: 2px solid #27AE60;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E3A5F;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        height: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load model, tokenizer and config (relative paths)
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path     = os.path.join(base_dir, "hate_speech_rnn_model.keras")
    tokenizer_path = os.path.join(base_dir, "tokenizer.joblib")
    config_path    = os.path.join(base_dir, "config.json")

    for path, name in [
        (model_path, "hate_speech_rnn_model.keras"),
        (tokenizer_path, "tokenizer.joblib"),
        (config_path, "config.json"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find '{name}'.\n"
                f"Expected at: {path}\n"
                f"Make sure this file is in the same folder as app.py"
            )

    model = load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    with open(config_path) as f:
        config = json.load(f)

    return model, tokenizer, config

# -------------------------------------------------
# Text cleaning (must match training)
# -------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------------------------
# Prediction function
# -------------------------------------------------
def predict(text, model, tokenizer, max_len, threshold=0.40):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    prob = float(model.predict(padded, verbose=0)[0][0])
    label = 1 if prob >= threshold else 0
    return label, prob, cleaned

# -------------------------------------------------
# Load everything once
# -------------------------------------------------
model, tokenizer, config = load_artifacts()
MAX_LEN   = config.get("MAX_LEN", 40)
THRESHOLD = config.get("THRESHOLD", 0.40)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.title("About the Model")
    st.markdown("""
    **Architecture**  
    `Embedding → Bidirectional LSTM → Dense → Sigmoid`

    **Improvements**
    - Bidirectional LSTM (instead of SimpleRNN)
    - Class weights for imbalance
    - Tunable decision threshold
    """)
    st.divider()
    st.caption("Built with TensorFlow + Streamlit")

# -------------------------------------------------
# Main UI
# -------------------------------------------------
st.markdown("## 🛡️ Hate Speech Classification")
st.markdown("Binary classifier (Hate Speech vs Non-Hate)")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area(
        "Enter a tweet / text:",
        height=150,
        placeholder="Paste any text here..."
    )

    if st.button("🔍 Analyze", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Running inference..."):
                label, prob, cleaned = predict(
                    user_input, model, tokenizer, MAX_LEN, THRESHOLD
                )

            st.markdown("#### Result")
            if label == 1:
                st.markdown(
                    f'<div class="result-box hate">⚠️ HATE SPEECH DETECTED<br>'
                    f'<small>Probability: {prob*100:.1f}% (threshold = {THRESHOLD})</small></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-box non-hate">✅ NON-HATE (Safe)<br>'
                    f'<small>Probability of hate: {prob*100:.1f}% (threshold = {THRESHOLD})</small></div>',
                    unsafe_allow_html=True
                )

            st.markdown("#### Probability of Hate Speech")
            st.progress(min(prob, 1.0))
            st.caption(f"Raw sigmoid output: **{prob:.4f}** | Decision threshold: **{THRESHOLD}**")

            with st.expander("Preprocessing details"):
                st.write("**Original:**")
                st.code(user_input)
                st.write("**Cleaned (fed to model):**")
                st.code(cleaned if cleaned else "(empty after cleaning)")

with col2:
    st.subheader("Quick Examples")
    examples = [
        "I love spending time with my family!",
        "You people are disgusting and should leave this country.",
        "Happy birthday to the best friend ever!",
        "I hate all of them, they don't belong here.",
        "This is such a beautiful sunny day outside.",
    ]
    for i, ex in enumerate(examples):
        if st.button(f"Example {i+1}", key=f"ex_{i}"):
            st.session_state["example"] = ex

    if "example" in st.session_state:
        st.info(st.session_state["example"])
        if st.button("Run this example"):
            label, prob, _ = predict(
                st.session_state["example"], model, tokenizer, MAX_LEN, THRESHOLD
            )
            if label == 1:
                st.error(f"⚠️ Hate Speech ({prob*100:.1f}%)")
            else:
                st.success(f"✅ Non-Hate ({(1-prob)*100:.1f}%)")

st.divider()
st.caption("Note: The original dataset is heavily imbalanced. Class weights + a lower threshold significantly improve hate-speech recall.")
