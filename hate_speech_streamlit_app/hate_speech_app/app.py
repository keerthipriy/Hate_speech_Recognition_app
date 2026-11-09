import re
import json
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page config
st.set_page_config(
    page_title="Hate Speech Classifier | Simple RNN",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a polished look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6A7A;
        text-align: center;
        margin-bottom: 2rem;
    }
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
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498DB;
        margin-bottom: 0.8rem;
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


@st.cache_resource
def load_artifacts():
    # Use relative paths so it works on any computer
    model = load_model("hate_speech_rnn_model.keras")
    tokenizer = joblib.load("tokenizer.joblib")
    with open("config.json") as f:
        config = json.load(f)
    return model, tokenizer, config
    config = json.load(f)
    return model, tokenizer, config

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text, model, tokenizer, max_len):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    prob = float(model.predict(padded, verbose=0)[0][0])
    label = 1 if prob >= 0.5 else 0
    return label, prob, cleaned

# Load model
model, tokenizer, config = load_artifacts()
MAX_LEN = config["MAX_LEN"]

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.title("About the Model")
    st.markdown("""
    **Architecture:**  
    `Embedding → SimpleRNN → Dense(sigmoid)`

    | Layer | Details |
    |-------|---------|
    | Embedding | vocab=5000, dim=64 |
    | SimpleRNN | 64 units |
    | Output | 1 unit (sigmoid) |

    **Training Settings:**
    - Loss: Binary Crossentropy
    - Optimizer: Adam
    - Epochs: 10
    - Batch Size: 32
    - Max sequence length: 25
    - Vocabulary: Top 5000 words

    **Dataset:**  
    31,962 tweets (imbalanced ~93:7)
    """)
    st.divider()
    st.markdown("**Preprocessing steps:**")
    st.markdown("""
    1. Lowercase conversion  
    2. Remove @mentions  
    3. Remove URLs  
    4. Keep only letters & spaces  
    5. Collapse extra whitespace
    """)
    st.divider()
    st.caption("Built with TensorFlow + Streamlit")

# Main content
st.markdown('<p class="main-header">🛡️ Hate Speech Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Binary classifier using a Simple Recurrent Neural Network (RNN)</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter a Tweet")
    user_input = st.text_area(
        "Type or paste a tweet here:",
        height=150,
        placeholder="e.g. I love this beautiful day! or any text you want to check...",
        label_visibility="collapsed"
    )

    analyze_btn = st.button("🔍 Analyze Tweet", type="primary")

    if analyze_btn:
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Running Simple RNN inference..."):
                label, prob, cleaned = predict(user_input, model, tokenizer, MAX_LEN)

            st.markdown("#### Result")
            if label == 1:
                st.markdown(
                    f'<div class="result-box hate">⚠️ HATE SPEECH DETECTED<br><small>Confidence: {prob*100:.1f}%</small></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-box non-hate">✅ NON-HATE (Safe)<br><small>Confidence: {(1-prob)*100:.1f}%</small></div>',
                    unsafe_allow_html=True
                )

            # Probability bar
            st.markdown("#### Probability of Hate Speech")
            st.progress(prob)
            st.caption(f"Model output (sigmoid): **{prob:.4f}**  |  Threshold = 0.5")

            with st.expander("View preprocessing details"):
                st.write("**Original text:**")
                st.code(user_input)
                st.write("**Cleaned text (fed to model):**")
                st.code(cleaned if cleaned else "(empty after cleaning)")

with col2:
    st.subheader("Quick Examples")
    examples = [
        ("I love spending time with my family on weekends!", 0),
        ("Happy birthday to the best friend ever!", 0),
        ("This is such a beautiful sunny day outside.", 0),
        ("You people are disgusting and should leave.", 1),
        ("I hate all of them, they don't belong here.", 1),
    ]

    for i, (ex, expected) in enumerate(examples):
        if st.button(f"Example {i+1}", key=f"ex_{i}"):
            st.session_state["example_text"] = ex

    if "example_text" in st.session_state:
        st.info(f"**Selected:** {st.session_state['example_text']}")
        if st.button("Run selected example"):
            user_input = st.session_state["example_text"]
            label, prob, cleaned = predict(user_input, model, tokenizer, MAX_LEN)
            if label == 1:
                st.error(f"⚠️ Hate Speech ({prob*100:.1f}%)")
            else:
                st.success(f"✅ Non-Hate ({(1-prob)*100:.1f}%)")

st.divider()

# Model performance section
st.subheader("📊 Model Performance (from original training)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Test Accuracy", "94.93%")
m2.metric("Hate F1-Score", "0.61")
m3.metric("Non-Hate F1", "0.97")
m4.metric("Macro F1", "0.79")

st.markdown("""
**Confusion Matrix (original run):**
|  | Predicted Non-Hate | Predicted Hate |
|--|--------------------|----------------|
| **Actual Non-Hate** | 5814 | 131 |
| **Actual Hate** | 193 | 255 |
""")

st.caption("Note: The dataset is heavily imbalanced (≈93% Non-Hate). Overall accuracy is high, but Hate-class recall is more important for real-world use.")
