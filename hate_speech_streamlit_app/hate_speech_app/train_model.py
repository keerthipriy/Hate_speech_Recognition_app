import re
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Hyperparameters (same as notebook)
VOCAB_SIZE = 5000
MAX_LEN = 25
EMBED_DIM = 64
RNN_UNITS = 64
EPOCHS = 10
BATCH_SIZE = 32

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Loading data...")
df = pd.read_csv("/home/workdir/artifacts/hate-speech-classification-rnn-main/imbalanced_data.csv")
df = df[['tweet', 'label']]
df['tweet'] = df['tweet'].apply(clean_text)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    df['tweet'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)

print("Tokenizing...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(
    tokenizer.texts_to_sequences(X_train),
    maxlen=MAX_LEN, padding='post', truncating='post'
)
X_test_seq = pad_sequences(
    tokenizer.texts_to_sequences(X_test),
    maxlen=MAX_LEN, padding='post', truncating='post'
)

print("Building model...")
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM),
    SimpleRNN(RNN_UNITS),
    Dense(1, activation='sigmoid'),
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training...")
history = model.fit(
    X_train_seq, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1,
)

print("Evaluating...")
loss, acc = model.evaluate(X_test_seq, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%  |  Test Loss: {loss:.4f}")

print("Saving model and tokenizer...")
model.save("/home/workdir/artifacts/hate_speech_app/hate_speech_rnn_model.keras")
joblib.dump(tokenizer, "/home/workdir/artifacts/hate_speech_app/tokenizer.joblib")

# Save preprocessing constants
import json
with open("/home/workdir/artifacts/hate_speech_app/config.json", "w") as f:
    json.dump({
        "VOCAB_SIZE": VOCAB_SIZE,
        "MAX_LEN": MAX_LEN,
        "EMBED_DIM": EMBED_DIM,
        "RNN_UNITS": RNN_UNITS
    }, f)

print("Done! Model and tokenizer saved.")
