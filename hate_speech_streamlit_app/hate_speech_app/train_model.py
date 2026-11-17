import re
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ----------------- Hyperparameters -----------------
VOCAB_SIZE   = 12000
MAX_LEN      = 50
EMBED_DIM    = 128
LSTM_UNITS   = 64
EPOCHS       = 12
BATCH_SIZE   = 64

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Loading data...")
df = pd.read_csv("twitter.csv")   # make sure this file is in the same folder

# ★★★ KEY CHANGE ★★★
# Treat both Hate Speech (0) and Offensive Language (1) as positive class
df['label'] = df['class'].apply(lambda x: 1 if x in [0, 1] else 0)

df['tweet'] = df['tweet'].apply(clean_text)

print("\nLabel distribution after change:")
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(
    df['tweet'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)

print("\nTokenizing...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train),
                            maxlen=MAX_LEN, padding='post', truncating='post')
X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test),
                            maxlen=MAX_LEN, padding='post', truncating='post')

# Class weights still help
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

print("\nBuilding model...")
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, mask_zero=True),
    Bidirectional(LSTM(LSTM_UNITS)),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy', 'Precision', 'Recall'])

model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]

print("\nTraining...")
history = model.fit(
    X_train_seq, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("\nEvaluating...")
loss, acc, prec, rec = model.evaluate(X_test_seq, y_test, verbose=0)
print(f"Test Accuracy : {acc*100:.2f}%")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall   : {rec:.4f}")

print("\nSaving artifacts...")
model.save("hate_speech_rnn_model.keras")
joblib.dump(tokenizer, "tokenizer.joblib")

with open("config.json", "w") as f:
    json.dump({
        "VOCAB_SIZE": VOCAB_SIZE,
        "MAX_LEN": MAX_LEN,
        "EMBED_DIM": EMBED_DIM,
        "LSTM_UNITS": LSTM_UNITS,
        "THRESHOLD": 0.45
    }, f, indent=2)

print("\nDone! Files saved:")
print("  - hate_speech_rnn_model.keras")
print("  - tokenizer.joblib")
print("  - config.json")
