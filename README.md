# Hate_speech_Recognition_app
A Deep Learning-based Hate Speech Detection system using NLP and Simple RNN. Classifies tweets into Hate Speech and Non-Hate Speech categories through text preprocessing, tokenization, and word embeddings. Built with Python, TensorFlow, Keras, and Streamlit for interactive real-time text classification.
# 🛡️ Hate Speech Detection Using Simple RNN

## 📌 Project Overview

This project implements a Deep Learning-based Hate Speech Detection system using Natural Language Processing (NLP) and a Simple Recurrent Neural Network (RNN).

The system analyzes user-provided text or tweets and classifies them into two categories:
- **Hate Speech (1)**
- **Non-Hate Speech (0)**

The model learns textual patterns using word embeddings and recurrent neural network layers. An interactive Streamlit application provides a simple interface for real-time predictions.

## 🎯 Objectives

- Identify potentially hateful content in social media text.
- Apply NLP techniques for text preprocessing.
- Build a binary text classification model using Simple RNN.
- Deploy the trained model through a Streamlit web application.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **NLP:** Text Cleaning, Tokenization, Sequence Padding
- **Model Architecture:** Embedding Layer, Simple RNN, Dense Layer
- **Libraries:** NumPy, Pandas, Scikit-learn, Joblib
- **Deployment:** Streamlit

## 🧠 Model Architecture

The model follows this architecture:

`Input Text → Tokenization → Padding → Embedding → Simple RNN → Dense (Sigmoid) → Prediction`

| Component | Configuration |
|---|---|
| Vocabulary Size | 5,000 |
| Maximum Sequence Length | 25 |
| Embedding Dimension | 64 |
| RNN Units | 64 |
| Output Layer | Dense (1 neuron) |
| Activation | Sigmoid |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Epochs | 10 |
| Batch Size | 32 |

## ⚙️ Text Preprocessing

The following preprocessing operations are performed:

1. Convert text to lowercase.
2. Remove user mentions (@username).
3. Remove URLs.
4. Remove special characters and numbers.
5. Remove extra whitespace.
6. Tokenize text into sequences.
7. Apply padding to maintain a fixed sequence length.

## 📊 Model Performance

The reported metrics from the original training run:

| Metric | Score |
|---|---:|
| Test Accuracy | 94.93% |
| Hate Speech F1-Score | 0.61 |
| Non-Hate Speech F1-Score | 0.97 |
| Macro F1-Score | 0.79 |

**Note:** The dataset is imbalanced, with approximately 93% Non-Hate and 7% Hate Speech samples. Therefore, accuracy alone does not represent the model's effectiveness. Hate Speech detection performance, especially recall and F1-score, is important.

## 🚀 Streamlit Application Features

- Interactive text input for tweet analysis.
- Real-time hate speech classification.
- Prediction confidence display.
- Probability visualization.
- Text preprocessing details.
- Quick example inputs.
- Model architecture and performance information.

## 📁 Project Structure

```text
Hate-Speech-Detection-Using-Simple-RNN/
│
├── app.py
├── train_model.py
├── imbalanced_data.csv
├── hate_speech_rnn_model.keras
├── tokenizer.joblib
├── config.json
├── requirements.txt
└── README.md
```

## 💻 Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Hate-Speech-Detection-Using-Simple-RNN
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

## 🔮 Future Enhancements

- Experiment with LSTM and GRU architectures.
- Improve minority-class detection using class weighting or resampling.
- Explore Transformer-based models such as BERT.
- Evaluate performance using precision, recall, F1-score, and confusion matrix.
- Extend classification to multilingual text.

## 👩‍💻 Author

**Keerthi Priya**

Interested in Artificial Intelligence, Machine Learning, Deep Learning, NLP, and Generative AI.

---

*This project was developed as a practical implementation of NLP-based binary text classification using Deep Learning.*
