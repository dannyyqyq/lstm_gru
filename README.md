# 📝 Next Word Prediction with LSTM

A simple LSTM-based next-word predictor trained on Shakespeare's *Hamlet*, with a Streamlit app for interactive use.

This project uses a Long Short-Term Memory (LSTM) neural network to predict the next word in a text sequence. It’s trained on *Hamlet* from the NLTK Gutenberg corpus and deployed via a minimal Streamlit interface.

## 🚀 Features
- **Dataset**: *Hamlet* by Shakespeare (NLTK Gutenberg).
- **Model**: LSTM with embedding, dropout, and dense layers.
- **App**: Streamlit frontend for real-time predictions.

# 🔧 Installation & Setup

### Prerequisites
- Python 3.12
- Git

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/dannyyqyq/lstm_gru
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the App Locally
```sh
streamlit run app.py
```

## 🧠 Model Details
- **Architecture**:
  - Embedding: 4,818 words → 100D vectors, input length = 13.
  - LSTM: 150 units (sequence output), Dropout 0.2, LSTM 100 units (final output).
  - Dense: 4,818 classes with softmax.
- **Training**: 20 epochs, Adam optimizer, ~16% training accuracy, ~6% validation accuracy.
- **Files**:
  - `app.py`: Streamlit app.
  - `next_word_lstm.keras`: Trained model.
  - `tokenizer.pkl`: Tokenizer object.
  - `training_notebook.ipynb`: Training script (optional).

## 📋 How it works:
- Input: "The sun is shining".
- Output: "The next word could be: <prediction>" (e.g., "and").

## ⚠️ Limitations
- Limited to *Hamlet*’s 4,818-word vocabulary; may predict `None` for out-of-vocab inputs.
- Low accuracy due to sparse data and task complexity.