import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Clear session and load model
tf.keras.backend.clear_session()
try:
    model = load_model('next_word_lstm.keras')
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load tokenizer
try:
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    st.success("Tokenizer loaded successfully")
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

# Function to predict next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    return tokenizer.index_word.get(predicted_word_index, None)

# Streamlit app
st.title("Next Word Prediction using LSTM")
text = st.text_input("Please enter a sequence of words", "The sun is shining")
if st.button("Predict Next Word"):
    max_sequence_length = model.input_shape[1] + 1  # 13 + 1 = 14
    prediction = predict_next_word(model, tokenizer, text, max_sequence_length)
    st.write(f"The next word could be: {prediction}")
