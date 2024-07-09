import streamlit as st
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# Memuat model Keras dari file HDF5
model = load_model('model.h5')

# Memuat LabelEncoder dari file pickle
with open('labelencoder.pkl', 'rb') as file:
    labelencoder = pickle.load(file)

# Fungsi untuk memprediksi genre dari file audio
def predict_genre(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0).reshape(1, -1)

    predicted_label = model.predict(mfccs_scaled_features)
    predicted_label = np.argmax(predicted_label, axis=1)
    prediction_class = labelencoder.inverse_transform(predicted_label)
    return prediction_class[0]

# Aplikasi Streamlit
st.title("Klasifikasi Genre Musik")

uploaded_file = st.file_uploader("Pilih File Musik ", type="wav")
if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file, format='audio/wav')
    
    prediction = predict_genre("temp.wav")
    st.write(f"Prediksi Genre: **{prediction}**")
