import streamlit as st
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# =========================
# KONFIGURASI
# =========================
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]

MODEL_PATHS = {
    "CNN (From Scratch)": "models/cnn_flower_model.keras",
    "MobileNetV2 (Pretrained)": "models/mobilenetv2_flower_model.keras",
    "EfficientNetB0 (Pretrained)": "models/efficientnetb0_flower_model.keras"
}

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_selected_model(path):
    return load_model(path)

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(image, model_type):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)

    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)

    if model_type == "CNN (From Scratch)":
        img_array = img_array / 255.0
    elif model_type == "MobileNetV2 (Pretrained)":
        img_array = mobilenet_preprocess(img_array)
    elif model_type == "EfficientNetB0 (Pretrained)":
        img_array = efficientnet_preprocess(img_array)

    return img_array

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="Flower Classification",
    layout="centered"
)

st.title("Flower Classification System")
st.write(
    "Klasifikasi citra bunga Daisy, Dandelion, Rose, Sunflower, dan Tulip "
    "menggunakan CNN dan Transfer Learning."
)

# =========================
# PILIH MODEL
# =========================
model_choice = st.selectbox(
    "Pilih Model",
    list(MODEL_PATHS.keys())
)

model = load_selected_model(MODEL_PATHS[model_choice])

# =========================
# UPLOAD GAMBAR
# =========================
uploaded_file = st.file_uploader(
    "Upload gambar bunga",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Gambar Input")
        st.image(image, width=320)

    with col2:
        input_tensor = preprocess_image(image, model_choice)
        predictions = model.predict(input_tensor)

        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        st.subheader("Hasil Prediksi")
        st.write(f"Model: {model_choice}")
        st.write(f"Prediksi: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

    st.subheader("Probabilitas Tiap Kelas")
    for i, class_name in enumerate(CLASS_NAMES):
        st.progress(float(predictions[0][i]))
        st.write(f"{class_name}: {predictions[0][i]*100:.2f}%")

