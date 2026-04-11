import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle

st.title("🖼️ Image Caption Generator")

# Load model and tokenizer
model = load_model("vgg16_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating..."):
            # your caption generation logic here
            st.success("Caption: **your generated caption here**")