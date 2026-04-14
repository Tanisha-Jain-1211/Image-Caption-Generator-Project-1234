import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

st.title("🖼️ Image Caption Generator")

# ---- Load models (cached so they don't reload on every interaction) ----
@st.cache_resource
def load_resources():
    caption_model = load_model("vgg16_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    # VGG16 for feature extraction
    vgg = VGG16()
    vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
    return caption_model, tokenizer, vgg

caption_model, tokenizer, vgg_model = load_resources()

# ---- Helper: extract image features ----
def extract_features(image: Image.Image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    return feature

# ---- Helper: convert index to word ----
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# ---- Helper: generate caption ----
def generate_caption(model, tokenizer, photo, max_length=35):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    # Clean up start token
    caption = in_text.replace('startseq', '').strip()
    return caption

# ---- UI ----
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            features = extract_features(image)
            caption = generate_caption(caption_model, tokenizer, features)
        st.success(f"Caption: **{caption}**")