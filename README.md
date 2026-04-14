# Image Caption Generation 

This project demonstrates an end-to-end system for generating captions for images. The pipeline involves:
- Using the **VGG16** model for feature extraction.
- Building and training a **combination of LSTM and a neural network** for generating captions.
- Evaluating the model's performance using the **BLEU score**.
- Converting the generated captions to speech using the **gTTS** library.
- Developing a user-friendly **frontend with Streamlit** for interacting with the model.

## Features ✨
- Extract high-level features from images using the pre-trained VGG16 model.
- Generate natural language descriptions for images using an LSTM + NN architecture.
- Assess the accuracy and fluency of generated captions using the BLEU evaluation metric.
- Enable audio playback of the generated captions via gTTS.
- Provide a web-based interface for seamless interaction using Streamlit.

---

## Dataset 📊
This project uses the **Flickr8k dataset**, which contains 8,000 images with five captions each, available on [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k). Download the dataset and prepare it for training by extracting the images and captions.

---

## Installation 🛠️

### Requirements 📋
Make sure you have the following dependencies installed:

- Python 3.7+
- TensorFlow / Keras
- NumPy
- Pillow
- gTTS
- Streamlit

### Setup ⚙️
1. Clone this repository:
```bash
git clone mygithubid
cd Image-Caption-Generation-Project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the VGG16 weights (pre-trained on ImageNet). Most versions of TensorFlow/Keras will download these automatically during initialization.

4. Prepare your dataset:
   - Download the Flickr8k dataset from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k).
   - Ensure you have a directory containing images for captioning.
   - Provide corresponding captions for training in a suitable format (e.g., a CSV or JSON file linking image paths with captions).

---

## Project Structure 
Below is the list of files included in this repository:

- **`app.py`**: A Streamlit-based web application for generating captions and converting them to speech.
- **`image-caption-generation.ipynb`**: The Jupyter Notebook containing the model training and experimentation code.
- **`model.h5`**: The pre-trained model file used for caption generation.
- **`predicted_caption.mp3`**: Example audio file of a generated caption.
- **`requirements.txt`**: List of dependencies required for running the project.
- **`tokenizer.pkl`**: The tokenizer object serialized for preprocessing and decoding captions.

---

## Usage 

### Launching the Streamlit Application
Run the following command to launch the frontend application:
```bash
streamlit run app.py
```
This opens the application in your default web browser, allowing you to:
- Upload images.
- View the generated captions.
- Listen to the audio playback of the captions.

---

## Project Architecture
1. **Feature Extraction**:
   - Pre-trained VGG16 is used to extract feature vectors from images.
   - These features are passed into the caption generation model.

2. **Caption Generation**:
   - The LSTM network handles sequential data to generate a coherent sequence of words.
   - Combined with additional dense layers to improve accuracy.

3. **Caption Evaluation**:
   - BLEU score is used as the evaluation metric to quantify the similarity between the generated captions and reference captions.

4. **Caption-to-Speech**:
   - gTTS library converts the generated text captions into audio, providing an auditory representation.

5. **Frontend Interface**:
   - Streamlit provides an intuitive user interface for interacting with the model.

---

## Example Output 

### Predicted Caption
"dog is jumping into the water."

### Audio Playback
Upon generating the caption, the system converts it into speech for audio playback.

---

## Contributing 
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

---

## Acknowledgements 
- [VGG16](https://arxiv.org/abs/1409.1556) - Visual Geometry Group
- [BLEU Score](https://en.wikipedia.org/wiki/BLEU) - Bilingual Evaluation Understudy Score
- [gTTS](https://gtts.readthedocs.io/) - Google Text-to-Speech library
- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

## 🚀 Thanks
