# Image to GenZ Caption Generator ⚡️

Image to GenZ Caption Generator is a deep learning & visual LLM application that generates fun, slang-filled descriptions and captions for any given image.

---

## Features
- **GenZ VLM Mode (Online / Highly Accurate)**: Uses the Gemini 1.5 Flash Visual Language Model (VLM) API to describe arbitrary images, identify complex contexts, read text, and generate high-fidelity Gen Z captions with emojis.
- **Local Model Mode (Offline / Fallback)**: Uses a local VGG16 CNN encoder to extract image features and an LSTM neural network decoder (trained on Flickr8k) to predict base captions offline, translating them into slang using a local dictionary.

---

## Tech Stack
- **Deep Learning**: TensorFlow, Keras (VGG16 & LSTM)
- **Visual Large Language Model**: Google Gemini API via `google-generativeai`
- **UI Frontend**: Streamlit
- **Preprocessing & Helpers**: Pillow, NumPy, Pickle

---

## Installation

### Prerequisites
This project uses **Git LFS (Large File Storage)** to manage large model files (`Image_Caption_Generator.h5` and `features.pickle`). Before cloning the repository, please make sure you have Git LFS installed:
- **macOS**: `brew install git-lfs`
- **Windows / Linux**: Download from [git-lfs.github.com](https://git-lfs.github.com/)

Once installed, run:
```bash
git lfs install
```

### Steps

1. Clone the repository and navigate into the folder:
   ```bash
   git clone https://github.com/taru469/Image-to-GenZCaption.git
   cd Image-to-GenZCaption
   ```

2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set your Gemini API key in your environment variables:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   ```

---

## How to Run

Launch the Streamlit web application using:
```bash
streamlit run deployment.py
```
Open the local URL in your web browser, upload any image, and select your preferred mode to generate Gen Z captions!

---

## The GenZ Slang Translation Strategy

For the local model mode, a rule-based mapping transforms basic descriptions:
* **Nouns**: `boy` $\rightarrow$ `lil bro`, `woman` $\rightarrow$ `queen`, `dog` $\rightarrow$ `doggo`, etc.
* **Verbs**: `running` $\rightarrow$ `speedrunning life`, `playing` $\rightarrow$ `doing side quests`, `eating` $\rightarrow$ `devouring`.
* **Atmosphere**: Appends random popular slang endings like `fr fr 💯`, `no cap 🧢`, `sheesh 🥶`, or `aura points +1000 📈`.

For the online mode, the Gemini API is prompted with specialized system guidelines to compose custom slang contextually.
