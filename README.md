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

1. Clone the repository and navigate into the folder:
   ```bash
   git clone <your-repo-link>
   cd IMG2TEXT
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
