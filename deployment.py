import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os
import pickle
import random

# Optional import for Gemini GenAI SDK
try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Constants
max_length = 31

# Slang translation dictionary for local model
SLANG_MAP = {
    "man": "bro",
    "boy": "lil bro",
    "woman": "queen",
    "girl": "gurl",
    "people": "squad",
    "crowd": "gang",
    "dog": "doggo",
    "dogs": "floofs",
    "cat": "cato",
    "cats": "catos",
    "running": "speedrunning life",
    "walking": "strutting",
    "playing": "doing side quests",
    "eating": "devouring",
    "jumping": "sending it",
    "sitting": "chilling",
    "standing": "standing like an NPC",
    "wearing": "rocking",
    "shirt": "fit",
    "jacket": "fit",
    "clothes": "drip",
    "hat": "drip",
    "water": "wet aesthetic",
    "beach": "beach vibes",
    "pool": "pool vibe",
    "mountain": "nature era",
    "hill": "nature era",
    "forest": "nature era",
    "snow": "frosty drip",
}

SUFFIXES = [
    "fr fr 💯",
    "no cap 🧢",
    "sheesh 🥶",
    "it's giving main character energy ✨",
    "understands the assignment 📝",
    "aura points +1000 📈",
    "out here living their best life 💅",
    "we are so back 🗣️"
]

@st.cache_resource
def load_local_models():
    # Verify file sizes to prevent loading Git LFS pointer files
    for filename in ["Image_Caption_Generator.h5", "tokenizer.pickle"]:
        if os.path.exists(filename) and os.path.getsize(filename) < 1000:
            raise ValueError(
                f"The file '{filename}' appears to be a Git LFS pointer rather than the actual asset. "
                "Please install Git LFS (git-lfs) and run `git lfs pull` to download the actual model assets."
            )

    # Lazy-load tf_keras for compatibility with Keras 2 models in a Keras 3 environment
    import tf_keras as tfk
    from tf_keras.applications.vgg16 import VGG16
    from tf_keras.models import Model

    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    model = tfk.models.load_model("Image_Caption_Generator.h5")
    
    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
        
    return vgg_model, model, tokenizer

def ind_to_word(index, tokenizer):
    for word, ind in tokenizer.word_index.items():
        if index == ind:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    from tf_keras.preprocessing.sequence import pad_sequences
    capt = 'start'
    for i in range(max_length):
        seq = tokenizer.texts_to_sequences([capt])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        y_hat = model.predict([image, seq], verbose=0)
        y_hat = np.argmax(y_hat)
        word = ind_to_word(y_hat, tokenizer)
        if word is None:
            break
        capt += ' ' + word
        if word == 'end':
            break
    return capt

def gen_caption_image(img, vgg_model, model, tokenizer, max_length):   
    from tf_keras.applications.vgg16 import preprocess_input
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    feature = vgg_model.predict(img, verbose=0)
    y_pred = predict_caption(model, feature, tokenizer, max_length)
    return y_pred

def translate_to_genz(caption):
    words = caption.replace("start", "").replace("end", "").strip().split()
    translated_words = []
    for word in words:
        cleaned = word.lower().strip()
        if cleaned in SLANG_MAP:
            translated_words.append(SLANG_MAP[cleaned])
        else:
            translated_words.append(word)
    
    suffix = random.choice(SUFFIXES)
    return " ".join(translated_words) + " " + suffix

# Streamlit UI
st.title("Image to GenZ Caption Generator ⚡️")

mode = st.radio("Choose Mode", ["GenZ VLM API (Online / Accurate)", "Local Model (Offline / Flickr8k)"])

if mode == "GenZ VLM API (Online / Accurate)":
    if not HAS_GEMINI:
        st.error("Please install the Google GenAI library: `pip install google-genai`")
    
    api_key_input = st.text_input("Enter Gemini API Key (or set GEMINI_API_KEY env variable)", type="password")
    api_key = api_key_input or os.environ.get("GEMINI_API_KEY")
    
    img = st.file_uploader("Upload your Image")
    
    if img and st.button("Generate GenZ Caption"):
        if not api_key:
            st.warning("Please provide a Gemini API Key to run in Online Mode.")
        else:
            image = Image.open(img)
            st.image(image, caption="Uploaded Image")
            with st.spinner("Letting the AI cook... 🍳"):
                try:
                    client = genai.Client(api_key=api_key)
                    prompt = (
                        "Analyze this image and write a caption in modern Gen Z slang. "
                        "Use popular terms like 'no cap', 'fr fr', 'cooked', 'vibes', 'rizz', 'aura', 'slay', "
                        "'giving', 'doing side quests', 'npc', etc. Keep it short, punchy, and include 1-2 relevant emojis."
                    )
                    response = client.models.generate_content(
                        model='gemini-3.6-flash',
                        contents=[prompt, image]
                    )
                    st.success("Here is your caption:")
                    st.subheader(response.text.strip())
                except Exception as e:
                    st.error(f"Error calling Gemini API: {str(e)}")
                    # Diagnostic helper
                    try:
                        client_diag = genai.Client(api_key=api_key)
                        models = [m.name for m in client_diag.models.list()]
                        st.info(f"Available models for your API Key: {', '.join(models)}")
                    except Exception as diag_err:
                        st.warning(f"Could not list available models: {str(diag_err)}")

else:
    img = st.file_uploader("Upload your Image")
    if img and st.button("Generate GenZ Caption"):
        image = Image.open(img).convert('RGB')
        st.image(image, caption="Uploaded Image")
        with st.spinner("Running offline model..."):
            try:
                from tf_keras.preprocessing.image import img_to_array
                vgg_model, model, tokenizer = load_local_models()
                image_resized = ImageOps.fit(image, (224, 224), Image.LANCZOS)
                img_array = img_to_array(image_resized)
                raw_caption = gen_caption_image(img_array, vgg_model, model, tokenizer, max_length)
                genz_caption = translate_to_genz(raw_caption)
                st.success("Here is your caption:")
                st.subheader(genz_caption)
            except Exception as e:
                st.error(f"Failed to load or execute local model: {str(e)}")
