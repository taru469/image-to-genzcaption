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

def download_lfs_files():
    import urllib.request
    import json
    
    st.info("Downloading local model weights and features (approx. 200MB)... Please wait 1-2 minutes. ⏳")
    url = "https://github.com/SanKolisetty/Image-to-Caption-Generator.git/info/lfs/objects/batch"
    headers = {
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json",
    }
    payload = {
        "operation": "download",
        "transfers": ["basic"],
        "objects": [
            {"oid": "704c870a8c4ce540100a1559b3320a5126377d8182b5c37b6920e693b78c12c5", "size": 70898676},
            {"oid": "488071137b3ab5994d8b749db2c01031d4ccc61eb4888bc8afe2f35b43cae0f3", "size": 133064982}
        ]
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            for obj in res.get("objects", []):
                oid = obj.get("oid")
                download_url = obj.get("actions", {}).get("download", {}).get("href")
                filename = "Image_Caption_Generator.h5" if oid.startswith("704c") else "features.pickle"
                if not os.path.exists(filename) or os.path.getsize(filename) < 1000:
                    urllib.request.urlretrieve(download_url, filename)
    except Exception as e:
        raise RuntimeError(f"Failed to auto-download model files: {str(e)}")

@st.cache_resource
def load_local_models():
    # If tokenizer is missing, download it
    if not os.path.exists("tokenizer.pickle") or os.path.getsize("tokenizer.pickle") < 1000:
        url = "https://raw.githubusercontent.com/SanKolisetty/Image-to-Caption-Generator/main/tokenizer.pickle"
        import urllib.request
        urllib.request.urlretrieve(url, "tokenizer.pickle")

    # If model weights are missing or are pointer files, download them
    for filename in ["Image_Caption_Generator.h5", "features.pickle"]:
        if not os.path.exists(filename) or os.path.getsize(filename) < 1000:
            download_lfs_files()
            break

    # Lazy-load tf_keras for compatibility
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
        y_hat = model([image, seq], training=False)
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
    feature = vgg_model(img, training=False)
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

# Initialize session state variables
if "generated_caption" not in st.session_state:
    st.session_state.generated_caption = ""
if "history" not in st.session_state:
    st.session_state.history = []

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
                        model='gemini-2.5-flash',
                        contents=[prompt, image]
                    )
                    st.session_state.generated_caption = response.text.strip()
                    st.session_state.history.append(st.session_state.generated_caption)
                    st.success("Here is your caption:")
                    st.subheader(st.session_state.generated_caption)
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
                st.session_state.generated_caption = genz_caption
                st.session_state.history.append(st.session_state.generated_caption)
                st.success("Here is your caption:")
                st.subheader(st.session_state.generated_caption)
            except Exception as e:
                st.error(f"Failed to load or execute local model: {str(e)}")
