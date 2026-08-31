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

def translate_to_genz(caption, intensity="Standard (Medium)"):
    import string
    words = caption.replace("start", "").replace("end", "").strip().split()
    translated_words = []
    
    chill_suffixes = ["", "fr 💯", "vibes ✨", "chilling 🌊"]
    standard_suffixes = SUFFIXES
    high_suffixes = SUFFIXES + [
        "absolute skibidi rizzler vibes fr 🥶",
        "cooked beyond belief 💀",
        "doing side quests at 3am 🗣️",
        "aura points +9999 📈"
    ]
    
    for word in words:
        left_punct = ""
        right_punct = ""
        
        while word and word[0] in string.punctuation:
            left_punct += word[0]
            word = word[1:]
            
        while word and word[-1] in string.punctuation:
            right_punct = word[-1] + right_punct
            word = word[:-1]
            
        cleaned = word.lower().strip()
        translated = word
        if cleaned in SLANG_MAP:
            if intensity == "Chill (Low)":
                if random.random() < 0.5:
                    translated = SLANG_MAP[cleaned]
            else:
                translated = SLANG_MAP[cleaned]
                
        final_word = left_punct + translated + right_punct
        translated_words.append(final_word)
        
        if intensity == "Max Aura (High)" and random.random() < 0.15:
            translated_words.append(random.choice(["literally", "fr fr", "no cap", "bruh"]))
            
    if intensity == "Chill (Low)":
        suffix = random.choice(chill_suffixes)
    elif intensity == "Max Aura (High)":
        suffix = random.choice(high_suffixes)
    else:
        suffix = random.choice(standard_suffixes)
        
    result = " ".join(translated_words)
    if suffix:
        result += " " + suffix
    return result

# Streamlit UI
st.title("Image to GenZ Caption Generator ⚡️")

# Initialize session state variables
if "generated_caption" not in st.session_state:
    st.session_state.generated_caption = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "active_image" not in st.session_state:
    st.session_state.active_image = None
if "last_file_key" not in st.session_state:
    st.session_state.last_file_key = ""

# Sidebar settings and history
st.sidebar.header("Settings ⚙️")
mode = st.sidebar.radio("Choose Mode", ["GenZ VLM API (Online / Accurate)", "Local Model (Offline / Flickr8k)"])
intensity = st.sidebar.select_slider("Select Slang Intensity", options=["Chill (Low)", "Standard (Medium)", "Max Aura (High)"], value="Standard (Medium)")

st.sidebar.markdown("---")
st.sidebar.subheader("Recent Captions 📜")
if st.session_state.history:
    for idx, hist_caption in enumerate(reversed(st.session_state.history)):
        st.sidebar.text_area(f"#{len(st.session_state.history) - idx}", value=hist_caption, height=80, key=f"hist_{idx}", disabled=True)
    if st.sidebar.button("Clear History"):
        st.session_state.history = []
        st.session_state.generated_caption = ""
        st.rerun()
else:
    st.sidebar.info("No generated captions yet.")

# Main area title and description
st.markdown("<h1 style='text-align: center;'>Image to GenZ Caption Generator ⚡️</h1>", unsafe_html=True)
st.write("Upload your image below and let the model cook up the perfect Gen Z slang caption for your social media posts.")

# Configuration input for online API mode in the main screen
api_key = None
if mode == "GenZ VLM API (Online / Accurate)":
    if not HAS_GEMINI:
        st.error("Please install the Google GenAI library: `pip install google-genai`")
    api_key_input = st.text_input("Enter Gemini API Key (or set GEMINI_API_KEY env variable)", type="password")
    api_key = api_key_input or os.environ.get("GEMINI_API_KEY")

# File uploader
img_file = st.file_uploader("Upload your Image", type=["png", "jpg", "jpeg", "webp"])

if img_file is not None:
    file_key = f"{img_file.name}_{img_file.size}"
    if st.session_state.last_file_key != file_key:
        st.session_state.last_file_key = file_key
        st.session_state.active_image = Image.open(img_file).convert("RGB")
        st.session_state.generated_caption = ""

if st.session_state.active_image is not None:
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Selected Image 🖼️")
        st.image(st.session_state.active_image, use_container_width=True)
        generate_btn = st.button("Generate GenZ Caption 🔥", use_container_width=True)
        
    with col2:
        st.subheader("Your Generated Caption ✨")
        
        if generate_btn:
            with st.spinner("Cooking the caption... 🍳"):
                if mode == "GenZ VLM API (Online / Accurate)":
                    if not api_key:
                        st.warning("Please provide a Gemini API Key to run in Online Mode.")
                    else:
                        try:
                            client = genai.Client(api_key=api_key)
                            if intensity == "Chill (Low)":
                                prompt_guidelines = (
                                    "Write a subtle, natural, laid-back caption with very light, natural Gen Z slang. "
                                    "Avoid over-the-top slang words. Keep it chill and clean. Use 0 or 1 emoji. "
                                    "Example caption style: 'just chilling on the beach today, immaculate vibes.'"
                                )
                            elif intensity == "Max Aura (High)":
                                prompt_guidelines = (
                                    "Write a heavy, high-energy, brainrot-infused Gen Z slang caption. "
                                    "Go completely all-out with intense slang terms. Use popular words like 'rizzler', 'skibidi', "
                                    "'gyatt', 'aura points', 'cooked', 'sus', 'let him cook', 'main character energy', 'fr fr', 'no cap'. "
                                    "Make it extremely funny, punchy, and use 2-3 emojis. "
                                    "Example caption style: 'bro is speedrunning life with +9999 aura points, absolute rizzler fr fr 💀🔥'"
                                )
                            else:  # Standard (Medium)
                                prompt_guidelines = (
                                    "Write a fun, punchy caption using classic Gen Z slang. "
                                    "Use popular terms like 'no cap', 'fr fr', 'vibes', 'rizz', 'aura', 'slay', 'giving', 'doing side quests', 'npc'. "
                                    "Balance the slang naturally so it is engaging but readable. Include 1-2 relevant emojis. "
                                    "Example caption style: 'lil bro is out here living their best life, no cap 🧢✨'"
                                )

                            prompt = (
                                f"You are a modern Gen Z content creator. Analyze the provided image and generate an appropriate caption.\n"
                                f"Formatting & Tone Guidelines:\n"
                                f"{prompt_guidelines}\n"
                                f"Make sure to output ONLY the final caption text. Do not add any conversational responses, prefixes, or tags."
                            )
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=[prompt, st.session_state.active_image]
                            )
                            st.session_state.generated_caption = response.text.strip()
                            if st.session_state.generated_caption not in st.session_state.history:
                                st.session_state.history.append(st.session_state.generated_caption)
                        except Exception as e:
                            st.error(f"Error calling Gemini API: {str(e)}")
                            try:
                                client_diag = genai.Client(api_key=api_key)
                                models = [m.name for m in client_diag.models.list()]
                                st.info(f"Available models for your API Key: {', '.join(models)}")
                            except Exception as diag_err:
                                st.warning(f"Could not list available models: {str(diag_err)}")
                else:
                    try:
                        from tf_keras.preprocessing.image import img_to_array
                        vgg_model, model, tokenizer = load_local_models()
                        image_resized = ImageOps.fit(st.session_state.active_image, (224, 224), Image.LANCZOS)
                        img_array = img_to_array(image_resized)
                        raw_caption = gen_caption_image(img_array, vgg_model, model, tokenizer, max_length)
                        genz_caption = translate_to_genz(raw_caption, intensity=intensity)
                        st.session_state.generated_caption = genz_caption
                        if st.session_state.generated_caption not in st.session_state.history:
                            st.session_state.history.append(st.session_state.generated_caption)
                    except Exception as e:
                        st.error(f"Failed to load or execute local model: {str(e)}")
        
        if st.session_state.generated_caption:
            st.info("💡 Click the copy icon on the top-right of the code block below to copy your caption!")
            st.code(st.session_state.generated_caption, language=None)
        else:
            st.write("Click 'Generate GenZ Caption 🔥' to start cooking!")
